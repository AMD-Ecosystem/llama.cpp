#include "graph-executor.h"

#include "backend-buffer-binding.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "hrx_runtime.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <utility>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

namespace ggml::hrx {

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

static CommandProgramBindings canonical_cross_shape_bindings(const CommandProgram & commands,
                                                               const CommandProgramBindings & physical) {
    std::unordered_map<int32_t, size_t> required_bytes;
    for (const Command & command : commands.commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != CommandBindingOrigin::GraphValue) {
                continue;
            }
            const size_t extent = binding.offset + binding.length;
            size_t & required = required_bytes[binding.value.value];
            required = std::max(required, extent);
        }
    }

    Status status;
    std::vector<CommandProgramBinding> normalized = physical.bindings();
    struct ReservedDeviceSpan {
        hrx_buffer_t buffer;
        uint64_t identity;
        size_t offset;
        size_t length;
    };
    std::vector<ReservedDeviceSpan> reserved_device_spans;
    for (const CommandProgramBinding & binding : normalized) {
        const auto found = required_bytes.find(binding.value.value);
        const bool has_requirement = found != required_bytes.end();
        const bool device_value = binding.buffer != nullptr && !binding.weight;
        const bool needs_reservation = has_requirement && found->second > binding.length;
        if (device_value && needs_reservation) {
            reserved_device_spans.push_back({ binding.buffer, binding.identity, binding.offset, found->second });
        }
    }
    for (CommandProgramBinding & binding : normalized) {
        const auto found = required_bytes.find(binding.value.value);
        if (binding.weight) {
            continue;
        }
        const size_t requested = found == required_bytes.end() ? 0 : found->second;
        size_t required = requested;
        for (const ReservedDeviceSpan & span : reserved_device_spans) {
            const bool same_storage = binding.buffer == span.buffer && binding.identity == span.identity;
            const bool same_offset = binding.offset == span.offset;
            if (same_storage && same_offset) {
                required = std::max(required, span.length);
            }
        }
        if (required <= binding.length) {
            continue;
        }
        const bool valid_offset = binding.offset <= binding.capacity;
        const bool sufficient_capacity = valid_offset && required <= binding.capacity - binding.offset;
        if (!sufficient_capacity) {
            status.log("cross-shape binding value %d requires %zu bytes but has %zu bytes capacity",
                       binding.value.value, required,
                       valid_offset ? binding.capacity - binding.offset : 0);
            continue;
        }
        binding.length = required;
    }
    return CommandProgramBindings::from_bindings(std::move(normalized), status);
}

Status GraphExecutor::context_valid_for_graph_programs() const {
    Status status;
    if (context_.device == nullptr) {
        status.log("missing HRX device context");
    } else if (context_.device->architecture.empty()) {
        status.log("missing HRX target");
    }
    return status;
}

Status GraphExecutor::context_valid_for_execution() const {
    return context_valid_for_graph_programs();
}

GraphSupportResult GraphExecutor::can_execute(const ggml_cgraph & graph) const {
    GraphSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    result.status = context_valid_for_graph_programs();
    if (!result.status.success()) {
        return result;
    }
    const KernelCorpus &            corpus = get_qwen_kernel_corpus();
    const GraphProgramSupportResult support =
        context_.graph_programs.check_support(graph, corpus, context_.device->architecture);
    result.supported = support.supported;
    result.status.append(support.status);
    return result;
}

CommandProgramBindings GraphExecutor::bind_external_value_buffers(const GraphProgramMatch & match) const {
    std::vector<CommandProgramBinding> bindings;
    Status                             status;
    bindings.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & external : match.external_bindings) {
        ValueBufferBinding    value_binding;
        if (ggml_backend_hrx_resolve_value_buffer(external.tensor, value_binding)) {
            bindings.push_back(make_command_program_binding(external.value, value_binding,
                                                            ggml_nbytes(external.tensor) == 0));
        } else {
            status.log("external value %d is not bound", external.value.value);
        }
    }
    return CommandProgramBindings::from_bindings(std::move(bindings), status);
}

GraphExecutionResult GraphExecutor::execute(const ggml_cgraph & graph) const {
    GraphExecutionResult result;
    CompileTraceExecution trace(context_.kernel_executables.trace_id(), graph.uid);
    if (graph.n_nodes == 0) {
        result.code = GGML_STATUS_SUCCESS;
        trace.success = true;
        return result;
    }
    result.status = context_valid_for_execution();
    if (!result.status.success()) {
        return result;
    }

    const KernelCorpus & corpus = get_qwen_kernel_corpus();
    GraphProgramLookup   lookup = context_.graph_programs.get_or_build(graph, corpus, context_.device->architecture);
    if (!lookup.valid()) {
        result.status.append(lookup.status);
        result.status.append(lookup.match.status);
        if (result.status.success()) {
            result.status.log("build HRX graph program failed");
        }
        return result;
    }

    bool family = lookup.program->cross_shape_family();
    const bool reusable_owner = !family && lookup.program->can_use_prepared_fast_path(graph);
    const bool mutable_sources_current = reusable_owner && lookup.program->mutable_sources_current();
    bool trusted_weights = reusable_owner && mutable_sources_current;
    GraphProgramMatch binding_match = std::move(lookup.match);
    if (reusable_owner) {
        binding_match = trusted_weights ? lookup.program->match_host_staging_graph(graph) :
                                          lookup.program->match_trusted_graph(graph, true);
    }
    if (!binding_match.valid()) {
        result.status.append(binding_match.status);
        return result;
    }

    CommandProgramBindings physical_bindings = bind_external_value_buffers(binding_match);
    if (!physical_bindings.valid()) {
        result.status.append(physical_bindings.status);
        return result;
    }
    CommandProgramBindings bindings = family ?
        canonical_cross_shape_bindings(lookup.program->commands(), physical_bindings) :
        std::move(physical_bindings);
    auto use_exact_fallback = [&]() {
        lookup = context_.graph_programs.build_exact_uncached(graph, corpus, context_.device->architecture);
        if (!lookup.valid()) {
            result.status.append(lookup.status);
            result.status.append(lookup.match.status);
            return false;
        }
        family = false;
        trusted_weights = false;
        binding_match = std::move(lookup.match);
        bindings = bind_external_value_buffers(binding_match);
        if (!bindings.valid()) {
            result.status.append(bindings.status);
            return false;
        }
        return true;
    };
    const bool force_exact = family && std::getenv("GGML_HRX_CROSS_SHAPE_FORCE_EXACT") != nullptr;
    const bool use_fallback = !bindings.valid() || force_exact;
    if (use_fallback) {
        if (!family) {
            result.status.append(bindings.status);
            return result;
        }
        if (!use_exact_fallback()) {
            return result;
        }
    }
    if (family && std::getenv("GGML_HRX_CROSS_SHAPE_TRACE") != nullptr) {
        std::fprintf(stderr, "HRX_CROSS_SHAPE_KEY kv=%d shape=%llu bindings=%llu count=%zu\n",
                     lookup.program->commands().cross_shape_kv_count,
                     static_cast<unsigned long long>(lookup.program->command_shape_hash()),
                     static_cast<unsigned long long>(command_program_bindings_hash(bindings).value),
                     bindings.bindings().size());
    }
    const CommandProgramExecutionContext execution_context = {
        context_.device->device,
        context_.stream,
        context_.device->architecture.c_str(),
        &corpus,
        &context_.kernel_executables,
        &context_.transient_arena,
        &context_.host_transfers,
        &context_.host_weights,
        &context_.device->host_buffers,
        &context_.graph_replay_state,
        trace.ids.cache_id != 0 ? &trace : nullptr,
    };
    std::shared_ptr<PreparedCommandProgramCacheEntry> entry;
    if (trusted_weights) {
        entry = lookup.program->prepared_entry();
    } else if (graph.uid != 0 && lookup.uncached_program == nullptr) {
        const uint64_t cache_uid = family ? UINT64_C(0x71753132352d6373) : lookup.program->uid();
        entry = family ?
            context_.prepared_programs.get_or_create_family(execution_context, cache_uid,
                                                            lookup.program->command_shape(), bindings) :
            context_.prepared_programs.get_or_create(execution_context, cache_uid,
                                                     lookup.program->command_shape_hash(), bindings);
        if (entry == nullptr && !use_exact_fallback()) {
            return result;
        }
    }
    const PreparedCommandProgramCacheExecutionResult execution = entry != nullptr ?
        context_.prepared_programs.execute_with_result(execution_context, lookup.program->commands(), bindings, entry) :
        context_.prepared_programs.execute_with_result(execution_context, 0, lookup.program->command_shape_hash(),
                                                       lookup.program->commands(), bindings);
    result.graph_replay_event = execution.graph_replay_event;
    if (!execution.success) {
        result.status.append(execution.status);
        if (result.status.success()) {
            result.status.log("execute HRX command program failed");
        }
        return result;
    }

    const bool validated_entry = entry != nullptr && !trusted_weights;
    if (validated_entry) {
        lookup.program->bind_prepared_graph(graph, binding_match, bindings, std::move(entry));
    }

    if (family && std::getenv("GGML_HRX_CROSS_SHAPE_TRACE") != nullptr) {
        const PreparedCommandProgramCacheRetention retained = context_.prepared_programs.retention();
        std::fprintf(stderr,
                     "HRX_CROSS_SHAPE_MEMORY kv=%d entries=%zu staging=%zu constants=%zu recorded_arenas=%zu\n",
                     lookup.program->commands().cross_shape_kv_count, retained.entries,
                     retained.staging_bytes, retained.constant_bytes, retained.recorded_arena_bytes);
    }

    result.code = GGML_STATUS_SUCCESS;
    trace.success = true;
    return result;
}

}  // namespace ggml::hrx
