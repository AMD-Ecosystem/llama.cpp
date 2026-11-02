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

namespace ggml::hrx {

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

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

    const bool reusable_owner = lookup.program->can_use_prepared_fast_path(graph);
    const bool mutable_sources_current = reusable_owner && lookup.program->mutable_sources_current();
    const bool trusted_weights = reusable_owner && mutable_sources_current;
    GraphProgramMatch binding_match = std::move(lookup.match);
    if (reusable_owner) {
        binding_match = trusted_weights ? lookup.program->match_host_staging_graph(graph) :
                                          lookup.program->match_trusted_graph(graph, true);
    }
    if (!binding_match.valid()) {
        result.status.append(binding_match.status);
        return result;
    }

    CommandProgramBindings bindings = bind_external_value_buffers(binding_match);
    if (!bindings.valid()) {
        result.status.append(bindings.status);
        return result;
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
    } else if (graph.uid != 0) {
        entry = context_.prepared_programs.get_or_create(execution_context, lookup.program->uid(),
                                                        lookup.program->command_shape_hash(), bindings);
    }
    const PreparedCommandProgramCacheExecutionResult execution = entry != nullptr ?
        context_.prepared_programs.execute_with_result(execution_context, lookup.program->commands(), bindings, entry) :
        context_.prepared_programs.execute_with_result(execution_context, 0, lookup.program->command_shape_hash(),
                                                       lookup.program->commands(), bindings);
    result.graph_replay_event = execution.graph_replay_event;
    if (execution.execution_resources != nullptr) {
        // A failed return can still follow queued writes.
        lookup.program->mark_imported_writes(graph);
    }
    if (!execution.success) {
        result.status.append(execution.status);
        if (result.status.success()) {
            result.status.log("execute HRX command program failed");
        }
        return result;
    }

    const bool validated_entry = entry != nullptr && !trusted_weights;
    if (validated_entry) {
        lookup.program->bind_prepared_graph(graph, binding_match, std::move(entry));
    }

    result.code = GGML_STATUS_SUCCESS;
    trace.success = true;
    return result;
}

}  // namespace ggml::hrx
