#include "prepared-command-program-cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unordered_set>
#include <utility>

namespace ggml::hrx {
namespace {

static void mix_hash(uint64_t & hash, uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
}

static uint64_t hash_text(const char * text) {
    uint64_t hash = UINT64_C(1469598103934665603);
    if (text == nullptr) {
        return hash;
    }
    while (*text != 0) {
        mix_hash(hash, static_cast<unsigned char>(*text));
        ++text;
    }
    return hash;
}

static void apply_graph_replay_result(PreparedCommandProgramCacheExecutionResult & result,
                                      const RecordedCommandGraphExecutionResult &  replay) {
    result.graph_replay_event                        = replay.event;
    result.graph_replay_ineligible_reason            = replay.ineligible_reason;
    result.graph_replay_build_ns                     = replay.build_ns;
    result.graph_replay_launch_ns                    = replay.launch_ns;
    result.graph_replay_total_ns                     = replay.total_ns();
    result.graph_replay_dispatches                   = replay.dispatch_count;
    result.graph_replay_transient_allocation_changed = replay.transient_allocation_changed;
}

static bool graph_replay_should_fallback(HrxGraphReplayEvent event) {
    return event == HrxGraphReplayEvent::Ineligible || event == HrxGraphReplayEvent::BuildFailed;
}

static bool family_binding_matches(const CommandProgramBinding & lhs, const CommandProgramBinding & rhs) {
    const bool same_value = lhs.value == rhs.value;
    const bool same_storage = lhs.buffer == rhs.buffer && lhs.host_data == rhs.host_data &&
                              lhs.source_host_data == rhs.source_host_data;
    const bool same_region = lhs.offset == rhs.offset && lhs.length == rhs.length &&
                             lhs.capacity == rhs.capacity;
    const bool same_identity = lhs.identity == rhs.identity && lhs.generation == rhs.generation;
    const bool same_kind = lhs.weight == rhs.weight && lhs.empty_value == rhs.empty_value;
    const bool same_layout = lhs.layout == rhs.layout && lhs.layout_source_type == rhs.layout_source_type &&
                             lhs.input_size == rhs.input_size && lhs.output_size == rhs.output_size &&
                             lhs.source_length == rhs.source_length;
    return same_value && same_storage && same_region && same_identity && same_kind && same_layout;
}

static bool family_bindings_match(const std::vector<CommandProgramBinding> & cached,
                                  const CommandProgramBindings & current) {
    if (cached.size() != current.bindings().size()) {
        return false;
    }
    for (size_t i = 0; i < cached.size(); ++i) {
        if (!family_binding_matches(cached[i], current.bindings()[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace

uint64_t command_program_shape_hash(const std::string & command_shape) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const char c : command_shape) {
        mix_hash(hash, static_cast<unsigned char>(c));
    }
    return hash;
}

size_t PreparedCommandProgramCache::KeyHash::operator()(const Key & key) const {
    uint64_t hash = UINT64_C(1469598103934665603);
    mix_hash(hash, key.graph_uid);
    mix_hash(hash, key.target_hash);
    mix_hash(hash, key.command_shape_hash);
    mix_hash(hash, key.bindings_hash);
    return static_cast<size_t>(hash);
}

PreparedCommandProgramCache::Key PreparedCommandProgramCache::cache_key(uint64_t graph_uid,
                                                                        const CommandProgramExecutionContext & context,
                                                                        uint64_t                       command_shape_hash,
                                                                        const CommandProgramBindings & bindings) const {
    return {
        graph_uid,
        hash_text(context.target),
        command_shape_hash,
        command_program_bindings_hash(bindings).value,
    };
}

bool PreparedCommandProgramCache::execute(const CommandProgramExecutionContext & context,
                                          uint64_t                               graph_uid,
                                          const std::string &                    command_shape,
                                          const CommandProgram &                 commands,
                                          const CommandProgramBindings &         bindings) {
    return execute_with_result(context, graph_uid, command_shape, commands, bindings).success;
}

PreparedCommandProgramCacheExecutionResult PreparedCommandProgramCache::execute_with_result(
    const CommandProgramExecutionContext & context,
    uint64_t                               graph_uid,
    const std::string &                    command_shape,
    const CommandProgram &                 commands,
    const CommandProgramBindings &         bindings) {
    return execute_with_result(context, graph_uid, command_program_shape_hash(command_shape), commands, bindings);
}

PreparedCommandProgramCacheExecutionResult PreparedCommandProgramCache::execute_with_result(
    const CommandProgramExecutionContext & context,
    uint64_t                               graph_uid,
    uint64_t                               command_shape_hash,
    const CommandProgram &                 commands,
    const CommandProgramBindings &         bindings) {
    PreparedCommandProgramCacheExecutionResult result;
    if (graph_uid == 0 || !commands.valid() || !bindings.valid()) {
        result.graph_replay_event             = HrxGraphReplayEvent::Ineligible;
        result.graph_replay_ineligible_reason = "uncached_graph";
        auto prepared = prepare_command_program(context, commands, bindings);
        if (!prepared.valid()) {
            result.status.append(prepared.status);
            return result;
        }
        // HRX command buffers retain queued buffers and executables even after this uncached owner is released.
        result.success = bind_and_execute_prepared_command_program(context, commands, bindings, prepared);
        if (!result.success) {
            result.status.log("execute uncached HRX command program failed");
        }
        return result;
    }

    return execute_with_result(context, commands, bindings,
                               get_or_create(context, graph_uid, command_shape_hash, bindings));
}

std::shared_ptr<PreparedCommandProgramCacheEntry> PreparedCommandProgramCache::get_or_create(
    const CommandProgramExecutionContext & context,
    uint64_t graph_uid,
    uint64_t command_shape_hash,
    const CommandProgramBindings & bindings) {
    const Key key = cache_key(graph_uid, context, command_shape_hash, bindings);
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = programs_.find(key);
    if (found != programs_.end()) {
        return found->second;
    }
    auto entry = std::make_shared<PreparedCommandProgramCacheEntry>();
    programs_.emplace(key, entry);
    return entry;
}

std::shared_ptr<PreparedCommandProgramCacheEntry> PreparedCommandProgramCache::get_or_create_family(
    const CommandProgramExecutionContext & context,
    uint64_t graph_uid,
    const std::string & command_shape,
    const CommandProgramBindings & bindings) {
    const Key key = cache_key(graph_uid, context, command_program_shape_hash(command_shape), bindings);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = programs_.find(key);
    if (found != programs_.end()) {
        const PreparedCommandProgramCacheEntry & entry = *found->second;
        const bool same_shape = entry.family_signature_set && entry.family_command_shape == command_shape;
        const bool same_bindings = same_shape && family_bindings_match(entry.family_bindings, bindings);
        return same_bindings ? found->second : nullptr;
    }
    auto entry = std::make_shared<PreparedCommandProgramCacheEntry>();
    entry->family_command_shape = command_shape;
    entry->family_bindings = bindings.bindings();
    entry->family_signature_set = true;
    programs_.emplace(key, entry);
    return entry;
}

PreparedCommandProgramCacheExecutionResult PreparedCommandProgramCache::execute_with_result(
    const CommandProgramExecutionContext & context,
    const CommandProgram & commands,
    const CommandProgramBindings & bindings,
    const std::shared_ptr<PreparedCommandProgramCacheEntry> & entry) {
    PreparedCommandProgramCacheExecutionResult result;
    std::lock_guard<std::mutex> entry_lock(entry->mutex);
    const bool replay_disabled = graph_replay_disabled_by_environment();
    const bool serial_execution = debug_serial_command_execution_enabled();
    const bool execute_directly = replay_disabled || serial_execution;
    if (!entry->ready) {
        auto prepared = prepare_command_program(context, commands, bindings);
        if (!prepared.valid()) {
            result.status.append(prepared.status);
            return result;
        }
        entry->prepared = std::move(prepared);
        entry->ready = true;
        record_build();
    } else {
        record_hit();
    }

    if (commands.cross_shape_kv_count != 0) {
        const auto argument = std::find_if(entry->prepared.program_constants.begin(),
                                           entry->prepared.program_constants.end(),
                                           [](const PreparedProgramConstantBuffer & constant) {
                                               return constant.name == "cross_shape_kv_count";
                                           });
        if (argument == entry->prepared.program_constants.end()) {
            result.status.log("missing cross-shape KV argument buffer");
            return result;
        }
        const int32_t active_kv = commands.cross_shape_kv_count;
        result.status.append(context.host_transfers->upload_async(context.stream, &active_kv,
                                                                   argument->buffer, 0, sizeof(active_kv)));
        if (!result.status.success()) {
            return result;
        }
    }

    if (execute_directly) {
        result.graph_replay_event = HrxGraphReplayEvent::Disabled;
        result.graph_replay_ineligible_reason = serial_execution ? "debug_serial_execution" :
                                                                  "disabled_by_environment";
        result.success = bind_and_execute_prepared_command_program(context, commands, bindings, entry->prepared);
    } else {
        const char * rebuild_setting = std::getenv("GGML_HRX_REBUILD_GRAPH_REPLAY");
        const bool rebuild_requested = rebuild_setting != nullptr && rebuild_setting[0] != '\0';
        const bool rebuild_enabled = rebuild_requested && rebuild_setting[0] != '0';
        if (rebuild_enabled) {
            entry->recorded = {};
        }
        const auto replay = bind_and_launch_recorded_command_graph(
            context, commands, bindings, entry->prepared, entry->recorded);
        apply_graph_replay_result(result, replay);
        if (commands.cross_shape_kv_count != 0 && std::getenv("GGML_HRX_CROSS_SHAPE_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "HRX_CROSS_SHAPE_REPLAY kv=%d entry=%p exec=%p event=%d build_ns=%llu launch_ns=%llu arena=%llu bytes=%zu\n",
                         commands.cross_shape_kv_count, static_cast<void *>(entry.get()),
                         static_cast<void *>(entry->recorded.exec), static_cast<int>(replay.event),
                         static_cast<unsigned long long>(replay.build_ns),
                         static_cast<unsigned long long>(replay.launch_ns),
                         static_cast<unsigned long long>(entry->recorded.bound_transient_arena_allocation_id),
                         entry->recorded.bound_transient_arena_capacity);
        }
        result.success = replay.success;
        const bool fallback = !replay.success && graph_replay_should_fallback(replay.event);
        if (fallback) {
            result.success = bind_and_execute_prepared_command_program(context, commands, bindings, entry->prepared);
        } else if (!replay.success) {
            result.status.append(replay.status);
        }
    }
    const bool missing_error = !result.success && result.status.success();
    if (missing_error) {
        result.status.log("execute prepared HRX command program failed");
    }
    return result;
}

PreparedCommandProgramCacheStats PreparedCommandProgramCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

PreparedCommandProgramCacheRetention PreparedCommandProgramCache::retention() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PreparedCommandProgramCacheRetention result;
    result.entries = programs_.size();
    std::unordered_set<uint64_t> recorded_arenas;
    for (const auto & item : programs_) {
        const PreparedCommandProgramCacheEntry & entry = *item.second;
        for (const HostStagingBuffer & staging : entry.prepared.host_staging) {
            result.staging_bytes += staging.length;
        }
        for (const PreparedProgramConstantBuffer & constant : entry.prepared.program_constants) {
            result.constant_bytes += constant.size;
        }
        const RecordedCommandGraph & recorded = entry.recorded;
        const bool has_arena = recorded.bound_transient_arena_allocation_id != kInvalidTransientArenaAllocationId;
        const bool first_reference = has_arena ?
            recorded_arenas.insert(recorded.bound_transient_arena_allocation_id).second : false;
        const bool count_arena = recorded.valid() && has_arena && first_reference;
        if (count_arena) {
            result.recorded_arena_bytes += recorded.bound_transient_arena_capacity;
        }
    }
    return result;
}

void PreparedCommandProgramCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    programs_.clear();
}

void PreparedCommandProgramCache::record_build() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.builds;
}

void PreparedCommandProgramCache::record_hit() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.hits;
}

}  // namespace ggml::hrx
