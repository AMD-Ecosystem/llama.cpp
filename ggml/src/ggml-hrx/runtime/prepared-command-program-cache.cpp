#include "prepared-command-program-cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
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
