#pragma once

#include "command-program-executor.h"
#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program.h"
#include "runtime/graph-replay.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

class GraphProgram;

uint64_t command_program_shape_hash(const std::string & command_shape);

struct PreparedCommandProgramCacheStats {
    uint64_t builds = 0;
    uint64_t hits   = 0;
};

struct PreparedCommandProgramCacheRetention {
    size_t entries = 0;
    size_t recorded_arenas = 0;
    size_t staging_bytes = 0;
    size_t constant_bytes = 0;
    size_t recorded_arena_bytes = 0;

    size_t retained_bytes() const { return staging_bytes + constant_bytes + recorded_arena_bytes; }
};

struct PreparedCommandProgramCacheEntry {
    std::mutex mutex;
    // Release recorded handles before the prepared bindings they reference.
    PreparedCommandProgram prepared;
    RecordedCommandGraph   recorded;
    bool                   ready = false;
};

struct PreparedCommandProgramCacheExecutionResult {
    bool                success = false;
    Status              status;
    HrxGraphReplayEvent graph_replay_event = HrxGraphReplayEvent::Disabled;
    std::string         graph_replay_ineligible_reason;
    uint64_t            graph_replay_build_ns                     = 0;
    uint64_t            graph_replay_launch_ns                    = 0;
    uint64_t            graph_replay_total_ns                     = 0;
    size_t              graph_replay_dispatches                   = 0;
    bool                graph_replay_transient_allocation_changed = false;
};

class PreparedCommandProgramCache {
  public:
    // Full physical bindings select the entry; a validated graph instance can retain it for warm execution.
    std::shared_ptr<PreparedCommandProgramCacheEntry> get_or_create(
        const CommandProgramExecutionContext & context,
        uint64_t graph_uid,
        uint64_t command_shape_hash,
        const CommandProgramBindings & bindings,
        const GraphProgram * owner = nullptr);

    PreparedCommandProgramCacheExecutionResult execute_with_result(
        const CommandProgramExecutionContext & context,
        const CommandProgram & commands,
        const CommandProgramBindings & bindings,
        const std::shared_ptr<PreparedCommandProgramCacheEntry> & entry);

    bool execute(const CommandProgramExecutionContext & context,
                 uint64_t                               graph_uid,
                 const std::string &                    command_shape,
                 const CommandProgram &                 commands,
                 const CommandProgramBindings &         bindings);

    PreparedCommandProgramCacheExecutionResult execute_with_result(const CommandProgramExecutionContext & context,
                                                                   uint64_t                               graph_uid,
                                                                   const std::string &                    command_shape,
                                                                   const CommandProgram &                 commands,
                                                                   const CommandProgramBindings &         bindings);

    PreparedCommandProgramCacheExecutionResult execute_with_result(const CommandProgramExecutionContext & context,
                                                                   uint64_t                               graph_uid,
                                                                   uint64_t                               command_shape_hash,
                                                                   const CommandProgram &                 commands,
                                                                   const CommandProgramBindings &         bindings);

    PreparedCommandProgramCacheStats stats() const;
    size_t entries() const;

    // Call while graph execution is serialized, before freeing completed cached work.
    PreparedCommandProgramCacheRetention retention() const;

    std::vector<std::shared_ptr<PreparedCommandProgramCacheEntry>> take_uid(uint64_t uid);

    void clear();

  private:
    struct Key {
        uint64_t graph_uid          = 0;
        uint64_t target_hash        = 0;
        uint64_t command_shape_hash = 0;
        uint64_t bindings_hash      = 0;

        bool operator==(const Key & other) const {
            return graph_uid == other.graph_uid && target_hash == other.target_hash &&
                   command_shape_hash == other.command_shape_hash && bindings_hash == other.bindings_hash;
        }
    };

    struct KeyHash {
        size_t operator()(const Key & key) const;
    };

    Key cache_key(uint64_t                               graph_uid,
                  const CommandProgramExecutionContext & context,
                  uint64_t                               command_shape_hash,
                  const CommandProgramBindings &         bindings) const;

    void record_build();
    void record_hit();

    mutable std::mutex                                       mutex_;
    std::unordered_map<Key, std::shared_ptr<PreparedCommandProgramCacheEntry>, KeyHash> programs_;
    PreparedCommandProgramCacheStats                         stats_;
};

}  // namespace ggml::hrx
