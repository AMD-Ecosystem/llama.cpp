#pragma once

#include "dispatch/command-program.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/command-program-executor.h"
#include "runtime/prepared-command-program-cache.h"
#include "status.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace ggml::hrx {

struct GraphProgramExternalBinding {
    ValueId             value;
    const ggml_tensor * tensor = nullptr;
};

enum class GraphProgramExternalSlotKind {
    Node,
    Source,
};

struct GraphProgramExternalSlot {
    ValueId                      value;
    GraphProgramExternalSlotKind kind         = GraphProgramExternalSlotKind::Node;
    size_t                       node_index   = 0;
    int32_t                      source_index = -1;
    bool                         weight_layout_eligible = true;
};

struct GraphProgramMatch {
    std::vector<GraphProgramExternalBinding> external_bindings;
    Status                                   status;

    bool valid() const { return status.success(); }
};

class GraphProgram {
  public:
    GraphProgram(uint64_t                        uid,
                 std::string                     target,
                 std::unique_ptr<Graph>          graph,
                 std::unique_ptr<CommandProgram> commands,
                 std::string                     command_shape);

    uint64_t uid() const { return uid_; }

    const std::string & target() const { return target_; }

    const std::string & command_shape() const { return command_shape_; }

    uint64_t command_shape_hash() const { return command_shape_hash_; }

    const Graph & graph() const { return *graph_; }

    Graph & graph() { return *graph_; }

    const CommandProgram & commands() const { return *commands_; }

    bool cross_shape_family() const { return commands_->cross_shape_kv_count != 0; }

    GraphProgramMatch match_current_graph(const ggml_cgraph & graph) const;
    GraphProgramMatch match_trusted_graph(const ggml_cgraph & graph, bool bind_external = true) const;
    GraphProgramMatch match_host_staging_graph(const ggml_cgraph & graph) const;
    Status capture_external_slots(const ggml_cgraph & graph, const GraphProgramMatch & match);

    bool can_use_prepared_fast_path(const ggml_cgraph & graph) const;
    bool mutable_sources_current() const;
    std::shared_ptr<PreparedCommandProgramCacheEntry> prepared_entry() const;
    void bind_prepared_graph(const ggml_cgraph & graph, const GraphProgramMatch & match,
                             const CommandProgramBindings & bindings,
                             std::shared_ptr<PreparedCommandProgramCacheEntry> entry);

  private:
    const GraphProgramExternalSlot * find_external_slot(ValueId value) const;
    const ggml_tensor *              resolve_external_slot(const ggml_cgraph &              graph,
                                                           const GraphProgramExternalSlot & slot,
                                                           Status &                         status) const;

    uint64_t                        uid_ = 0;
    std::string                     target_;
    std::unique_ptr<Graph>          graph_;
    std::unique_ptr<const CommandProgram> commands_;
    std::string                     command_shape_;
    uint64_t                        command_shape_hash_ = 0;

    std::vector<GraphProgramExternalSlot> external_slots_;
    struct MutableSourceWatch {
        const ggml_tensor * tensor;
        uint64_t            identity;
        size_t              offset;
        size_t              length;
    };
    std::vector<MutableSourceWatch> mutable_source_watches_;
    std::unordered_map<int32_t, size_t>   external_slot_by_value_;
    uint64_t                              fast_path_uid_ = 0;
    const ggml_tensor * const *           fast_path_nodes_ = nullptr;
    mutable std::mutex                    prepared_mutex_;
    std::shared_ptr<PreparedCommandProgramCacheEntry> prepared_entry_;
};

struct GraphProgramCacheStats {
    uint64_t builds                  = 0;
    uint64_t hits                    = 0;
    size_t entries                   = 0;
};

struct GraphProgramLookup {
    GraphProgram *                program = nullptr;
    std::unique_ptr<GraphProgram> uncached_program;
    GraphProgramMatch             match;
    Status                        status;

    bool valid() const { return program != nullptr && status.success() && match.valid(); }
};

struct GraphProgramSupportResult {
    bool   supported = false;
    Status status;

    bool valid() const { return supported && status.success(); }
};

class GraphProgramCache {
  public:
    bool can_execute(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target) const;

    GraphProgramSupportResult check_support(const ggml_cgraph &  graph,
                                            const KernelCorpus & corpus,
                                            const std::string &  target) const;

    GraphProgramLookup get_or_build(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target);
    GraphProgramLookup build_exact_uncached(const ggml_cgraph & graph, const KernelCorpus & corpus,
                                             const std::string & target) const;

    GraphProgramCacheStats stats() const;

    void clear();

  private:
    struct ValidatedGraphMatch {
        GraphProgram *              program = nullptr;
        const ggml_tensor * const * nodes   = nullptr;
    };

    GraphProgramLookup build_from_imported(const ggml_cgraph &  graph,
                                           Graph &&             imported_graph,
                                           const KernelCorpus & corpus,
                                           const std::string &  target);

    std::unique_ptr<GraphProgram> build_program_from_imported(uint64_t             uid,
                                                              Graph &&             imported_graph,
                                                              const KernelCorpus & corpus,
                                                              const std::string &  target,
                                                              Status &             errors,
                                                              bool                 allow_cross_shape = true) const;

    mutable std::mutex                                          mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<GraphProgram>> programs_;
    std::unordered_map<uint64_t, ValidatedGraphMatch>           validated_matches_;
    GraphProgram *                                              last_program_ = nullptr;
    GraphProgramCacheStats                                      stats_;
};

bool can_execute_standalone_op_as_graph(const ggml_tensor * op, const std::string & target);

}  // namespace ggml::hrx
