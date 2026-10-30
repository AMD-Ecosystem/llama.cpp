#pragma once

#include "graph/value-map.h"
#include "command-program.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include <vector>

namespace ggml::hrx {

struct CommandProgramBinding {
    // A buffer is directly bindable by an HRX command program. Host data requires residency or staging before
    // execution. These are alternate storage forms and should not both be populated.
    ValueId      value;
    hrx_buffer_t buffer     = nullptr;
    size_t       offset     = 0;
    size_t       length     = 0;
    uint64_t     identity   = 0;
    uint64_t     generation = 0;
    size_t       capacity   = 0;
    void *       host_data  = nullptr;
    bool         weight     = false;
    bool         empty_value = false;
    // Retained host bytes are separate from the address used to bind a command.
    const void * source_host_data = nullptr;
    size_t       source_offset    = 0;
    size_t       source_capacity  = 0;
    ggml_type    source_type      = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> source_ne = {};
    std::array<size_t, GGML_MAX_DIMS>  source_nb = {};
    // Materialized bindings are selected by the requesting command, not just by ValueId.
    std::string  layout = kNativeWeightLayout;
    ggml_type    layout_source_type = GGML_TYPE_COUNT;
    int64_t      input_size = 0;
    int64_t      output_size = 0;
    size_t       source_length = 0;

    bool requires_materialization() const { return host_data != nullptr; }
    // Mutable host staging refreshes at launch; retained weight content is prepared ahead of execution.
    bool tracks_content_generation() const { return weight && source_host_data != nullptr; }
};

CommandProgramBinding make_command_program_binding(
    ValueId value, const ValueBufferBinding & buffer, ggml_type type,
    const std::array<int64_t, GGML_MAX_DIMS> & ne, const std::array<size_t, GGML_MAX_DIMS> & nb, bool empty_value);

struct CommandProgramBindingsFingerprint {
    std::string value;
};

struct CommandProgramBindingsHash {
    uint64_t value = 0;
};

class CommandProgramBindings {
  public:
    static CommandProgramBindings from_value_map(const ValueMap & values);
    static CommandProgramBindings from_bindings(std::vector<CommandProgramBinding> bindings,
                                                const Status &                     errors = {});

    const CommandProgramBinding * find(ValueId value) const;
    const CommandProgramBinding * find(const CommandBinding & request) const;

    const std::vector<CommandProgramBinding> & bindings() const { return bindings_; }

    bool valid() const { return status.success(); }

    Status status;

  private:
    std::vector<CommandProgramBinding> bindings_;
};

CommandProgramBindingsHash        command_program_bindings_hash(const CommandProgramBindings & bindings);
CommandProgramBindingsFingerprint command_program_bindings_fingerprint(const CommandProgramBindings & bindings);

}  // namespace ggml::hrx
