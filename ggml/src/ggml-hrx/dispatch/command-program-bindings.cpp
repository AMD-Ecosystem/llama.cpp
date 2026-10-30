#include "command-program-bindings.h"

#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static void mix_hash(uint64_t & hash, uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
}

}  // namespace

CommandProgramBinding make_command_program_binding(
    ValueId value, const ValueBufferBinding & buffer, ggml_type type,
    const std::array<int64_t, GGML_MAX_DIMS> & ne, const std::array<size_t, GGML_MAX_DIMS> & nb,
    bool empty_value) {
    CommandProgramBinding binding = { value, buffer.buffer, buffer.offset, buffer.length, buffer.identity,
                                      buffer.generation, buffer.capacity, buffer.host_data, buffer.weight,
                                      empty_value };
    binding.source_host_data = buffer.source_host_data;
    binding.source_offset = buffer.source_offset;
    binding.source_capacity = buffer.source_capacity;
    binding.source_type = type;
    binding.source_ne = ne;
    binding.source_nb = nb;
    return binding;
}

CommandProgramBindings CommandProgramBindings::from_value_map(const ValueMap & values) {
    std::vector<CommandProgramBinding> bindings;
    CommandProgramBindings             result;
    for (const ValueId id : values.external_value_ids()) {
        const Value * value = values.find(id);
        if (value == nullptr) {
            result.status.log("external value %d does not exist", id.value);
            continue;
        }
        const std::optional<ValueBufferBinding> buffer = values.resolve_buffer_binding(id);
        if (!buffer.has_value()) {
            result.status.log("external value %d is not bound", id.value);
            continue;
        }
        CommandProgramBinding binding = make_command_program_binding(
            value->id, *buffer, value->type, value->ne, value->nb, value->byte_count == 0);
        bindings.push_back(binding);
    }
    return from_bindings(std::move(bindings), result.status);
}

CommandProgramBindings CommandProgramBindings::from_bindings(std::vector<CommandProgramBinding> bindings,
                                                             const Status &                     errors) {
    CommandProgramBindings result;
    result.status.append(errors);
    result.bindings_ = std::move(bindings);
    for (const CommandProgramBinding & binding : result.bindings_) {
        if (binding.buffer == nullptr && binding.host_data == nullptr) {
            result.status.log("external value %d has a null binding", binding.value.value);
        }
        if (binding.length == 0 && !binding.empty_value) {
            result.status.log("external value %d has an empty binding", binding.value.value);
        }
    }
    return result;
}

const CommandProgramBinding * CommandProgramBindings::find(ValueId value) const {
    for (const CommandProgramBinding & binding : bindings_) {
        if (binding.value == value) {
            return &binding;
        }
    }
    return nullptr;
}

const CommandProgramBinding * CommandProgramBindings::find(const CommandBinding & request) const {
    for (const CommandProgramBinding & binding : bindings_) {
        const bool same_value = binding.value == request.value;
        const bool same_layout = binding.layout == request.layout;
        const bool native_layout = request.layout == kNativeWeightLayout;
        const bool same_transformation = native_layout ||
            (binding.layout_source_type == request.source_type && binding.input_size == request.input_size &&
             binding.output_size == request.output_size && binding.source_length == request.source_length &&
             binding.length == request.length);
        const bool matches_request = same_value && same_layout && same_transformation;
        if (matches_request) {
            return &binding;
        }
    }
    return nullptr;
}

CommandProgramBindingsHash command_program_bindings_hash(const CommandProgramBindings & bindings) {
    uint64_t hash = UINT64_C(1469598103934665603);
    mix_hash(hash, UINT64_C(0x6872782d62696e64));
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        mix_hash(hash, static_cast<uint64_t>(static_cast<int64_t>(binding.value.value)));
        mix_hash(hash, binding.host_data != nullptr ? 1 : 0);
        mix_hash(hash, binding.identity);
        mix_hash(hash, binding.tracks_content_generation() ? binding.generation : 0);
        mix_hash(hash, static_cast<uint64_t>(binding.capacity));
        mix_hash(hash, static_cast<uint64_t>(binding.offset));
        mix_hash(hash, static_cast<uint64_t>(binding.length));
        mix_hash(hash, binding.weight ? 1 : 0);
        mix_hash(hash, binding.empty_value ? 1 : 0);
        mix_hash(hash, binding.source_host_data != nullptr ? 1 : 0);
        mix_hash(hash, static_cast<uint64_t>(binding.source_offset));
        mix_hash(hash, static_cast<uint64_t>(binding.source_capacity));
        mix_hash(hash, static_cast<uint64_t>(binding.source_type));
        for (size_t dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            mix_hash(hash, static_cast<uint64_t>(binding.source_ne[dimension]));
            mix_hash(hash, static_cast<uint64_t>(binding.source_nb[dimension]));
        }
    }
    mix_hash(hash, static_cast<uint64_t>(bindings.bindings().size()));
    return { hash };
}

CommandProgramBindingsFingerprint command_program_bindings_fingerprint(const CommandProgramBindings & bindings) {
    std::ostringstream out;
    out << "hrx-bindings-v1";
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        out << "|value=" << binding.value.value << "|kind=" << (binding.host_data != nullptr ? "host" : "device")
            << "|identity=" << binding.identity << "|generation="
            << (binding.tracks_content_generation() ? binding.generation : 0)
            << "|capacity=" << binding.capacity << "|offset=" << binding.offset << "|length=" << binding.length
            << "|weight=" << (binding.weight ? 1 : 0) << "|empty=" << (binding.empty_value ? 1 : 0)
            << "|source_host=" << (binding.source_host_data != nullptr ? 1 : 0)
            << "|source_offset=" << binding.source_offset << "|source_capacity=" << binding.source_capacity
            << "|source_type=" << static_cast<int>(binding.source_type);
        for (size_t dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            out << "|ne=" << binding.source_ne[dimension] << "|nb=" << binding.source_nb[dimension];
        }
    }
    return { out.str() };
}

}  // namespace ggml::hrx
