#include "backend-buffer-binding.h"

#include "ggml-backend-impl.h"
#include "ggml.h"

#include <mutex>
#include <unordered_map>

namespace {

std::mutex foreign_host_generations_mutex;
// GGML does not notify HRX when a foreign backend destroys a buffer. The source lifetime contract holds while an
// HRX backend is alive, and the table is retired with the last such backend.
std::unordered_map<uint64_t, uint64_t> foreign_host_generations;
size_t active_hrx_backends = 0;

uint64_t foreign_host_generation(uint64_t identity) {
    std::lock_guard<std::mutex> lock(foreign_host_generations_mutex);
    const auto found = foreign_host_generations.find(identity);
    return found == foreign_host_generations.end() ? 1 : found->second;
}

}  // namespace

void ggml_backend_hrx_foreign_host_backend_started() {
    std::lock_guard<std::mutex> lock(foreign_host_generations_mutex);
    ++active_hrx_backends;
}

void ggml_backend_hrx_foreign_host_backend_stopped() {
    std::lock_guard<std::mutex> lock(foreign_host_generations_mutex);
    GGML_ASSERT(active_hrx_backends > 0);
    --active_hrx_backends;
    if (active_hrx_backends == 0) {
        foreign_host_generations.clear();
    }
}

ggml_backend_hrx_buffer_context * ggml_backend_hrx_buffer_context_from_buffer(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

void * ggml_backend_hrx_buffer_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_buffer_context_from_buffer(buffer)->base;
}

bool ggml_backend_hrx_tensor_binding(const ggml_tensor *                tensor,
                                     ggml_backend_hrx_buffer_context ** out_context,
                                     size_t *                           out_offset) {
    if (tensor == nullptr) {
        return false;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr || buffer->iface.get_base != ggml_backend_hrx_buffer_base) {
        return false;
    }
    auto *       context = ggml_backend_hrx_buffer_context_from_buffer(buffer);
    const size_t offset  = ggml_backend_hrx_tensor_offset(context, tensor);
    const bool has_device_buffer = context->buffer != nullptr;
    const bool is_imported = context->imported;
    const bool has_storage = has_device_buffer || is_imported;
    const bool offset_fits = offset <= buffer->size;
    const bool tensor_size_fits = offset_fits ? ggml_nbytes(tensor) <= buffer->size - offset : false;
    const bool valid_binding = has_storage && offset_fits && tensor_size_fits;
    if (!valid_binding) {
        return false;
    }
    *out_context = context;
    *out_offset  = offset;
    return true;
}

bool ggml_backend_hrx_resolve_value_buffer(const ggml_tensor * tensor, ggml::hrx::ValueBufferBinding & binding) {
    ggml_backend_hrx_buffer_context * context = nullptr;
    size_t                            offset  = 0;
    if (!ggml_backend_hrx_tensor_binding(tensor, &context, &offset)) {
        if (tensor == nullptr) {
            return false;
        }
        const ggml_tensor *   root   = tensor->view_src != nullptr ? tensor->view_src : tensor;
        ggml_backend_buffer_t buffer = root->buffer;
        if (buffer == nullptr || !ggml_backend_buffer_is_host(buffer)) {
            return false;
        }
        void *       base     = ggml_backend_buffer_get_base(buffer);
        const size_t capacity = ggml_backend_buffer_get_size(buffer);
        if (base == nullptr || tensor->data == nullptr ||
            static_cast<const uint8_t *>(tensor->data) < static_cast<const uint8_t *>(base)) {
            return false;
        }
        const size_t host_offset =
            static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - static_cast<const uint8_t *>(base));
        if (host_offset > capacity || ggml_nbytes(tensor) > capacity - host_offset) {
            return false;
        }
        const uint64_t buffer_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer));
        const uint64_t base_address   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base));
        binding.host_data             = base;
        binding.offset                = host_offset;
        binding.length                = ggml_nbytes(tensor);
        binding.identity =
            buffer_address ^ (base_address + 0x9e3779b97f4a7c15ull + (buffer_address << 6) + (buffer_address >> 2));
        binding.generation = foreign_host_generation(binding.identity);
        binding.capacity   = capacity;
        binding.weight     = ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        // Standard GGUF weights are immutable, so their native copy can use the shared weight cache.
        // Layout eligibility remains a separate import-only decision.
        binding.source_host_data = base;
        binding.source_offset = host_offset;
        binding.source_capacity = capacity;
        return true;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    const bool is_imported = context->imported;
    const bool is_host_buffer = ggml_backend_buffer_is_host(buffer);
    const bool ordinary_device_buffer = !is_imported && !is_host_buffer;
    const bool directly_bindable = ordinary_device_buffer || context->direct_host_binding;
    // Coherent HRX host allocations are directly device-addressable. Represent them with an HRX buffer handle so
    // command-program preparation bypasses host materialization. Noncoherent host allocations remain host data.
    binding.buffer               = directly_bindable ? context->buffer : nullptr;
    const bool shared_host_region = context->retained_base != nullptr;
    binding.host_data            = directly_bindable ? nullptr :
                                   (shared_host_region ? context->retained_base : context->base);
    binding.offset               = shared_host_region ? context->physical_offset + offset : offset;
    binding.length               = ggml_nbytes(tensor);
    binding.identity             = context->identity;
    binding.generation           = ggml_backend_hrx_buffer_generation(context);
    binding.capacity             = shared_host_region ? context->physical_capacity : buffer->size;
    binding.weight = buffer != nullptr && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    const bool retained_host_source = is_imported || is_host_buffer;
    binding.source_host_data = retained_host_source ?
        (shared_host_region ? context->retained_base : context->base) : nullptr;
    binding.source_offset = binding.offset;
    binding.source_capacity = binding.capacity;
    return true;
}

bool ggml_backend_hrx_get_retained_region(const ggml_tensor * tensor, ggml_backend_hrx_imported_region & region) {
    ggml_backend_hrx_buffer_context * context = nullptr;
    size_t                            offset  = 0;
    const bool hrx_bound = ggml_backend_hrx_tensor_binding(tensor, &context, &offset);
    if (hrx_bound) {
        ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
        const bool retained = context->imported || ggml_backend_buffer_is_host(buffer);
        if (!retained) {
            return false;
        }
        region.identity   = context->identity;
        region.generation = ggml_backend_hrx_buffer_generation(context);
        region.offset     = context->retained_base != nullptr ? context->physical_offset + offset : offset;
    } else {
        ggml::hrx::ValueBufferBinding binding;
        const bool resolved = ggml_backend_hrx_resolve_value_buffer(tensor, binding);
        const bool has_host_source = resolved && binding.host_data != nullptr;
        if (!has_host_source) {
            return false;
        }
        region.identity = binding.identity;
        region.generation = binding.generation;
        region.offset = binding.offset;
    }
    region.length     = ggml_nbytes(tensor);
    region.type       = tensor->type;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        region.ne[i] = tensor->ne[i];
        region.nb[i] = tensor->nb[i];
    }
    return true;
}

void ggml_backend_hrx_mark_foreign_host_region_written(const ggml_tensor * tensor) {
    ggml_backend_hrx_imported_region region = {};
    if (!ggml_backend_hrx_get_retained_region(tensor, region)) {
        return;
    }
    std::lock_guard<std::mutex> lock(foreign_host_generations_mutex);
    uint64_t & generation = foreign_host_generations[region.identity];
    generation = generation == 0 ? 2 : generation + 1;
}

bool ggml_backend_hrx_get_imported_region(const ggml_tensor * tensor, ggml_backend_hrx_imported_region & region) {
    ggml_backend_hrx_buffer_context * context = nullptr;
    size_t offset = 0;
    const bool bound = ggml_backend_hrx_tensor_binding(tensor, &context, &offset);
    return bound && context->imported && ggml_backend_hrx_get_retained_region(tensor, region);
}

bool ggml_backend_hrx_weight_has_repackable_source(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const ggml_tensor * root = tensor->view_src != nullptr ? tensor->view_src : tensor;
    ggml_backend_buffer_t buffer = root->buffer;
    if (buffer == nullptr) {
        return false;
    }
    if (buffer->iface.get_base == ggml_backend_hrx_buffer_base) {
        const auto * context = ggml_backend_hrx_buffer_context_from_buffer(buffer);
        const bool borrowed_import = context->imported;
        const bool known_host_alias = context->retained_base != nullptr;
        const bool repackable_source = borrowed_import && !known_host_alias;
        return repackable_source;
    }
    return false;
}

bool ggml_backend_hrx_weight_layout_eligible(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return true;
    }
    const ggml_tensor * root = tensor->view_src != nullptr ? tensor->view_src : tensor;
    ggml_backend_buffer_t buffer = root->buffer;
    const bool is_weight = buffer != nullptr &&
                           ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    return !is_weight || ggml_backend_hrx_weight_has_repackable_source(tensor);
}
