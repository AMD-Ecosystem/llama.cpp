#pragma once

#include "backend-context.h"
#include "ggml-backend.h"
#include "graph/value-map.h"

#include <array>
#include <cstddef>

struct ggml_tensor;

// The buffer identity and byte range name the source independently of a resident HRX allocation.
// GGML's tensor metadata supplies the logical shape, including views into a shared import.
struct ggml_backend_hrx_imported_region {
    uint64_t                          identity;
    uint64_t                          generation;
    size_t                            offset;
    size_t                            length;
    ggml_type                         type;
    std::array<int64_t, GGML_MAX_DIMS> ne;
    std::array<size_t, GGML_MAX_DIMS>  nb;
};

ggml_backend_hrx_buffer_context * ggml_backend_hrx_buffer_context_from_buffer(ggml_backend_buffer_t buffer);
size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor);
void * ggml_backend_hrx_buffer_base(ggml_backend_buffer_t buffer);

bool ggml_backend_hrx_tensor_binding(const ggml_tensor *                tensor,
                                     ggml_backend_hrx_buffer_context ** out_context,
                                     size_t *                           out_offset);
bool ggml_backend_hrx_resolve_value_buffer(const ggml_tensor * tensor, ggml::hrx::ValueBufferBinding & binding);
bool ggml_backend_hrx_get_imported_region(const ggml_tensor * tensor, ggml_backend_hrx_imported_region & region);
// Only borrowed imports outside HRX-owned host storage may use alternative weight layouts.
bool ggml_backend_hrx_weight_has_repackable_source(const ggml_tensor * tensor);
bool ggml_backend_hrx_weight_layout_eligible(const ggml_tensor * tensor);
bool ggml_backend_hrx_get_retained_region(const ggml_tensor * tensor, ggml_backend_hrx_imported_region & region);
void ggml_backend_hrx_mark_imported_region_written(const ggml_tensor * tensor);
void ggml_backend_hrx_mark_retained_region_written(const ggml_tensor * tensor);
void ggml_backend_hrx_mark_foreign_host_region_written(const ggml_tensor * tensor);
void ggml_backend_hrx_foreign_host_backend_started();
void ggml_backend_hrx_foreign_host_backend_stopped();
