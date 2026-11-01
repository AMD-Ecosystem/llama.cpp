#pragma once

#include "dispatch/command-plan.h"
#include "ggml.h"
#include "graph/value-map.h"

#include <array>
#include <cstddef>
#include <limits>

namespace ggml::hrx {

inline bool strided_f32_storage_span_bytes(const Value & value, size_t & byte_count) {
    if (value.type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const bool positive_stride = value.nb[i] != 0;
        const bool aligned_stride = value.nb[i] % sizeof(float) == 0;
        const bool supported_stride = positive_stride && aligned_stride;
        if (!supported_stride) {
            return false;
        }
    }

    size_t max_offset = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0) {
            return false;
        }
        const size_t extent = static_cast<size_t>(value.ne[i] - 1);
        if (extent != 0 && value.nb[i] > std::numeric_limits<size_t>::max() / extent) {
            return false;
        }
        const size_t dim_offset = extent * value.nb[i];
        if (max_offset > std::numeric_limits<size_t>::max() - dim_offset) {
            return false;
        }
        max_offset += dim_offset;
    }
    if (max_offset > std::numeric_limits<size_t>::max() - sizeof(float)) {
        return false;
    }

    byte_count = max_offset + sizeof(float);
    if (value.storage_offset > std::numeric_limits<size_t>::max() - byte_count) {
        return false;
    }
    return value.storage_offset + byte_count <= value.storage_byte_count;
}

inline bool strided_f32_storage_span_elements(const Value & value, size_t & element_count) {
    size_t byte_count = 0;
    if (!strided_f32_storage_span_bytes(value, byte_count) || byte_count % sizeof(float) != 0) {
        return false;
    }
    element_count = byte_count / sizeof(float);
    return true;
}

inline bool strided_f16_storage_span_bytes(const Value & value, size_t & byte_count) {
    if (value.type != GGML_TYPE_F16) {
        return false;
    }
    size_t max_offset = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const bool positive_extent = value.ne[i] > 0;
        const bool aligned_stride = value.nb[i] > 0 && value.nb[i] % sizeof(ggml_fp16_t) == 0;
        if (!positive_extent || !aligned_stride) {
            return false;
        }
        const size_t extent = static_cast<size_t>(value.ne[i] - 1);
        if (extent != 0 && value.nb[i] > std::numeric_limits<size_t>::max() / extent) {
            return false;
        }
        const size_t dim_offset = extent * value.nb[i];
        if (max_offset > std::numeric_limits<size_t>::max() - dim_offset) {
            return false;
        }
        max_offset += dim_offset;
    }
    if (max_offset > std::numeric_limits<size_t>::max() - sizeof(ggml_fp16_t)) {
        return false;
    }
    byte_count = max_offset + sizeof(ggml_fp16_t);
    if (value.storage_offset > std::numeric_limits<size_t>::max() - byte_count) {
        return false;
    }
    return value.storage_offset + byte_count <= value.storage_byte_count;
}

struct F16AlternateView {
    DispatchBinding                       binding;
    std::array<size_t, GGML_MAX_DIMS>     strides = {};
};

inline bool find_f16_alternate_view(const Graph & graph,
                                    const CommandPlan & plan,
                                    const Value & source,
                                    F16AlternateView & view) {
    const Value * root = graph.values().find(source.storage_root);
    const bool packed_root = root != nullptr && root->type == GGML_TYPE_F32 && root->contiguous &&
                             root->storage_root == root->id && root->storage_offset == 0 &&
                             root->byte_count % sizeof(float) == 0;
    const bool source_is_f32_view = source.type == GGML_TYPE_F32 && source.nb[0] == sizeof(float) &&
                                    source.storage_offset % sizeof(float) == 0;
    const bool compatible_storage = packed_root && source_is_f32_view;
    if (!compatible_storage) {
        return false;
    }

    size_t source_span = 0;
    if (!strided_f32_storage_span_bytes(source, source_span)) {
        return false;
    }
    const size_t alternate_bytes = root->byte_count / sizeof(float) * sizeof(ggml_fp16_t);
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(plan, root->id, GGML_TYPE_F16, alternate_bytes);
    if (alternate == nullptr) {
        return false;
    }
    const size_t offset = source.storage_offset / sizeof(float) * sizeof(ggml_fp16_t);
    const size_t span = source_span / sizeof(float) * sizeof(ggml_fp16_t);
    const bool valid_offset = offset <= alternate_bytes;
    const bool valid_span = valid_offset && span <= alternate_bytes - offset;
    const bool in_bounds = valid_offset && valid_span;
    if (!in_bounds) {
        return false;
    }
    view.binding = { alternate->alternate_value, offset, span };
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        view.strides[i] = source.nb[i] / sizeof(float);
    }
    return true;
}

}  // namespace ggml::hrx
