#pragma once

#include "ggml.h"
#include "graph/value-map.h"

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

}  // namespace ggml::hrx
