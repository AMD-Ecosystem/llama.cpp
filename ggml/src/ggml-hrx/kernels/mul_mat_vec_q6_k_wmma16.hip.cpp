#include "mul_mat_vec_q6_k_q8_1_common.hip.inc"

typedef _Float16 hrx_q6_k_wmma16_half16_vec __attribute__((ext_vector_type(16)));

static __device__ __forceinline__ uint32_t hrx_q6_k_wmma16_pack_f16x2(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q6_k_wmma16_unpack_f16x2(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ hrx_q6_k_wmma16_half16_vec hrx_q6_k_wmma16_duplicate_input(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = hrx_q6_k_wmma16_pack_f16x2(x0, x1);
    const uint32_t p1 = hrx_q6_k_wmma16_pack_f16x2(x2, x3);
    const uint32_t p2 = hrx_q6_k_wmma16_pack_f16x2(x4, x5);
    const uint32_t p3 = hrx_q6_k_wmma16_pack_f16x2(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_q6_k_wmma16_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = hrx_q6_k_wmma16_unpack_f16x2(s0, 0);
    result[9] = hrx_q6_k_wmma16_unpack_f16x2(s0, 1);
    result[10] = hrx_q6_k_wmma16_unpack_f16x2(s1, 0);
    result[11] = hrx_q6_k_wmma16_unpack_f16x2(s1, 1);
    result[12] = hrx_q6_k_wmma16_unpack_f16x2(s2, 0);
    result[13] = hrx_q6_k_wmma16_unpack_f16x2(s2, 1);
    result[14] = hrx_q6_k_wmma16_unpack_f16x2(s3, 0);
    result[15] = hrx_q6_k_wmma16_unpack_f16x2(s3, 1);
    return result;
}

static __device__ __forceinline__ _Float16 hrx_q6_k_wmma16_load_a_value(
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row,
        long long k_index,
        long long blocks_per_row) {
    const hrx_block_q6_K_q8_1_lhs * block = src0 + row * blocks_per_row + (k_index >> 8);
    const int in_block = static_cast<int>(k_index & 255);
    const int group = in_block >> 5;
    const float d =
        __half2float(__ushort_as_half(block->d)) *
        static_cast<float>(hrx_q6_k_scale(block, group, in_block & 31));
    return static_cast<_Float16>(d * static_cast<float>(hrx_q6_k_value(block, in_block)));
}

static __device__ __forceinline__ hrx_q6_k_wmma16_half16_vec hrx_q6_k_wmma16_load_a_row_major(
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row0,
        long long k0,
        long long rows,
        long long blocks_per_row,
        unsigned int lane) {
    const long long row = row0 + static_cast<long long>(lane & 15u);
    const long long k_base = k0 + static_cast<long long>(lane >> 4) * 8;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    if (row >= rows) {
        return hrx_q6_k_wmma16_duplicate_input(zero, zero, zero, zero, zero, zero, zero, zero);
    }
    return hrx_q6_k_wmma16_duplicate_input(
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 0, blocks_per_row),
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 1, blocks_per_row),
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 2, blocks_per_row),
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 3, blocks_per_row),
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 4, blocks_per_row),
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 5, blocks_per_row),
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 6, blocks_per_row),
        hrx_q6_k_wmma16_load_a_value(src0, row, k_base + 7, blocks_per_row));
}

static __device__ __forceinline__ hrx_q6_k_wmma16_half16_vec hrx_q6_k_wmma16_load_b_col_major(
        const float * src1,
        long long col0,
        long long k0,
        long long k,
        long long cols,
        unsigned int lane) {
    const long long col = col0 + static_cast<long long>(lane & 15u);
    const long long k_base = k0 + static_cast<long long>(lane >> 4) * 8;
    const _Float16 zero = static_cast<_Float16>(0.0f);
    if (col >= cols) {
        return hrx_q6_k_wmma16_duplicate_input(zero, zero, zero, zero, zero, zero, zero, zero);
    }
    const float * ptr = src1 + col * k + k_base;
    return hrx_q6_k_wmma16_duplicate_input(
        static_cast<_Float16>(ptr[0]),
        static_cast<_Float16>(ptr[1]),
        static_cast<_Float16>(ptr[2]),
        static_cast<_Float16>(ptr[3]),
        static_cast<_Float16>(ptr[4]),
        static_cast<_Float16>(ptr[5]),
        static_cast<_Float16>(ptr[6]),
        static_cast<_Float16>(ptr[7]));
}

static __device__ __forceinline__ void hrx_q6_k_wmma16_store_acc_f16_row_major(
        float * dst,
        long long rows_stride,
        long long row0,
        long long col0,
        long long rows,
        long long cols,
        hrx_q6_k_wmma16_half16_vec acc,
        unsigned int lane) {
    const long long row_base = row0 + static_cast<long long>(lane >> 4);
    const long long col = col0 + static_cast<long long>(lane & 15u);
    if (col >= cols) {
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const long long row = row_base + static_cast<long long>(i * 2);
        if (row < rows) {
            dst[col * rows_stride + row] = static_cast<float>(acc[i * 2]);
        }
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void hrx_mul_mat_vec_q6_k_wmma16x16_f16acc_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 31u;
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 16;
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 16;
    if (row0 >= rows || col0 >= cols) {
        return;
    }

    const long long blocks_per_row = k / 256;
    hrx_q6_k_wmma16_half16_vec acc = {};

    for (long long k0 = 0; k0 < k; k0 += 16) {
        const hrx_q6_k_wmma16_half16_vec a =
            hrx_q6_k_wmma16_load_a_row_major(src0, row0, k0, rows, blocks_per_row, lane);
        const hrx_q6_k_wmma16_half16_vec b =
            hrx_q6_k_wmma16_load_b_col_major(src1, col0, k0, k, cols, lane);
        acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc, false);
    }

    hrx_q6_k_wmma16_store_acc_f16_row_major(dst, rows, row0, col0, rows, cols, acc, lane);
}
