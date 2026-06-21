#include "mul_mat_vec_q6_k_q8_1_common.hip.inc"

typedef _Float16 hrx_q6_k_id_wmma16_staged_half16_vec __attribute__((ext_vector_type(16)));
typedef _Float16 hrx_q6_k_id_wmma16_staged_half8_vec __attribute__((ext_vector_type(8)));

struct hrx_mul_mat_id_q6_k_wmma16_staged_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long route_capacity;
    long long src0_nb1;
    long long src0_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long dst_nb1;
    long long dst_nb2;
};

static __device__ __forceinline__ uint32_t hrx_q6_k_id_wmma16_staged_pack_f16x2(_Float16 lo, _Float16 hi) {
    union {
        _Float16 h[2];
        uint32_t u;
    } pack;
    pack.h[0] = lo;
    pack.h[1] = hi;
    return pack.u;
}

static __device__ __forceinline__ _Float16 hrx_q6_k_id_wmma16_staged_unpack_f16x2(uint32_t bits, int idx) {
    union {
        uint32_t u;
        _Float16 h[2];
    } pack;
    pack.u = bits;
    return pack.h[idx];
}

static __device__ __forceinline__ hrx_q6_k_id_wmma16_staged_half16_vec hrx_q6_k_id_wmma16_staged_duplicate_input(
        _Float16 x0, _Float16 x1, _Float16 x2, _Float16 x3,
        _Float16 x4, _Float16 x5, _Float16 x6, _Float16 x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = hrx_q6_k_id_wmma16_staged_pack_f16x2(x0, x1);
    const uint32_t p1 = hrx_q6_k_id_wmma16_staged_pack_f16x2(x2, x3);
    const uint32_t p2 = hrx_q6_k_id_wmma16_staged_pack_f16x2(x4, x5);
    const uint32_t p3 = hrx_q6_k_id_wmma16_staged_pack_f16x2(x6, x7);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_q6_k_id_wmma16_staged_half16_vec result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    result[4] = x4;
    result[5] = x5;
    result[6] = x6;
    result[7] = x7;
    result[8] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s0, 0);
    result[9] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s0, 1);
    result[10] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s1, 0);
    result[11] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s1, 1);
    result[12] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s2, 0);
    result[13] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s2, 1);
    result[14] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s3, 0);
    result[15] = hrx_q6_k_id_wmma16_staged_unpack_f16x2(s3, 1);
    return result;
}

static __device__ __forceinline__ const hrx_block_q6_K_q8_1_lhs * hrx_q6_k_id_wmma16_staged_block(
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row,
        long long k_index,
        const hrx_mul_mat_id_q6_k_wmma16_staged_constants & c,
        long long expert) {
    const char * expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    return reinterpret_cast<const hrx_block_q6_K_q8_1_lhs *>(
        expert_base + row * c.src0_nb1 + (k_index >> 8) * sizeof(hrx_block_q6_K_q8_1_lhs));
}

static __device__ __forceinline__ _Float16 hrx_q6_k_id_wmma16_staged_load_a_value(
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row,
        long long k_index,
        const hrx_mul_mat_id_q6_k_wmma16_staged_constants & c,
        long long expert) {
    const hrx_block_q6_K_q8_1_lhs * block =
        hrx_q6_k_id_wmma16_staged_block(src0, row, k_index, c, expert);
    const int in_block = static_cast<int>(k_index & 255);
    const int group = in_block >> 5;
    const float d =
        __half2float(__ushort_as_half(block->d)) *
        static_cast<float>(hrx_q6_k_scale(block, group, in_block & 31));
    return static_cast<_Float16>(d * static_cast<float>(hrx_q6_k_value(block, in_block)));
}

static __device__ __forceinline__ _Float16 hrx_q6_k_id_wmma16_staged_load_b_value(
        const float * src1,
        uint32_t route,
        long long k_index,
        const hrx_mul_mat_id_q6_k_wmma16_staged_constants & c) {
    const long long id = static_cast<long long>(route % static_cast<uint32_t>(c.n_ids));
    const long long token = static_cast<long long>(route / static_cast<uint32_t>(c.n_ids));
    const char * src1_base = reinterpret_cast<const char *>(src1) + id * c.src1_nb1 + token * c.src1_nb2;
    return static_cast<_Float16>(*reinterpret_cast<const float *>(src1_base + k_index * sizeof(float)));
}

static __device__ __forceinline__ hrx_q6_k_id_wmma16_staged_half16_vec hrx_q6_k_id_wmma16_staged_load_frag(
        const _Float16 * sh,
        int tile,
        int k_tile,
        unsigned int lane) {
    constexpr int SHARED_STRIDE = 48;
    const int row = tile * 16 + static_cast<int>(lane & 15u);
    const int k_base = k_tile * 16;
    hrx_q6_k_id_wmma16_staged_half16_vec result;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = sh[row * SHARED_STRIDE + k_base + i];
    }
    return result;
}

static __device__ __forceinline__ void hrx_q6_k_id_wmma16_staged_store_acc(
        float * dst,
        const uint32_t * expert_routes,
        uint32_t count,
        uint32_t route_index,
        long long row0,
        int row_tile,
        int col_tile,
        const hrx_mul_mat_id_q6_k_wmma16_staged_constants & c,
        hrx_q6_k_id_wmma16_staged_half8_vec acc,
        unsigned int lane) {
    const long long row_lane = static_cast<long long>(lane >> 4);
    const uint32_t col = static_cast<uint32_t>(col_tile * 16) + static_cast<uint32_t>(lane & 15u);
    const uint32_t active_route_index = route_index + col;
    if (active_route_index >= count) {
        return;
    }
    const uint32_t route = expert_routes[active_route_index];
    const long long id = static_cast<long long>(route % static_cast<uint32_t>(c.n_ids));
    const long long token = static_cast<long long>(route / static_cast<uint32_t>(c.n_ids));
    char * dst_base = reinterpret_cast<char *>(dst) + id * c.dst_nb1 + token * c.dst_nb2;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const long long row = row0 + static_cast<long long>(row_tile * 16) + row_lane + static_cast<long long>(reg * 4);
        if (row < c.rows) {
            *reinterpret_cast<float *>(dst_base + row * sizeof(float)) = static_cast<float>(acc[reg * 2]);
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void hrx_mul_mat_id_q6_k_wmma16x16_staged_vk64_f16acc_wg256_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q6_k_wmma16_staged_constants c) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int SHARED_STRIDE = 48;
    constexpr int WAVE = 64;
    constexpr int WAVE_COUNT = 4;
    constexpr int TILES_PER_WAVE = 4;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int wave = tid / WAVE;
    const unsigned int lane = tid & 63u;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const uint32_t route_base0 = static_cast<uint32_t>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const long long expert = static_cast<long long>(__builtin_amdgcn_workgroup_id_z());
    if (expert >= c.n_experts || row_base >= c.rows) {
        return;
    }

    const uint32_t count = counts[expert];
    if (route_base0 >= count) {
        return;
    }

    const uint32_t * expert_routes = routes + expert * c.route_capacity;
    const uint32_t route_tile_span = static_cast<uint32_t>(((c.n_tokens + BN - 1) / BN) * BN);
    const _Float16 zero = static_cast<_Float16>(0.0f);

    __shared__ _Float16 sh_a[BM * SHARED_STRIDE];
    __shared__ _Float16 sh_b[BN * SHARED_STRIDE];

    for (uint32_t route_base = route_base0; route_base < count; route_base += route_tile_span) {
        hrx_q6_k_id_wmma16_staged_half8_vec acc[TILES_PER_WAVE] = {};

        for (long long k0 = 0; k0 < c.k; k0 += BK) {
            for (int idx = static_cast<int>(tid); idx < BM * BK; idx += 256) {
                const int r = idx / BK;
                const int kk = idx - r * BK;
                const long long row = row_base + static_cast<long long>(r);
                sh_a[r * SHARED_STRIDE + kk] = row < c.rows ?
                    hrx_q6_k_id_wmma16_staged_load_a_value(src0, row, k0 + kk, c, expert) : zero;
            }
            for (int idx = static_cast<int>(tid); idx < BN * BK; idx += 256) {
                const int col = idx / BK;
                const int kk = idx - col * BK;
                const uint32_t active_route_index = route_base + static_cast<uint32_t>(col);
                _Float16 value = zero;
                if (active_route_index < count) {
                    value = hrx_q6_k_id_wmma16_staged_load_b_value(
                        src1, expert_routes[active_route_index], k0 + kk, c);
                }
                sh_b[col * SHARED_STRIDE + kk] = value;
            }
            __syncthreads();

            const int wave_row = static_cast<int>(wave & 1u);
            const int wave_col = static_cast<int>(wave >> 1);
#pragma unroll
            for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
                const int row_tile = wave_row * 2 + (tile_iter & 1);
                const int col_tile = wave_col * 2 + (tile_iter >> 1);
#pragma unroll
                for (int k_tile = 0; k_tile < 2; ++k_tile) {
                    const hrx_q6_k_id_wmma16_staged_half16_vec a =
                        hrx_q6_k_id_wmma16_staged_load_frag(sh_a, row_tile, k_tile, lane);
                    const hrx_q6_k_id_wmma16_staged_half16_vec b =
                        hrx_q6_k_id_wmma16_staged_load_frag(sh_b, col_tile, k_tile, lane);
                    acc[tile_iter] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a, b, acc[tile_iter], false);
                }
            }
            __syncthreads();
        }

        const int wave_row = static_cast<int>(wave & 1u);
        const int wave_col = static_cast<int>(wave >> 1);
#pragma unroll
        for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
            const int row_tile = wave_row * 2 + (tile_iter & 1);
            const int col_tile = wave_col * 2 + (tile_iter >> 1);
            hrx_q6_k_id_wmma16_staged_store_acc(
                dst, expert_routes, count, route_base, row_base, row_tile, col_tile, c, acc[tile_iter], lane);
        }
    }
}
