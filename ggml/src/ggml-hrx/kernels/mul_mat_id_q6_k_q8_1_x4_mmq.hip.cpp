#include "mul_mat_vec_q6_k_q8_1_common.hip.inc"

struct hrx_mul_mat_id_q6_k_grouped_constants {
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

struct hrx_q6_k_moe_a_cache {
    int qs[8];
    float d[2];
};

struct hrx_q8_1_q6_moe_b_cache {
    int qs[8];
    float d;
    uint32_t route;
};

struct hrx_q8_1_q6_moe_b_pending {
    int4 qs;
    unsigned short d;
    uint32_t route;
    bool valid;
};

static __device__ __forceinline__ const hrx_block_q6_K_q8_1_lhs * hrx_q6_k_moe_block(
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row,
        long long kb,
        long long src0_nb1,
        long long src0_nb2,
        long long expert) {
    const char * expert_base = reinterpret_cast<const char *>(src0) + expert * src0_nb2;
    return reinterpret_cast<const hrx_block_q6_K_q8_1_lhs *>(
        expert_base + row * src0_nb1 + (kb >> 3) * sizeof(hrx_block_q6_K_q8_1_lhs));
}

static __device__ __forceinline__ void hrx_q6_k_moe_mmq_load_a(
        hrx_q6_k_moe_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row,
        long long kb,
        int iqs,
        long long rows,
        long long src0_nb1,
        long long src0_nb2,
        long long expert) {
    if (row >= rows) {
        buf_a[buf_idx].qs[iqs] = 0;
        if (iqs == 0 || iqs == 4) {
            buf_a[buf_idx].d[iqs >> 2] = 0.0f;
        }
        return;
    }

    const hrx_block_q6_K_q8_1_lhs * block =
        hrx_q6_k_moe_block(src0, row, kb, src0_nb1, src0_nb2, expert);
    const int group = static_cast<int>(kb & 7);
    buf_a[buf_idx].qs[iqs] = hrx_q6_k_pack4(block, group, iqs);
    if (iqs == 0 || iqs == 4) {
        buf_a[buf_idx].d[iqs >> 2] =
            __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(hrx_q6_k_scale(block, group, iqs * 4));
    }
}

static __device__ __forceinline__ hrx_q8_1_q6_moe_b_pending hrx_q8_1_q6_moe_mmq_fetch_b(
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        const uint32_t * expert_routes,
        uint32_t count,
        uint32_t route_index,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    hrx_q8_1_q6_moe_b_pending pending = {};
    pending.valid = route_index < count;
    pending.route = pending.valid ? expert_routes[route_index] : 0u;
    if (!pending.valid) {
        return pending;
    }

    const long long linear_block = static_cast<long long>(pending.route) * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q6 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    pending.qs = *reinterpret_cast<const int4 *>(&rhs->qs[inner * 8 + iqs_vec4 * 4]);
    if (iqs_vec4 == 0) {
        pending.d = rhs->ds[inner * 2 + 0];
    }
    return pending;
}

static __device__ __forceinline__ void hrx_q8_1_q6_moe_mmq_commit_b(
        hrx_q8_1_q6_moe_b_cache * buf_b,
        int buf_idx,
        const hrx_q8_1_q6_moe_b_pending & pending,
        int iqs_vec4) {
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 0] = pending.valid ? pending.qs.x : 0;
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 1] = pending.valid ? pending.qs.y : 0;
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 2] = pending.valid ? pending.qs.z : 0;
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 3] = pending.valid ? pending.qs.w : 0;
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = pending.valid ? __half2float(__ushort_as_half(pending.d)) : 0.0f;
        buf_b[buf_idx].route = pending.route;
    }
}

template <int BN, int TN>
static __device__ __forceinline__ void hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq_wg64_impl(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q6_k_grouped_constants c) {
    constexpr int BM = 64;
    constexpr int BK_STEP = 1;
    constexpr int BLOCK_SIZE = 64;
    constexpr int WARP = 64;
    constexpr int WM = 64;
    constexpr int WN = BN;
    constexpr int WMITER = 1;
    constexpr int TM = 4;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;
    constexpr int LOAD_STRIDE_A = BLOCK_SIZE * LOAD_VEC_A / 32;
    constexpr int LOAD_STRIDE_B = BLOCK_SIZE * LOAD_VEC_B / 32;
    constexpr int LOADS_A = BM / LOAD_STRIDE_A;
    constexpr int LOADS_B = (BN + LOAD_STRIDE_B - 1) / LOAD_STRIDE_B;

    static_assert(BN == 16 || BN == 32 || BN == 64, "unexpected Q6 MoE MMQ route tile");
    static_assert(TN == 1, "unexpected Q6 MoE MMQ thread route tile");
    static_assert(WNITER * WARP * TM * TN * WMITER == WM * WN, "invalid Q6 MoE MMQ tile");
    static_assert(WSUBM == 64, "unexpected Q6 MoE MMQ M subtile shape");
    static_assert(LOADS_A == 8, "unexpected Q6 MoE MMQ A load shape");
    static_assert((BN == 16 && LOADS_B == 1) || (BN == 32 && LOADS_B == 1) || (BN == 64 && LOADS_B == 2),
            "unexpected Q6 MoE MMQ B load shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const uint32_t route_base0 = static_cast<uint32_t>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const long long expert = static_cast<long long>(__builtin_amdgcn_workgroup_id_z());
    if (expert >= c.n_experts) {
        return;
    }
    const uint32_t count = counts[expert];
    if (route_base0 >= count || row_base >= c.rows) {
        return;
    }

    const int tiw = static_cast<int>(tid);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const uint32_t * expert_routes = routes + expert * c.route_capacity;
    const long long q8_blocks_per_col = c.k / 32;
    const uint32_t route_tile_span = static_cast<uint32_t>(((c.n_tokens + BN - 1) / BN) * BN);

    __shared__ hrx_q6_k_moe_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_q6_moe_b_cache buf_b[BN * BK_STEP];

    for (uint32_t route_base = route_base0; route_base < count; route_base += route_tile_span) {
        float sum[WNITER * TM * TN] = {};

        for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
            const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
            const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
            const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
            const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q8_1_q6_moe_b_pending pending_b[LOADS_B];
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_A; ++load_i) {
                    const int r = loadc_a + load_i * LOAD_STRIDE_A;
                    hrx_q6_k_moe_mmq_load_a(
                        buf_a,
                        k_step * BM + r,
                        src0,
                        row_base + r,
                        kb_base + k_step,
                        loadr_a,
                        c.rows,
                        c.src0_nb1,
                        c.src0_nb2,
                        expert);
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_B; ++load_i) {
                    const int col = loadc_b + load_i * LOAD_STRIDE_B;
                    pending_b[load_i] = col < BN ?
                        hrx_q8_1_q6_moe_mmq_fetch_b(
                            src1,
                            expert_routes,
                            count,
                            route_base + col,
                            kb_base + k_step,
                            loadr_b,
                            q8_blocks_per_col) :
                        hrx_q8_1_q6_moe_b_pending {};
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_B; ++load_i) {
                    const int col = loadc_b + load_i * LOAD_STRIDE_B;
                    if (col < BN) {
                        hrx_q8_1_q6_moe_mmq_commit_b(buf_b, k_step * BN + col, pending_b[load_i], loadr_b);
                    }
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q6_k_moe_a_cache cache_a[TM];
                #pragma unroll
                for (int cr = 0; cr < TM; ++cr) {
                    cache_a[cr] = buf_a[k_step * BM + tiwr * TM + cr];
                }
                #pragma unroll
                for (int wsic = 0; wsic < WNITER; ++wsic) {
                    hrx_q8_1_q6_moe_b_cache cache_b[TN];
                    #pragma unroll
                    for (int cc = 0; cc < TN; ++cc) {
                        cache_b[cc] = buf_b[k_step * BN + wsic * WSUBN + tiwc * TN + cc];
                    }
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        #pragma unroll
                        for (int cc = 0; cc < TN; ++cc) {
                            int qsum0 = 0;
                            int qsum1 = 0;
                            #pragma unroll
                            for (int iqs = 0; iqs < 4; ++iqs) {
                                qsum0 += hrx_sdot4_q6_q8_1_qpack(cache_a[cr].qs[iqs], cache_b[cc].qs[iqs]);
                            }
                            #pragma unroll
                            for (int iqs = 4; iqs < 8; ++iqs) {
                                qsum1 += hrx_sdot4_q6_q8_1_qpack(cache_a[cr].qs[iqs], cache_b[cc].qs[iqs]);
                            }
                            sum[(wsic * TM + cr) * TN + cc] += cache_b[cc].d *
                                (cache_a[cr].d[0] * static_cast<float>(qsum0) +
                                 cache_a[cr].d[1] * static_cast<float>(qsum1));
                        }
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int wsic = 0; wsic < WNITER; ++wsic) {
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                const long long row = row_base + tiwr * TM + cr;
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    const int col = wsic * WSUBN + tiwc * TN + cc;
                    const uint32_t route_index = route_base + static_cast<uint32_t>(col);
                    if (row < c.rows && route_index < count) {
                        const uint32_t route = buf_b[col].route;
                        const long long id = static_cast<long long>(route % static_cast<uint32_t>(c.n_ids));
                        const long long token = static_cast<long long>(route / static_cast<uint32_t>(c.n_ids));
                        char * dst_base = reinterpret_cast<char *>(dst) + id * c.dst_nb1 + token * c.dst_nb2;
                        *reinterpret_cast<float *>(dst_base + row * sizeof(float)) =
                            sum[(wsic * TM + cr) * TN + cc];
                    }
                }
            }
        }
    }
}

extern "C" __global__ void hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x16_wg64_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q6_k_grouped_constants c) {
    hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq_wg64_impl<16, 1>(src0, src1, counts, routes, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x32_wg64_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q6_k_grouped_constants c) {
    hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq_wg64_impl<32, 1>(src0, src1, counts, routes, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq64x64_wg64_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q6_k_grouped_constants c) {
    hrx_mul_mat_id_q6_k_grouped_q8_1_x4_mmq_wg64_impl<64, 1>(src0, src1, counts, routes, dst, c);
}
