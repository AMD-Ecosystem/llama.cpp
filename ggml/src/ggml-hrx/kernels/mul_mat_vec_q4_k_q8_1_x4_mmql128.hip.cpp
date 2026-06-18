#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HRX_Q4_K_Q8_1_X4_MMQL128_EXPORT
#define HRX_Q4_K_Q8_1_X4_MMQL128_EXPORT hrx_mul_mat_vec_q4_k_q8_1_x4_mmql128x128_wg256_f32
#endif

#ifndef HRX_Q4_K_Q8_1_X4_MMQL128_BK_STEP
#define HRX_Q4_K_Q8_1_X4_MMQL128_BK_STEP 1
#endif

#ifndef HRX_Q4_K_Q8_1_X4_MMQL128_BM
#define HRX_Q4_K_Q8_1_X4_MMQL128_BM 128
#endif

#ifndef HRX_Q4_K_Q8_1_X4_MMQL128_BN
#define HRX_Q4_K_Q8_1_X4_MMQL128_BN 128
#endif

#ifndef HRX_Q4_K_Q8_1_X4_MMQL128_WN
#define HRX_Q4_K_Q8_1_X4_MMQL128_WN 64
#endif

#ifndef HRX_Q4_K_Q8_1_X4_MMQL128_A_PAD_WORDS
#define HRX_Q4_K_Q8_1_X4_MMQL128_A_PAD_WORDS 0
#endif

#ifndef HRX_Q4_K_Q8_1_X4_MMQL128_B_PAD_WORDS
#define HRX_Q4_K_Q8_1_X4_MMQL128_B_PAD_WORDS 0
#endif

struct hrx_block_q4_K_q8_1_mmql_lhs {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_block_q8_1_x4_rhs_q4_mmql {
    unsigned short ds[8];
    int qs[32];
};

struct hrx_q4_k_mmql_a_cache {
    int qs[4];
    float d;
    float min;
#if HRX_Q4_K_Q8_1_X4_MMQL128_A_PAD_WORDS > 0
    int pad[HRX_Q4_K_Q8_1_X4_MMQL128_A_PAD_WORDS];
#endif
};

struct hrx_q8_1_mmql_b_cache_q4 {
    int qs[8];
    float d;
    float s;
#if HRX_Q4_K_Q8_1_X4_MMQL128_B_PAD_WORDS > 0
    int pad[HRX_Q4_K_Q8_1_X4_MMQL128_B_PAD_WORDS];
#endif
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_q8_1_mmql(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ int hrx_udot4_q4_q8_1_mmql(uint32_t qpack, int rpack) {
    return __builtin_amdgcn_sudot4(false, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ uint32_t hrx_q4_k_mmql_pack8(
        const hrx_block_q4_K_q8_1_mmql_lhs * block,
        int group,
        int iqs_pair) {
    const int qs_base = (group >> 1) * 32 + iqs_pair * 8;
    const uint32_t raw0 = *reinterpret_cast<const uint32_t *>(block->qs + qs_base);
    const uint32_t raw1 = *reinterpret_cast<const uint32_t *>(block->qs + qs_base + 4);
    const int shift = (group & 1) * 4;
    const uint32_t vals0 = (raw0 >> shift) & 0x0F0F0F0Fu;
    const uint32_t vals1 = (raw1 >> shift) & 0x0F0F0F0Fu;
    return vals0 | (vals1 << 4);
}

static __device__ __forceinline__ void hrx_q4_k_mmql_load_a(
        hrx_q4_k_mmql_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q4_K_q8_1_mmql_lhs * src0,
        long long row,
        long long kb,
        int iqs_pair,
        long long blocks_per_row) {
    const hrx_block_q4_K_q8_1_mmql_lhs * block = src0 + row * blocks_per_row + (kb >> 3);
    const int group = static_cast<int>(kb & 7);
    buf_a[buf_idx].qs[iqs_pair] = static_cast<int>(hrx_q4_k_mmql_pack8(block, group, iqs_pair));
    if (iqs_pair == 0) {
        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q8_1_mmql(group, block->scales, &sc, &m);
        buf_a[buf_idx].d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        buf_a[buf_idx].min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    }
}

static __device__ __forceinline__ void hrx_q4_k_mmql_load_b(
        hrx_q8_1_mmql_b_cache_q4 * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q4_mmql * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q4_mmql * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
        buf_b[buf_idx].s = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 1]));
    }
}

extern "C" __global__ void HRX_Q4_K_Q8_1_X4_MMQL128_EXPORT(
        const hrx_block_q4_K_q8_1_mmql_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q4_mmql * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    constexpr int BM = HRX_Q4_K_Q8_1_X4_MMQL128_BM;
    constexpr int BN = HRX_Q4_K_Q8_1_X4_MMQL128_BN;
    constexpr int BK_STEP = HRX_Q4_K_Q8_1_X4_MMQL128_BK_STEP;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP = 64;
    constexpr int WM = 64;
    constexpr int WN = HRX_Q4_K_Q8_1_X4_MMQL128_WN;
    constexpr int WMITER = 1;
    constexpr int TM = 4;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 8;
    constexpr int LOAD_VEC_B = 16;

    static_assert(BM == 128 && (BN == 128 || BN == 64), "unexpected Q4 MMQ tile shape");
    static_assert(WNITER == 8 || WNITER == 4, "unexpected Q4 MMQ column iteration count");
    static_assert(WSUBM == 64 && WSUBN == 8, "unexpected Vulkan large Q4 MMQ subtile shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int warp_i = static_cast<int>(tid / WARP);
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    __shared__ hrx_q4_k_mmql_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_mmql_b_cache_q4 buf_b[BN * BK_STEP];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const bool full_tile = row_base + BM <= rows && col_base + BN <= cols;

    float sum[WNITER * TM * TN] = {};

    for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
        const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
        const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
        const int loadstride_a = BLOCK_SIZE * LOAD_VEC_A / 32;
        const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
        const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));
        const int loadstride_b = BLOCK_SIZE * LOAD_VEC_B / 32;
        if (full_tile) {
            for (int r = loadc_a; r < BM; r += loadstride_a) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q4_k_mmql_load_a(
                        buf_a,
                        k_step * BM + r,
                        src0,
                        row_base + r,
                        kb_base + k_step,
                        loadr_a,
                        blocks_per_row);
                }
            }
            for (int c = loadc_b; c < BN; c += loadstride_b) {
                #pragma unroll
                for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                    hrx_q4_k_mmql_load_b(
                        buf_b,
                        k_step * BN + c,
                        src1,
                        col_base + c,
                        kb_base + k_step,
                        loadr_b,
                        q8_blocks_per_col);
                }
            }
        } else {
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                for (int r = loadc_a; r < BM; r += loadstride_a) {
                    const int buf_idx = k_step * BM + r;
                    if (row_base + r < rows) {
                        hrx_q4_k_mmql_load_a(
                            buf_a,
                            buf_idx,
                            src0,
                            row_base + r,
                            kb_base + k_step,
                            loadr_a,
                            blocks_per_row);
                    } else {
                        buf_a[buf_idx].qs[loadr_a] = 0;
                        if (loadr_a == 0) {
                            buf_a[buf_idx].d = 0.0f;
                            buf_a[buf_idx].min = 0.0f;
                        }
                    }
                }
            }

            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                for (int c = loadc_b; c < BN; c += loadstride_b) {
                    const int buf_idx = k_step * BN + c;
                    if (col_base + c < cols) {
                        hrx_q4_k_mmql_load_b(
                            buf_b,
                            buf_idx,
                            src1,
                            col_base + c,
                            kb_base + k_step,
                            loadr_b,
                            q8_blocks_per_col);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            buf_b[buf_idx].qs[loadr_b * 4 + j] = 0;
                        }
                        if (loadr_b == 0) {
                            buf_b[buf_idx].d = 0.0f;
                            buf_b[buf_idx].s = 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            hrx_q4_k_mmql_a_cache cache_a[TM];
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                cache_a[cr] = buf_a[k_step * BM + warp_r * WM + tiwr * TM + cr];
            }

            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    hrx_q8_1_mmql_b_cache_q4 cache_b =
                        buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            const uint32_t qpack =
                                (static_cast<uint32_t>(cache_a[cr].qs[iqs >> 1]) >> ((iqs & 1) * 4)) &
                                0x0F0F0F0Fu;
                            qsum += hrx_udot4_q4_q8_1_mmql(qpack, cache_b.qs[iqs]);
                        }
                        sum[(wsic * TN + cc) * TM + cr] +=
                            cache_a[cr].d * cache_b.d * static_cast<float>(qsum) -
                            cache_a[cr].min * cache_b.s;
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
            const long long row = row_base + warp_r * WM + tiwr * TM + cr;
            #pragma unroll
            for (int cc = 0; cc < TN; ++cc) {
                const long long col = col_base + warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
                if (full_tile || (row < rows && col < cols)) {
                    dst[col * rows + row] = sum[(wsic * TN + cc) * TM + cr];
                }
            }
        }
    }
}
