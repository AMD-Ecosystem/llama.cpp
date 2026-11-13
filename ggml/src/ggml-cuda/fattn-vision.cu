#include "fattn-vision.cuh"

#if defined(GGML_USE_HIP) && defined(RDNA3)
#define FATTN_VISION_DEVICE
#endif

#ifdef FATTN_VISION_DEVICE
typedef _Float16 fv_h16 __attribute__((ext_vector_type(16)));
typedef float    fv_f8  __attribute__((ext_vector_type(8)));
#endif

static constexpr int FV_NW = 8;        // waves per block, 16 queries per wave
static constexpr int FV_BQ = 16*FV_NW; // queries per block
static constexpr int FV_BK = 64;       // keys per K/V tile

#ifdef FATTN_VISION_DEVICE
static __device__ __forceinline__ fv_h16 fv_load16(const _Float16 * p) {
    fv_h16 r;
    ((uint4 *) &r)[0] = ((const uint4 *) p)[0];
    ((uint4 *) &r)[1] = ((const uint4 *) p)[1];
    return r;
}
#endif // FATTN_VISION_DEVICE

// mask tile classes for the prefill path
enum : uint8_t {
    FV_TILE_ZERO = 0, // all mask values are 0
    FV_TILE_INF  = 1, // all mask values are -inf, the tile is skipped
    FV_TILE_MIX  = 2, // mask is added to the scores
};

// one block per (KV tile, query block): class of the mask values of the queries < nq and keys < nkv
template <int BQ, int BK>
static __global__ void __launch_bounds__(FV_NW*32, 1) flash_attn_ext_prefill_classify(
        const char * __restrict__ mask, uint8_t * __restrict__ cls, const int nq, const int nkv,
        const int64_t nbm1, const int64_t nbm3) {
    const int t  = blockIdx.x;
    const int qb = blockIdx.y;
    const int b  = blockIdx.z;
    mask += b*nbm3;

    int any_nonzero = 0;
    int any_finite  = 0;
    constexpr int NCH = BK/8; // 16 byte chunks per mask row
    for (int c = threadIdx.x; c < BQ*NCH; c += FV_NW*32) {
        const int q = qb*BQ + c / NCH;
        const int k = t*BK + (c % NCH)*8;
        if (q >= nq || k >= nkv) {
            continue;
        }
        const uint16_t * row = (const uint16_t *) (mask + q*nbm1);
        uint16_t x[8];
        if (k + 8 <= nkv) {
            *(uint4 *) x = *(const uint4 *) (row + k);
        } else {
            for (int j = 0; j < 8; ++j) {
                x[j] = k + j < nkv ? row[k + j] : 0xFC00;
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            any_nonzero |= (x[j] & 0x7FFF) != 0;
            any_finite  |= x[j] != 0xFC00;
        }
    }
    any_nonzero = __syncthreads_or(any_nonzero);
    any_finite  = __syncthreads_or(any_finite);
    if (threadIdx.x == 0) {
        cls[(int64_t(b)*gridDim.y + qb)*gridDim.x + t] = !any_finite ? FV_TILE_INF : (any_nonzero ? FV_TILE_MIX : FV_TILE_ZERO);
    }
}

// Each wave computes S^T = K*Q^T and O^T += V^T*P^T for 16 queries, so one lane owns one query column.
// K/V tiles go through registers to LDS (V transposed); the next tile is loaded while the current one is used.
// MASK: f16 mask + GQA (prefill); tiles are classified by flash_attn_ext_prefill_classify, -inf tiles are skipped.
template <int D, int DP, bool MASK>
static __global__ void __launch_bounds__(FV_NW*32, 1) flash_attn_ext_vision(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, float * __restrict__ dst,
        const int nq, const int nkv, const int nh, const float qscale,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const char * __restrict__ mask, const uint8_t * __restrict__ cls, const int gqa,
        const int64_t nbm1, const int64_t nbm3, const int nbm) {
#ifdef FATTN_VISION_DEVICE
    constexpr int NCH = D/8;   // 16 byte chunks per K/V row
    constexpr int NKD = DP/16; // WMMA steps over the head dim
    constexpr int NT  = FV_NW*32;
    constexpr int NLD = (FV_BK/2*NCH + NT - 1)/NT;
    constexpr int SK  = DP + 8;    // Ks row stride
    constexpr int SV  = FV_BK + 8; // Vt row stride

    __shared__ __align__(16) _Float16 Ks[FV_BK*SK];
    __shared__ __align__(16) _Float16 Vt[DP*SV];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int lr   = lane % 16;
    const int hl   = lane / 16;

    // MASK: heads in x and the longest (last) causal query blocks first
    const int h  = MASK ? blockIdx.x : blockIdx.y;
    const int qblk = MASK ? gridDim.y - 1 - blockIdx.y : blockIdx.x;
    const int b  = blockIdx.z;
    const int hk = MASK ? h / gqa : h;

    Q += h*nbq2 + b*nbq3;
    K += hk*nbk2 + b*nbk3;
    V += hk*nbv2 + b*nbv3;

    if constexpr (D < DP) {
        for (int i = tid; i < FV_BK*(DP - D); i += NT) {
            Ks[(i / (DP - D))*SK + D + i % (DP - D)] = 0;
        }
        for (int i = tid; i < (DP - D)*SV; i += NT) {
            Vt[D*SV + i] = 0;
        }
    }

    const int q = qblk*FV_BQ + wave*16 + lr;
    fv_h16 qb[NKD];
#pragma unroll
    for (int c = 0; c < NKD; ++c) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d = c*16 + j;
            float x = 0.0f;
            if (d < D && q < nq) {
                x = ((const float *) (Q + q*nbq1))[d] * qscale;
            }
            qb[c][j] = (_Float16) x;
        }
    }

    // each load item is one 16 byte chunk of two consecutive K/V rows, so V can be written transposed as 32 bit pairs
    uint4 kr[NLD][2];
    uint4 vr[NLD][2];
    auto load_regs = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int c   = tid + i*NT;
            const int row = 2*(c / NCH);
            const int ch  = c % NCH;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                kr[i][r] = make_uint4(0, 0, 0, 0);
                vr[i][r] = make_uint4(0, 0, 0, 0);
                if (c < FV_BK/2*NCH && k0 + row + r < nkv) {
                    kr[i][r] = *(const uint4 *) (K + (k0 + row + r)*nbk1 + ch*16);
                    vr[i][r] = *(const uint4 *) (V + (k0 + row + r)*nbv1 + ch*16);
                }
            }
        }
    };
    auto store_lds = [&]() {
#pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int c   = tid + i*NT;
            const int row = 2*(c / NCH);
            const int ch  = c % NCH;
            if (c < FV_BK/2*NCH) {
                *(uint4 *) (Ks + (row + 0)*SK + ch*8) = kr[i][0];
                *(uint4 *) (Ks + (row + 1)*SK + ch*8) = kr[i][1];
                const uint32_t * v0 = (const uint32_t *) &vr[i][0];
                const uint32_t * v1 = (const uint32_t *) &vr[i][1];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    *(uint32_t *) (Vt + (ch*8 + 2*j + 0)*SV + row) = (v0[j] & 0x0000FFFFu) | (v1[j] << 16);
                    *(uint32_t *) (Vt + (ch*8 + 2*j + 1)*SV + row) = (v0[j] >> 16)         | (v1[j] & 0xFFFF0000u);
                }
            }
        }
    };

    float m = -INFINITY; // running max of the query column (log2 domain)
    float l = 0.0f;      // partial row sum, the other half-wave holds the rest
    fv_f8 o[NKD];
#pragma unroll
    for (int dt = 0; dt < NKD; ++dt) {
        o[dt] = fv_f8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    const int ntiles = (nkv + FV_BK - 1)/FV_BK;
    const uint8_t * cl = nullptr;
    const half * mrow = nullptr;
    if constexpr (MASK) {
        cl = cls + (int64_t(nbm == 1 ? 0 : b)*gridDim.y + qblk)*ntiles;
        mrow = (const half *) (mask + (nbm == 1 ? 0 : b)*nbm3 + int64_t(q < nq ? q : 0)*nbm1);
    }
    auto next_tile = [&](int t) {
        if constexpr (MASK) {
            while (t < ntiles && cl[t] == FV_TILE_INF) {
                ++t;
            }
        }
        return t;
    };

    int t = next_tile(0);
    if (t < ntiles) {
        load_regs(t*FV_BK);
    }
    while (t < ntiles) {
        __syncthreads();
        store_lds();
        __syncthreads();
        const int t_next = next_tile(t + 1);
        if (t_next < ntiles) {
            load_regs(t_next*FV_BK);
        }

        // s[st][v] = S^T[key st*16 + 2*v + hl][query lr]
        fv_f8 s[4];
#pragma unroll
        for (int st = 0; st < 4; ++st) {
            s[st] = fv_f8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c = 0; c < NKD; ++c) {
                const fv_h16 a = fv_load16(Ks + (st*16 + lr)*SK + c*16);
                s[st] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, qb[c], s[st]);
            }
        }

        const int kb = t*FV_BK;
        if (kb + FV_BK > nkv) {
#pragma unroll
            for (int st = 0; st < 4; ++st) {
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    if (kb + st*16 + 2*v + hl >= nkv) {
                        s[st][v] = -INFINITY;
                    }
                }
            }
        }
        if constexpr (MASK) {
            if (cl[t] == FV_TILE_MIX && q < nq) {
#pragma unroll
                for (int st = 0; st < 4; ++st) {
                    const int k0 = kb + st*16; // nkv % 16 == 0
                    if (k0 < nkv) {
                        uint32_t mw[8];
                        ((uint4 *) mw)[0] = ((const uint4 *) (mrow + k0))[0];
                        ((uint4 *) mw)[1] = ((const uint4 *) (mrow + k0))[1];
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            const uint16_t bits = hl ? mw[v] >> 16 : mw[v] & 0xFFFF;
                            s[st][v] += 1.44269504088896341f*__half2float(__ushort_as_half(bits));
                        }
                    }
                }
            }
        }

        float m_new = m;
#pragma unroll
        for (int st = 0; st < 4; ++st) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                m_new = fmaxf(m_new, s[st][v]);
            }
        }
        m_new = fmaxf(m_new, __shfl_xor(m_new, 16, 32));
        // a query column with only -inf so far keeps o = l = 0
        const float m_exp = MASK && m_new == -INFINITY ? 0.0f : m_new;
        const float alpha = __builtin_amdgcn_exp2f(m - m_exp);
        m = m_new;
        l *= alpha;
#pragma unroll
        for (int dt = 0; dt < NKD; ++dt) {
            o[dt] *= alpha;
        }

        // P^T as WMMA B operand: both half-waves need all 16 keys of the query column, keys 2*v + hl come from this lane
#pragma unroll
        for (int st = 0; st < 4; ++st) {
            union {
                fv_h16   h;
                uint32_t u[8];
            } pb;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p0 = __builtin_amdgcn_exp2f(s[st][2*j + 0] - m_exp);
                const float p1 = __builtin_amdgcn_exp2f(s[st][2*j + 1] - m_exp);
                l += p0 + p1;
                half2 w2 = __floats2half2_rn(p0, p1);
                const uint32_t w = *(const uint32_t *) &w2;
                const uint32_t x = __shfl_xor(w, 16, 32);
                const uint32_t lo = hl ? x : w;
                const uint32_t hi = hl ? w : x;
                pb.u[2*j + 0] = (lo & 0x0000FFFFu) | (hi << 16);
                pb.u[2*j + 1] = (lo >> 16)         | (hi & 0xFFFF0000u);
            }

            // o[dt][v] = O^T[dim dt*16 + 2*v + hl][query lr]
#pragma unroll
            for (int dt = 0; dt < NKD; ++dt) {
                const fv_h16 a = fv_load16(Vt + (dt*16 + lr)*SV + st*16);
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb.h, o[dt]);
            }
        }
        t = t_next;
    }

    l += __shfl_xor(l, 16, 32);
    if (q < nq) {
        const float il = MASK && l == 0.0f ? 0.0f : 1.0f/l;
        float * dst_q = dst + ((int64_t(b)*nq + q)*nh + h)*D;
#pragma unroll
        for (int dt = 0; dt < NKD; ++dt) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int d = dt*16 + 2*v + hl;
                if (d < D) {
                    dst_q[d] = o[dt][v]*il;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, dst, nq, nkv, nh, qscale, nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, mask, cls, gqa, nbm1, nbm3, nbm);
    NO_DEVICE_CODE;
#endif // FATTN_VISION_DEVICE
}

// D=256 prefill: the 16 queries of a group are handled by two waves, each owns half of the head dim of Q and O.
// The partial S of the two waves are summed through LDS, so both compute the same softmax.
static constexpr int FS_BK = 32;       // keys per K/V tile

// NG query groups of 16 per block; NG = 8 puts the S exchange buffer over the K tile to fit in LDS
// SXH: exchange the partial S in f16 (both waves round both halves the same way), so the buffer fits next to the K tile and one barrier per tile is saved
template <int D, int NG, bool SXH>
static __global__ void __launch_bounds__(NG*2*32, 1) flash_attn_ext_prefill_split(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, float * __restrict__ dst,
        const int nq, const int nkv, const int nh, const float qscale,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const char * __restrict__ mask, const uint8_t * __restrict__ cls, const int gqa,
        const int64_t nbm1, const int64_t nbm3, const int nbm) {
#ifdef FATTN_VISION_DEVICE
    constexpr int DH  = D/2;
    constexpr int NCH = D/8;
    constexpr int NKD = DH/16;
    constexpr int NT  = NG*2*32;
    constexpr bool SX_OVER_K = NG > 4 && !SXH;
    constexpr int NLD = (FS_BK/2*NCH + NT - 1)/NT;
    constexpr int SK  = D + 8;
    constexpr int SV  = FS_BK + 8;

    constexpr int KS_BYTES = FS_BK*SK*sizeof(_Float16);
    constexpr int SX_BYTES = 2*NG*32*16*(SXH ? sizeof(_Float16) : sizeof(float));
    __shared__ __align__(16) char smem[(SX_OVER_K ? (KS_BYTES > SX_BYTES ? KS_BYTES : SX_BYTES) : KS_BYTES + SX_BYTES) + D*SV*sizeof(_Float16)];
    _Float16 * Ks = (_Float16 *) smem;
    float (*Sx)[32][16] = (float (*)[32][16]) (smem + (SX_OVER_K ? 0 : KS_BYTES));
    _Float16 (*Sxh)[32][16] = (_Float16 (*)[32][16]) (smem + KS_BYTES);
    _Float16 * Vt = (_Float16 *) (smem + (SX_OVER_K ? (KS_BYTES > SX_BYTES ? KS_BYTES : SX_BYTES) : KS_BYTES + SX_BYTES));

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int lr   = lane % 16;
    const int hl   = lane / 16;
    const int hd   = wave % 2;

    const int h    = blockIdx.x;
    const int qblk = gridDim.y - 1 - blockIdx.y;
    const int b    = blockIdx.z;
    const int hk   = h / gqa;

    Q += h*nbq2 + b*nbq3;
    K += hk*nbk2 + b*nbk3;
    V += hk*nbv2 + b*nbv3;

    const int q = qblk*16*NG + (wave/2)*16 + lr;
    fv_h16 qb[NKD];
#pragma unroll
    for (int c = 0; c < NKD; ++c) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float x = q < nq ? ((const float *) (Q + q*nbq1))[hd*DH + c*16 + j] * qscale : 0.0f;
            qb[c][j] = (_Float16) x;
        }
    }

    uint4 kr[NLD][2];
    uint4 vr[NLD][2];
    auto load_regs = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int c   = tid + i*NT;
            const int row = 2*(c / NCH);
            const int ch  = c % NCH;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                kr[i][r] = make_uint4(0, 0, 0, 0);
                vr[i][r] = make_uint4(0, 0, 0, 0);
                if (c < FS_BK/2*NCH && k0 + row + r < nkv) {
                    kr[i][r] = *(const uint4 *) (K + (k0 + row + r)*nbk1 + ch*16);
                    vr[i][r] = *(const uint4 *) (V + (k0 + row + r)*nbv1 + ch*16);
                }
            }
        }
    };
    auto store_lds = [&]() {
#pragma unroll
        for (int i = 0; i < NLD; ++i) {
            const int c   = tid + i*NT;
            const int row = 2*(c / NCH);
            const int ch  = c % NCH;
            if (c < FS_BK/2*NCH) {
                *(uint4 *) (Ks + (row + 0)*SK + ch*8) = kr[i][0];
                *(uint4 *) (Ks + (row + 1)*SK + ch*8) = kr[i][1];
                const uint32_t * v0 = (const uint32_t *) &vr[i][0];
                const uint32_t * v1 = (const uint32_t *) &vr[i][1];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    *(uint32_t *) (Vt + (ch*8 + 2*j + 0)*SV + row) = (v0[j] & 0x0000FFFFu) | (v1[j] << 16);
                    *(uint32_t *) (Vt + (ch*8 + 2*j + 1)*SV + row) = (v0[j] >> 16)         | (v1[j] & 0xFFFF0000u);
                }
            }
        }
    };

    float m = -INFINITY;
    float l = 0.0f;
    fv_f8 o[NKD];
#pragma unroll
    for (int dt = 0; dt < NKD; ++dt) {
        o[dt] = fv_f8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    const int ntiles = (nkv + FS_BK - 1)/FS_BK;
    const uint8_t * cl = cls + (int64_t(nbm == 1 ? 0 : b)*gridDim.y + qblk)*ntiles;
    const half * mrow = (const half *) (mask + (nbm == 1 ? 0 : b)*nbm3 + int64_t(q < nq ? q : 0)*nbm1);
    auto next_tile = [&](int t) {
        while (t < ntiles && cl[t] == FV_TILE_INF) {
            ++t;
        }
        return t;
    };

    int t = next_tile(0);
    if (t < ntiles) {
        load_regs(t*FS_BK);
    }
    while (t < ntiles) {
        __syncthreads();
        store_lds();
        __syncthreads();
        const int t_next = next_tile(t + 1);
        if (t_next < ntiles) {
            load_regs(t_next*FS_BK);
        }

        fv_f8 s[2];
#pragma unroll
        for (int st = 0; st < 2; ++st) {
            s[st] = fv_f8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c = 0; c < NKD; ++c) {
                const fv_h16 a = fv_load16(Ks + (st*16 + lr)*SK + hd*DH + c*16);
                s[st] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, qb[c], s[st]);
            }
        }
        if constexpr (SX_OVER_K) {
            __syncthreads();
        }
        if constexpr (SXH) {
            typedef _Float16 fv_h8 __attribute__((ext_vector_type(8)));
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                const fv_h8 own = __builtin_convertvector(s[st], fv_h8);
                *(fv_h8 *) &Sxh[wave][lane][st*8] = own;
                s[st] = __builtin_convertvector(own, fv_f8);
            }
            __syncthreads();
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                typedef _Float16 fv_h8b __attribute__((ext_vector_type(8)));
                s[st] += __builtin_convertvector(*(const fv_h8b *) &Sxh[wave ^ 1][lane][st*8], fv_f8);
            }
        } else {
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                *(fv_f8 *) &Sx[wave][lane][st*8] = s[st];
            }
            __syncthreads();
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                s[st] += *(const fv_f8 *) &Sx[wave ^ 1][lane][st*8];
            }
        }

        const int kb = t*FS_BK;
#pragma unroll
        for (int st = 0; st < 2; ++st) {
            const int k0 = kb + st*16; // nkv % 16 == 0
            if (k0 >= nkv) {
                s[st] = fv_f8{-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
            } else if (cl[t] == FV_TILE_MIX && q < nq) {
                uint32_t mw[8];
                ((uint4 *) mw)[0] = ((const uint4 *) (mrow + k0))[0];
                ((uint4 *) mw)[1] = ((const uint4 *) (mrow + k0))[1];
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const uint16_t bits = hl ? mw[v] >> 16 : mw[v] & 0xFFFF;
                    s[st][v] += 1.44269504088896341f*__half2float(__ushort_as_half(bits));
                }
            }
        }

        float m_new = m;
#pragma unroll
        for (int st = 0; st < 2; ++st) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                m_new = fmaxf(m_new, s[st][v]);
            }
        }
        m_new = fmaxf(m_new, __shfl_xor(m_new, 16, 32));
        const float m_exp = m_new == -INFINITY ? 0.0f : m_new;
        const float alpha = __builtin_amdgcn_exp2f(m - m_exp);
        m = m_new;
        l *= alpha;
#pragma unroll
        for (int dt = 0; dt < NKD; ++dt) {
            o[dt] *= alpha;
        }

#pragma unroll
        for (int st = 0; st < 2; ++st) {
            union {
                fv_h16   h;
                uint32_t u[8];
            } pb;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p0 = __builtin_amdgcn_exp2f(s[st][2*j + 0] - m_exp);
                const float p1 = __builtin_amdgcn_exp2f(s[st][2*j + 1] - m_exp);
                l += p0 + p1;
                half2 w2 = __floats2half2_rn(p0, p1);
                const uint32_t w = *(const uint32_t *) &w2;
                const uint32_t x = __shfl_xor(w, 16, 32);
                const uint32_t lo = hl ? x : w;
                const uint32_t hi = hl ? w : x;
                pb.u[2*j + 0] = (lo & 0x0000FFFFu) | (hi << 16);
                pb.u[2*j + 1] = (lo >> 16)         | (hi & 0xFFFF0000u);
            }
#pragma unroll
            for (int dt = 0; dt < NKD; ++dt) {
                const fv_h16 a = fv_load16(Vt + (hd*DH + dt*16 + lr)*SV + st*16);
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, pb.h, o[dt]);
            }
        }
        t = t_next;
    }

    l += __shfl_xor(l, 16, 32);
    if (q < nq) {
        const float il = l == 0.0f ? 0.0f : 1.0f/l;
        float * dst_q = dst + ((int64_t(b)*nq + q)*nh + h)*D + hd*DH;
#pragma unroll
        for (int dt = 0; dt < NKD; ++dt) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                dst_q[dt*16 + 2*v + hl] = o[dt][v]*il;
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, dst, nq, nkv, nh, qscale, nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, mask, cls, gqa, nbm1, nbm3, nbm);
    NO_DEVICE_CODE;
#endif // FATTN_VISION_DEVICE
}

static bool fattn_vision_common_supported(int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];

    if (!GGML_CUDA_CC_IS_RDNA3_5(cc) || sinks || dst->src[5] ||
            (ggml_prec_get_acc(dst) != GGML_PREC_DEFAULT && ggml_prec_get_acc(dst) != GGML_PREC_F32) || K->ne[2] <= 0) {
        return false;
    }
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->ne[0] != Q->ne[0] || V->ne[0] != Q->ne[0] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2]) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || !ggml_is_contiguous(dst)) {
        return false;
    }
    for (const ggml_tensor * t : {K, V}) {
        if ((uintptr_t) t->data % 16 != 0 || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0) {
            return false;
        }
    }
    return Q->ne[1] >= FV_BQ/2 && K->ne[1] >= 1 && Q->ne[1] <= INT_MAX && K->ne[1] <= INT_MAX;
}

bool ggml_cuda_flash_attn_ext_vision_supported(int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int64_t D = Q->ne[0];
    return !dst->src[3] && (D == 64 || D == 72 || D == 80) && K->ne[2] == Q->ne[2] && fattn_vision_common_supported(cc, dst);
}

bool ggml_cuda_flash_attn_ext_prefill_supported(int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int64_t D = Q->ne[0];
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3])) {
        return false;
    }
    if (K->ne[1] % 16 != 0 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->nb[0] != sizeof(half) ||
            (uintptr_t) mask->data % 16 != 0 || mask->nb[1] % 16 != 0 || (mask->ne[3] > 1 && mask->nb[3] % 16 != 0)) {
        return false;
    }
    return (D == 64 || D == 128 || D == 256) && Q->ne[2] % K->ne[2] == 0 && fattn_vision_common_supported(cc, dst);
}

#ifndef GGML_CUDA_FATTN_PREFILL_MASK_CACHE
#define GGML_CUDA_FATTN_PREFILL_MASK_CACHE 1
#endif

void ggml_cuda_flash_attn_ext_prefill_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    auto & cache = ctx.fattn_cls;
    GGML_ASSERT(!cache.buf);
    cache.mask = nullptr;
    cache.inputs.clear();
    GGML_UNUSED(cgraph);
#if GGML_CUDA_FATTN_PREFILL_MASK_CACHE
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_FATTN_PREFILL_MASK_CACHE");
        return !env || atoi(env) != 0;
    }();
    static const bool prefill = []() {
        const char * env = getenv("GGML_CUDA_FATTN_PREFILL");
        return env && atoi(env) != 0;
    }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!enabled || !prefill || !GGML_CUDA_CC_IS_RDNA3_5(cc) || ctx.curr_stream_no != 0) {
        return;
    }
    size_t size = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_FLASH_ATTN_EXT || !ggml_cuda_flash_attn_ext_prefill_supported(cc, node)) {
            continue;
        }
        const ggml_tensor * mask = node->src[3];
        if (!mask || mask->op != GGML_OP_NONE || mask->view_src) {
            continue;
        }
        if (std::find(cache.inputs.begin(), cache.inputs.end(), mask) == cache.inputs.end()) {
            bool written = false;
            const uintptr_t base = (uintptr_t) mask->data;
            for (int j = 0; j < cgraph->n_nodes; ++j) {
                const ggml_tensor * other = cgraph->nodes[j];
                if (other->op == GGML_OP_NONE || other->op == GGML_OP_VIEW || other->op == GGML_OP_RESHAPE ||
                        other->op == GGML_OP_PERMUTE || other->op == GGML_OP_TRANSPOSE) {
                    continue;
                }
                const uintptr_t addr = (uintptr_t) other->data;
                written = written || (base <= addr ? addr - base < ggml_nbytes(mask) : base - addr < ggml_nbytes(other));
            }
            if (written) {
                continue;
            }
            cache.inputs.push_back(mask);
        }
        // D256 can use 64 query rows per tile; reserve for that smaller tile too.
        const size_t nq = (node->src[0]->ne[1] + 63)/64;
        const size_t nk = (node->src[1]->ne[1] + FS_BK - 1)/FS_BK;
        const size_t nb = mask->ne[3];
        GGML_ASSERT(nk > 0 && nb > 0 && nq <= SIZE_MAX/nk && nq*nk <= SIZE_MAX/nb);
        size = std::max(size, nq*nk*nb);
    }
    // Reserve before any node scratch so this allocation is released last.
    if (size) {
        cache.buf = std::make_unique<ggml_cuda_pool_alloc<uint8_t>>(ctx.pool(), size);
    }
#endif
}

void ggml_cuda_flash_attn_ext_prefill_release(ggml_backend_cuda_context & ctx) {
    ctx.fattn_cls.buf.reset();
    ctx.fattn_cls.mask = nullptr;
    ctx.fattn_cls.inputs.clear();
}

static void fattn_vision_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool use_mask) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = use_mask ? dst->src[3] : nullptr;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const float qscale = scale * 1.44269504088896341f; // exp -> exp2

    const int nq  = Q->ne[1];
    const int nkv = K->ne[1];
    const int nh  = Q->ne[2];
    const bool split = mask && Q->ne[0] == 256;
    // D256 uses 128 queries per block by default; 64 queries remain available for comparison.
    static const int split_ng = getenv("GGML_CUDA_FATTN_PREFILL_D256_NG") && atoi(getenv("GGML_CUDA_FATTN_PREFILL_D256_NG")) == 4 ? 4 : 8;
    const int bq  = split ? 16*split_ng : FV_BQ;
    const int bk  = split ? FS_BK : FV_BK;
    const int nqb = (nq + bq - 1)/bq;
    const int ntk = (nkv + bk - 1)/bk;
    const int nbm = mask ? mask->ne[3] : 0;

    const dim3 block(FV_NW*32, 1, 1);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<uint8_t> cls_local(ctx.pool());
    uint8_t * cls = nullptr;
    auto & cache = ctx.fattn_cls;
    const int64_t key[7] = {nq, nkv, bq, bk, nbm, mask ? int64_t(mask->nb[1]) : 0, mask ? int64_t(mask->nb[3]) : 0};
    const bool reuse = mask && ctx.curr_stream_no == 0 && cache.buf &&
        std::find(cache.inputs.begin(), cache.inputs.end(), mask) != cache.inputs.end();
    if (reuse && cache.mask == mask && memcmp(cache.key, key, sizeof(key)) == 0) {
        cls = cache.buf->get();
    } else if (mask) {
        if (reuse) {
            cache.mask = mask;
            memcpy(cache.key, key, sizeof(key));
            cls = cache.buf->get();
        } else {
            cls = cls_local.alloc(size_t(nbm)*nqb*ntk);
        }
        if (split && split_ng == 8) {
            flash_attn_ext_prefill_classify<128, FS_BK><<<dim3(ntk, nqb, nbm), block, 0, stream>>>(
                (const char *) mask->data, cls, nq, nkv, mask->nb[1], mask->nb[3]);
        } else if (split) {
            flash_attn_ext_prefill_classify<64, FS_BK><<<dim3(ntk, nqb, nbm), block, 0, stream>>>(
                (const char *) mask->data, cls, nq, nkv, mask->nb[1], mask->nb[3]);
        } else {
            flash_attn_ext_prefill_classify<FV_BQ, FV_BK><<<dim3(ntk, nqb, nbm), block, 0, stream>>>(
                (const char *) mask->data, cls, nq, nkv, mask->nb[1], mask->nb[3]);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    const dim3 grid = mask ? dim3(nh, nqb, Q->ne[3]) : dim3(nqb, nh, Q->ne[3]);

    if (split) {
#define FS_LAUNCH(NG_, SXH_) \
        flash_attn_ext_prefill_split<256, NG_, SXH_><<<grid, dim3(NG_*2*32, 1, 1), 0, stream>>>( \
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (float *) dst->data, \
            nq, nkv, nh, qscale, Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3], \
            (const char *) mask->data, cls, nh / K->ne[2], mask->nb[1], mask->nb[3], nbm)
        // opt-in (GGML_CUDA_FATTN_PREFILL_D256_SXH): f16 exchange of the partial scores, one barrier less per tile
        static const bool sxh = getenv("GGML_CUDA_FATTN_PREFILL_D256_SXH") && atoi(getenv("GGML_CUDA_FATTN_PREFILL_D256_SXH")) != 0;
        if (split_ng == 8) {
            if (sxh) {
                FS_LAUNCH(8, true);
            } else {
                FS_LAUNCH(8, false);
            }
        } else {
            FS_LAUNCH(4, false);
        }
#undef FS_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }

#define FV_LAUNCH(D_, DP_, M_) \
    flash_attn_ext_vision<D_, DP_, M_><<<grid, block, 0, stream>>>( \
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (float *) dst->data, \
        nq, nkv, nh, qscale, Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3], \
        mask ? (const char *) mask->data : nullptr, cls, nh / K->ne[2], mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, nbm)

    if (mask) {
        switch (Q->ne[0]) {
            case  64: FV_LAUNCH( 64,  64, true); break;
            case 128: FV_LAUNCH(128, 128, true); break;
            default: GGML_ABORT("fatal error");
        }
    } else {
        switch (Q->ne[0]) {
            case 64: FV_LAUNCH(64, 64, false); break;
            case 72: FV_LAUNCH(72, 80, false); break;
            case 80: FV_LAUNCH(80, 80, false); break;
            default: GGML_ABORT("fatal error");
        }
    }
#undef FV_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_vision(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    fattn_vision_launch(ctx, dst, false);
}

void ggml_cuda_flash_attn_ext_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    fattn_vision_launch(ctx, dst, true);
}
