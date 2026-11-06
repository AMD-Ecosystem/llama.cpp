#include "fattn-vision.cuh"

#if defined(GGML_USE_HIP) && defined(RDNA3)
#define FATTN_VISION_DEVICE
#endif

typedef _Float16 fv_h16 __attribute__((ext_vector_type(16)));
typedef float    fv_f8  __attribute__((ext_vector_type(8)));

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

// Each wave computes S^T = K*Q^T and O^T += V^T*P^T for 16 queries, so one lane owns one query column.
// K/V tiles go through registers to LDS (V transposed); the next tile is loaded while the current one is used.
template <int D, int DP>
static __global__ void __launch_bounds__(FV_NW*32, 1) flash_attn_ext_vision(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, float * __restrict__ dst,
        const int nq, const int nkv, const int nh, const float qscale,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3) {
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

    const int h = blockIdx.y;
    const int b = blockIdx.z;

    Q += h*nbq2 + b*nbq3;
    K += h*nbk2 + b*nbk3;
    V += h*nbv2 + b*nbv3;

    if constexpr (D < DP) {
        for (int i = tid; i < FV_BK*(DP - D); i += NT) {
            Ks[(i / (DP - D))*SK + D + i % (DP - D)] = 0;
        }
        for (int i = tid; i < (DP - D)*SV; i += NT) {
            Vt[D*SV + i] = 0;
        }
    }

    const int q = blockIdx.x*FV_BQ + wave*16 + lr;
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
    load_regs(0);
    for (int t = 0; t < ntiles; ++t) {
        __syncthreads();
        store_lds();
        __syncthreads();
        if (t + 1 < ntiles) {
            load_regs((t + 1)*FV_BK);
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

        float m_new = m;
#pragma unroll
        for (int st = 0; st < 4; ++st) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                m_new = fmaxf(m_new, s[st][v]);
            }
        }
        m_new = fmaxf(m_new, __shfl_xor(m_new, 16, 32));
        const float alpha = __builtin_amdgcn_exp2f(m - m_new);
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
                const float p0 = __builtin_amdgcn_exp2f(s[st][2*j + 0] - m);
                const float p1 = __builtin_amdgcn_exp2f(s[st][2*j + 1] - m);
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
    }

    l += __shfl_xor(l, 16, 32);
    if (q < nq) {
        const float il = 1.0f/l;
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
    GGML_UNUSED_VARS(Q, K, V, dst, nq, nkv, nh, qscale, nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3);
    NO_DEVICE_CODE;
#endif // FATTN_VISION_DEVICE
}

bool ggml_cuda_flash_attn_ext_vision_supported(int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!GGML_CUDA_CC_IS_RDNA3(cc) || mask || sinks) {
        return false;
    }
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const int64_t D = Q->ne[0];
    if ((D != 64 && D != 72 && D != 80) || K->ne[0] != D || V->ne[0] != D) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->ne[2] != Q->ne[2] || V->ne[2] != Q->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || K->ne[1] != V->ne[1]) {
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

void ggml_cuda_flash_attn_ext_vision(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const float qscale = scale * 1.44269504088896341f; // exp -> exp2

    const int nq  = Q->ne[1];
    const int nkv = K->ne[1];
    const int nh  = Q->ne[2];

    const dim3 grid((nq + FV_BQ - 1)/FV_BQ, nh, Q->ne[3]);
    const dim3 block(FV_NW*32, 1, 1);
    cudaStream_t stream = ctx.stream();

#define FV_LAUNCH(D_, DP_) \
    flash_attn_ext_vision<D_, DP_><<<grid, block, 0, stream>>>( \
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (float *) dst->data, \
        nq, nkv, nh, qscale, Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3])

    switch (Q->ne[0]) {
        case 64: FV_LAUNCH(64, 64); break;
        case 72: FV_LAUNCH(72, 80); break;
        case 80: FV_LAUNCH(80, 80); break;
        default: GGML_ABORT("fatal error");
    }
#undef FV_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
