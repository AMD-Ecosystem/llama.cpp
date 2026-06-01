#include <float.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_flash_attn_ext_f32_f16_decode_gqa4_constants {
    long long D;
    long long KV;
    long long N;
    long long H;
    long long H_KV;
    long long S;
    long long q_nb1;
    long long q_nb2;
    long long q_nb3;
    long long k_nb1;
    long long k_nb2;
    long long k_nb3;
    long long v_nb1;
    long long v_nb2;
    long long v_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    long long mask_nb0;
    long long mask_nb1;
    long long mask_nb3;
    float     scale;
    int       has_mask;
    float     max_bias;
    float     m0;
    float     m1;
    float     logit_softcap;
    int       n_head_log2;
    int       has_sinks;
};

static __device__ __forceinline__ float hrx_fa_gqa4_load_f16(const char * ptr) {
    return __half2float(*reinterpret_cast<const __half *>(ptr));
}

extern "C" __global__ __launch_bounds__(256) void hrx_flash_attn_ext_f32_f16_decode_gqa4(
    const float *                                    q,
    const __half *                                   k,
    const __half *                                   v,
    const __half *                                   mask,
    const float *                                    sinks,
    float *                                          dst,
    hrx_flash_attn_ext_f32_f16_decode_gqa4_constants c) {
    (void) sinks;

    constexpr int D      = 128;
    constexpr int GQA    = 4;
    constexpr int MAX_KV = 512;

    __shared__ float q_tile[GQA][D];
    __shared__ float out_acc[GQA][D];
    __shared__ float partial[256];
    __shared__ float scores[GQA];
    __shared__ float m_vals[GQA];
    __shared__ float l_vals[GQA];

    const long long    kv_head = __builtin_amdgcn_workgroup_id_x();
    const long long    token   = __builtin_amdgcn_workgroup_id_y();
    const long long    seq     = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid     = __builtin_amdgcn_workitem_id_x();

    if (kv_head >= c.H_KV || token >= c.N || seq >= c.S || c.D != D || c.KV > MAX_KV || c.H != c.H_KV * GQA ||
        c.N != 1 || c.has_sinks || c.max_bias != 0.0f || c.logit_softcap != 0.0f) {
        return;
    }

    for (unsigned int i = tid; i < GQA * D; i += 256) {
        const unsigned int row  = i / D;
        const unsigned int d    = i - row * D;
        const long long    head = kv_head * GQA + row;
        const char * q_head     = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head * c.q_nb2 + seq * c.q_nb3;
        q_tile[row][d]          = *reinterpret_cast<const float *>(q_head + d * static_cast<long long>(sizeof(float)));
        out_acc[row][d]         = 0.0f;
    }
    if (tid < GQA) {
        m_vals[tid] = -FLT_MAX;
        l_vals[tid] = 0.0f;
    }
    __syncthreads();

    const char * k_head   = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head   = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;
    const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;

    for (long long t = 0; t < c.KV; ++t) {
        float mask_value = 0.0f;
        if (c.has_mask) {
            mask_value = hrx_fa_gqa4_load_f16(mask_row + t * c.mask_nb0);
        }

        const char * k_row = k_head + t * c.k_nb1;
#pragma unroll
        for (int h = 0; h < GQA; ++h) {
            float local = 0.0f;
            if (tid < D && mask_value > -60000.0f) {
                local = q_tile[h][tid] * hrx_fa_gqa4_load_f16(k_row + tid * static_cast<long long>(sizeof(__half)));
            }
            partial[tid] = local;
            __syncthreads();

            for (int stride = 64; stride > 0; stride >>= 1) {
                if (tid < static_cast<unsigned int>(stride)) {
                    partial[tid] += partial[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                scores[h] = mask_value <= -60000.0f ? -FLT_MAX : partial[0] * c.scale + mask_value;
            }
            __syncthreads();
        }

        if (tid < D) {
#pragma unroll
            for (int h = 0; h < GQA; ++h) {
                const float  score = scores[h];
                const float  old_m = m_vals[h];
                const float  new_m = fmaxf(old_m, score);
                const float  alpha = __expf(old_m - new_m);
                const float  beta  = score <= -FLT_MAX * 0.5f ? 0.0f : __expf(score - new_m);
                const char * v_row = v_head + t * c.v_nb1;
                const float  vv    = hrx_fa_gqa4_load_f16(v_row + tid * static_cast<long long>(sizeof(__half)));
                out_acc[h][tid]    = out_acc[h][tid] * alpha + beta * vv;
                if (tid == 0) {
                    l_vals[h] = l_vals[h] * alpha + beta;
                    m_vals[h] = new_m;
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int out_idx = tid; out_idx < GQA * (D / 4); out_idx += 256) {
        const unsigned int row  = out_idx / (D / 4);
        const unsigned int d    = (out_idx - row * (D / 4)) * 4;
        const long long    head = kv_head * GQA + row;
        char *      dst_head = reinterpret_cast<char *>(dst) + head * c.dst_nb1 + token * c.dst_nb2 + seq * c.dst_nb3;
        const float inv_l    = 1.0f / l_vals[row];
        *reinterpret_cast<float4 *>(dst_head + d * static_cast<long long>(sizeof(float))) =
            make_float4(out_acc[row][d] * inv_l, out_acc[row][d + 1] * inv_l, out_acc[row][d + 2] * inv_l,
                        out_acc[row][d + 3] * inv_l);
    }
}
