#include <float.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

struct hrx_flash_attn_ext_f32_f16_decode_gqa4_tiled_split_constants {
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

static __device__ __forceinline__ float hrx_fa_gqa4_tiled_load_f16(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

static __device__ __forceinline__ float4 hrx_fa_gqa4_tiled_load_f16x4(const char * ptr) {
    const __half2 h01 = *reinterpret_cast<const __half2 *>(ptr);
    const __half2 h23 = *reinterpret_cast<const __half2 *>(ptr + 2 * static_cast<int>(sizeof(__half)));
    const float2  f01 = __half22float2(h01);
    const float2  f23 = __half22float2(h23);
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}

static __device__ __forceinline__ float4 hrx_fa_gqa4_tiled_load_f32x4(const char * ptr) {
    return *reinterpret_cast<const float4 *>(ptr);
}

static __device__ __forceinline__ float hrx_fa_gqa4_tiled_dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static __device__ __forceinline__ float4 hrx_fa_gqa4_tiled_zero4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

static __device__ __forceinline__ float4 hrx_fa_gqa4_tiled_madd4(float4 acc, float scale, float4 v) {
    acc.x += scale * v.x;
    acc.y += scale * v.y;
    acc.z += scale * v.z;
    acc.w += scale * v.w;
    return acc;
}

static __device__ __forceinline__ float4 hrx_fa_gqa4_tiled_scale4(float4 v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

static __device__ __forceinline__ float hrx_fa_gqa4_tiled_sum_dsplit(float v) {
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        v += __shfl_xor(v, mask, 32);
    }
    return v;
}

static __device__ __forceinline__ float hrx_fa_gqa4_tiled_sum_cols(float v) {
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 16, 32);
    return v;
}

static __device__ __forceinline__ float hrx_fa_gqa4_tiled_max_cols(float v) {
    v = fmaxf(v, __shfl_xor(v, 8, 32));
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    return v;
}

static __device__ __forceinline__ float4 hrx_fa_gqa4_tiled_sum_cols4(float4 v) {
    v.x = hrx_fa_gqa4_tiled_sum_cols(v.x);
    v.y = hrx_fa_gqa4_tiled_sum_cols(v.y);
    v.z = hrx_fa_gqa4_tiled_sum_cols(v.z);
    v.w = hrx_fa_gqa4_tiled_sum_cols(v.w);
    return v;
}

extern "C" __global__ __launch_bounds__(64) void hrx_flash_attn_ext_f32_f16_decode_gqa4_tiled_split(
    const float *                                                q,
    const __half *                                               k,
    const __half *                                               v,
    const __half *                                               mask,
    const float *                                                sinks,
    float *                                                      dst,
    float *                                                      scratch,
    hrx_flash_attn_ext_f32_f16_decode_gqa4_tiled_split_constants c) {
    (void) sinks;
    (void) dst;
    constexpr int GQA             = 4;
    constexpr int SPLITS          = 8;
    constexpr int D               = 128;
    constexpr int KV_TILE         = 64;
    constexpr int D_SPLIT         = 8;
    constexpr int BC              = 16;
    constexpr int COLS_PER_THREAD = 4;
    constexpr int VEC_PER_THREAD  = D / (4 * D_SPLIT);

    const long long    x         = __builtin_amdgcn_workgroup_id_x();
    const long long    split     = x % SPLITS;
    const long long    kv_head   = x / SPLITS;
    const long long    token     = __builtin_amdgcn_workgroup_id_y();
    const long long    seq       = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid       = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane      = tid & 31;
    const unsigned int row_group = tid >> 5;
    const unsigned int d_tid     = lane & (D_SPLIT - 1);
    const unsigned int col_tid   = lane >> 3;

    if (kv_head >= c.H_KV || token >= c.N || seq >= c.S || c.D != D || c.KV != 512 || c.N != 1 || c.S != 1 ||
        c.H != c.H_KV * GQA || c.has_sinks || c.max_bias != 0.0f || c.logit_softcap != 0.0f) {
        return;
    }

    const long long split_begin = split * KV_TILE;
    const long long split_end   = split_begin + KV_TILE;
    const int       row0        = static_cast<int>(row_group * 2);
    const int       row1        = row0 + 1;
    const long long head0       = kv_head * GQA + row0;
    const long long head1       = kv_head * GQA + row1;
    const bool      valid0      = row0 < GQA && head0 < c.H;
    const bool      valid1      = row1 < GQA && head1 < c.H;

    const char * k_head   = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head   = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;
    const char * q_head0  = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head0 * c.q_nb2 + seq * c.q_nb3;
    const char * q_head1  = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head1 * c.q_nb2 + seq * c.q_nb3;
    const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;

    float  l0 = 0.0f;
    float  l1 = 0.0f;
    float  m0 = -FLT_MAX * 0.5f;
    float  m1 = -FLT_MAX * 0.5f;
    float4 q0[VEC_PER_THREAD];
    float4 q1[VEC_PER_THREAD];
    float4 out0[VEC_PER_THREAD];
    float4 out1[VEC_PER_THREAD];
#pragma unroll
    for (int d = 0; d < VEC_PER_THREAD; ++d) {
        const int vec_index       = d * D_SPLIT + static_cast<int>(d_tid);
        const int byte_offset_f32 = vec_index * 4 * static_cast<int>(sizeof(float));
        q0[d]   = hrx_fa_gqa4_tiled_scale4(hrx_fa_gqa4_tiled_load_f32x4(q_head0 + byte_offset_f32), c.scale);
        q1[d]   = hrx_fa_gqa4_tiled_scale4(hrx_fa_gqa4_tiled_load_f32x4(q_head1 + byte_offset_f32), c.scale);
        out0[d] = hrx_fa_gqa4_tiled_zero4();
        out1[d] = hrx_fa_gqa4_tiled_zero4();
    }

    for (long long jb = split_begin; jb < split_end; jb += BC) {
        float scores0[COLS_PER_THREAD];
        float scores1[COLS_PER_THREAD];
#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const long long kv_col = jb + ci * 4 + col_tid;
            const float     mask_value =
                hrx_fa_gqa4_tiled_load_f16(reinterpret_cast<const __half *>(mask_row), kv_col * c.mask_nb0);

            float        s0    = 0.0f;
            float        s1    = 0.0f;
            const char * k_row = k_head + kv_col * c.k_nb1;
#pragma unroll
            for (int d = 0; d < VEC_PER_THREAD; ++d) {
                const int    vec_index       = d * D_SPLIT + static_cast<int>(d_tid);
                const int    byte_offset_f16 = vec_index * 4 * static_cast<int>(sizeof(__half));
                const float4 kv              = hrx_fa_gqa4_tiled_load_f16x4(k_row + byte_offset_f16);
                s0 += hrx_fa_gqa4_tiled_dot4(q0[d], kv);
                s1 += hrx_fa_gqa4_tiled_dot4(q1[d], kv);
            }
            s0          = hrx_fa_gqa4_tiled_sum_dsplit(s0) + mask_value;
            s1          = hrx_fa_gqa4_tiled_sum_dsplit(s1) + mask_value;
            scores0[ci] = s0;
            scores1[ci] = s1;
        }

        float row_max0 = -FLT_MAX * 0.5f;
        float row_max1 = -FLT_MAX * 0.5f;
#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            row_max0 = fmaxf(row_max0, scores0[ci]);
            row_max1 = fmaxf(row_max1, scores1[ci]);
        }
        row_max0 = hrx_fa_gqa4_tiled_max_cols(row_max0);
        row_max1 = hrx_fa_gqa4_tiled_max_cols(row_max1);

        const float old_m0     = m0;
        const float old_m1     = m1;
        m0                     = fmaxf(m0, row_max0);
        m1                     = fmaxf(m1, row_max1);
        const float old_scale0 = __expf(old_m0 - m0);
        const float old_scale1 = __expf(old_m1 - m1);
        l0 *= old_scale0;
        l1 *= old_scale1;
#pragma unroll
        for (int d = 0; d < VEC_PER_THREAD; ++d) {
            out0[d] = hrx_fa_gqa4_tiled_scale4(out0[d], old_scale0);
            out1[d] = hrx_fa_gqa4_tiled_scale4(out1[d], old_scale1);
        }

#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const long long kv_col = jb + ci * 4 + col_tid;
            const float     p0     = __expf(scores0[ci] - m0);
            const float     p1     = __expf(scores1[ci] - m1);
            l0 += p0;
            l1 += p1;
            const char * v_row = v_head + kv_col * c.v_nb1;
#pragma unroll
            for (int d = 0; d < VEC_PER_THREAD; ++d) {
                const int    vec_index       = d * D_SPLIT + static_cast<int>(d_tid);
                const int    byte_offset_f16 = vec_index * 4 * static_cast<int>(sizeof(__half));
                const float4 vv              = hrx_fa_gqa4_tiled_load_f16x4(v_row + byte_offset_f16);
                out0[d]                      = hrx_fa_gqa4_tiled_madd4(out0[d], p0, vv);
                out1[d]                      = hrx_fa_gqa4_tiled_madd4(out1[d], p1, vv);
            }
        }
    }

    l0 = hrx_fa_gqa4_tiled_sum_cols(l0);
    l1 = hrx_fa_gqa4_tiled_sum_cols(l1);
#pragma unroll
    for (int d = 0; d < VEC_PER_THREAD; ++d) {
        out0[d] = hrx_fa_gqa4_tiled_sum_cols4(out0[d]);
        out1[d] = hrx_fa_gqa4_tiled_sum_cols4(out1[d]);
    }

    if (col_tid == 0) {
        const size_t partial_count =
            static_cast<size_t>(c.S) * static_cast<size_t>(c.N) * static_cast<size_t>(c.H) * SPLITS;
        float *      scratch_o = scratch;
        float *      scratch_l = scratch_o + partial_count * D;
        float *      scratch_m = scratch_l + partial_count;
        const size_t base0 =
            (((static_cast<size_t>(seq) * c.N + static_cast<size_t>(token)) * c.H + static_cast<size_t>(head0)) *
                 SPLITS +
             static_cast<size_t>(split));
        const size_t base1 =
            (((static_cast<size_t>(seq) * c.N + static_cast<size_t>(token)) * c.H + static_cast<size_t>(head1)) *
                 SPLITS +
             static_cast<size_t>(split));
        if (d_tid == 0) {
            scratch_l[base0] = l0;
            scratch_m[base0] = m0;
            scratch_l[base1] = l1;
            scratch_m[base1] = m1;
        }
#pragma unroll
        for (int d = 0; d < VEC_PER_THREAD; ++d) {
            const int vec_index                                                = d * D_SPLIT + static_cast<int>(d_tid);
            *reinterpret_cast<float4 *>(scratch_o + base0 * D + vec_index * 4) = out0[d];
            *reinterpret_cast<float4 *>(scratch_o + base1 * D + vec_index * 4) = out1[d];
        }
    }
}
