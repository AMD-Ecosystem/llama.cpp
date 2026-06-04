#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct hrx_flash_attn_ext_f32_f16_decode_constants {
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
    float scale;
    int has_mask;
    float max_bias;
    float m0;
    float m1;
    float logit_softcap;
    int n_head_log2;
    int has_sinks;
};

static __device__ __forceinline__ float hrx_fa_split_load_f16(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

static __device__ __forceinline__ float4 hrx_fa_split_load_f16x4(const char * ptr) {
    const __half * h = reinterpret_cast<const __half *>(ptr);
    return make_float4(__half2float(h[0]), __half2float(h[1]), __half2float(h[2]), __half2float(h[3]));
}

static __device__ __forceinline__ float4 hrx_fa_split_load_f32x4(const char * ptr) {
    return *reinterpret_cast<const float4 *>(ptr);
}

static __device__ __forceinline__ float hrx_fa_split_dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static __device__ __forceinline__ float4 hrx_fa_split_f4_zero() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

static __device__ __forceinline__ float4 hrx_fa_split_f4_madd(float4 acc, float scale, float4 v) {
    acc.x += scale * v.x;
    acc.y += scale * v.y;
    acc.z += scale * v.z;
    acc.w += scale * v.w;
    return acc;
}

// Reduce across the d_tid dimension (D_SPLIT=8 lanes -> masks 4,2,1).
static __device__ __forceinline__ float hrx_fa_split_sum_dsplit(float v) {
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        v += __shfl_xor(v, mask, 32);
    }
    return v;
}

// Reduce across the col_tid dimension (col_tid in [0,4) -> masks 8,16).
static __device__ __forceinline__ float hrx_fa_split_sum_cols(float v) {
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 16, 32);
    return v;
}

static __device__ __forceinline__ float hrx_fa_split_max_cols(float v) {
    v = fmaxf(v, __shfl_xor(v, 8, 32));
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    return v;
}

static __device__ __forceinline__ float4 hrx_fa_split_sum_cols4(float4 v) {
    v.x = hrx_fa_split_sum_cols(v.x);
    v.y = hrx_fa_split_sum_cols(v.y);
    v.z = hrx_fa_split_sum_cols(v.z);
    v.w = hrx_fa_split_sum_cols(v.w);
    return v;
}

static __device__ __forceinline__ float4 hrx_fa_split_scale4(float4 v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

static __device__ __forceinline__ float hrx_fa_split_alibi_slope(
        const hrx_flash_attn_ext_f32_f16_decode_constants c,
        long long head) {
    if (c.max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = head < c.n_head_log2 ? c.m0 : c.m1;
    const int exp_h = head < c.n_head_log2 ? static_cast<int>(head + 1) :
        static_cast<int>(2 * (head - c.n_head_log2) + 1);
    return powf(base, exp_h);
}

template <int D, int GQA>
static __device__ __forceinline__ void hrx_flash_attn_ext_f32_f16_decode_split_impl(
        const float * q,
        const __half * k,
        const __half * v,
        const __half * mask,
        const float * sinks,
        float * dst,
        float * scratch,
        hrx_flash_attn_ext_f32_f16_decode_constants c) {
    (void) sinks;   // sinks + normalization are the reduce kernel's job
    (void) dst;     // present only to keep the binding order identical to the single-kernel ABI
    constexpr int SPLITS = 8;
    constexpr int D_SPLIT = 8;
    constexpr int BC = 32;
    constexpr int COLS_PER_THREAD = 8;
    constexpr int VEC_PER_THREAD = D / (4 * D_SPLIT);
    constexpr int N_ROW_GROUPS = 4;                  // 128 threads / 32 (wavefront)
    constexpr int HEADS_PER_RG = GQA / N_ROW_GROUPS; // GQA-8 -> 2 heads/row_group, GQA-4 -> 1

    const long long x = __builtin_amdgcn_workgroup_id_x();
    const long long split = x % SPLITS;
    const long long kv_head = x / SPLITS;
    const long long token = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & 31;
    const unsigned int row_group = tid >> 5;
    const unsigned int d_tid = lane & (D_SPLIT - 1);
    const unsigned int col_tid = lane >> 3;

    if (kv_head >= c.H_KV || token >= c.N || seq >= c.S || c.D != D || c.H != c.H_KV * GQA) {
        return;
    }

    const long long split_chunk = (((c.KV + SPLITS - 1) / SPLITS) + (BC - 1)) & ~(static_cast<long long>(BC) - 1);
    const long long split_begin = split * split_chunk;
    const long long split_end = split_begin + split_chunk < c.KV ? split_begin + split_chunk : c.KV;

    int row[HEADS_PER_RG];
    long long head[HEADS_PER_RG];
    bool valid[HEADS_PER_RG];
    const char * q_head[HEADS_PER_RG];
    float slope[HEADS_PER_RG];
#pragma unroll
    for (int h = 0; h < HEADS_PER_RG; ++h) {
        row[h] = static_cast<int>(row_group) * HEADS_PER_RG + h;
        head[h] = kv_head * GQA + row[h];
        valid[h] = row[h] < GQA && head[h] < c.H;
        q_head[h] = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head[h] * c.q_nb2 + seq * c.q_nb3;
        slope[h] = valid[h] ? hrx_fa_split_alibi_slope(c, head[h]) : 1.0f;
    }

    const char * k_head = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;
    const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;

    float l[HEADS_PER_RG];
    float m[HEADS_PER_RG];
    float4 out[HEADS_PER_RG][VEC_PER_THREAD];
#pragma unroll
    for (int h = 0; h < HEADS_PER_RG; ++h) {
        l[h] = 0.0f;
        m[h] = -FLT_MAX * 0.5f;
#pragma unroll
        for (int d = 0; d < VEC_PER_THREAD; ++d) {
            out[h][d] = hrx_fa_split_f4_zero();
        }
    }

    for (long long jb = split_begin; jb < split_end; jb += BC) {
        float scores[HEADS_PER_RG][COLS_PER_THREAD];
#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const long long kv_col = jb + ci * 4 + col_tid;
            const bool valid_col = kv_col < split_end;
            float mask_value = 0.0f;
            if (c.has_mask && valid_col) {
                mask_value = hrx_fa_split_load_f16(reinterpret_cast<const __half *>(mask_row), kv_col * c.mask_nb0);
            }

            float s[HEADS_PER_RG];
#pragma unroll
            for (int h = 0; h < HEADS_PER_RG; ++h) {
                s[h] = 0.0f;
            }
            if (valid_col && (!c.has_mask || mask_value > -60000.0f)) {
                const char * k_row = k_head + kv_col * c.k_nb1;
#pragma unroll
                for (int d = 0; d < VEC_PER_THREAD; ++d) {
                    const int vec_index = d * D_SPLIT + static_cast<int>(d_tid);
                    const int byte_offset_f32 = vec_index * 4 * static_cast<int>(sizeof(float));
                    const int byte_offset_f16 = vec_index * 4 * static_cast<int>(sizeof(__half));
                    const float4 kv = hrx_fa_split_load_f16x4(k_row + byte_offset_f16);
#pragma unroll
                    for (int h = 0; h < HEADS_PER_RG; ++h) {
                        if (valid[h]) {
                            const float4 qv = hrx_fa_split_scale4(
                                hrx_fa_split_load_f32x4(q_head[h] + byte_offset_f32), c.scale);
                            s[h] += hrx_fa_split_dot4(qv, kv);
                        }
                    }
                }
#pragma unroll
                for (int h = 0; h < HEADS_PER_RG; ++h) {
                    s[h] = hrx_fa_split_sum_dsplit(s[h]);
                    if (c.logit_softcap != 0.0f) {
                        s[h] = c.logit_softcap * tanhf(s[h]);
                    }
                    if (c.has_mask) {
                        s[h] += slope[h] * mask_value;
                    }
                }
            } else {
#pragma unroll
                for (int h = 0; h < HEADS_PER_RG; ++h) {
                    s[h] = -FLT_MAX * 0.5f;
                }
            }
#pragma unroll
            for (int h = 0; h < HEADS_PER_RG; ++h) {
                scores[h][ci] = valid[h] ? s[h] : -FLT_MAX * 0.5f;
            }
        }

        float row_max[HEADS_PER_RG];
#pragma unroll
        for (int h = 0; h < HEADS_PER_RG; ++h) {
            row_max[h] = -FLT_MAX * 0.5f;
        }
#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
#pragma unroll
            for (int h = 0; h < HEADS_PER_RG; ++h) {
                row_max[h] = fmaxf(row_max[h], scores[h][ci]);
            }
        }
#pragma unroll
        for (int h = 0; h < HEADS_PER_RG; ++h) {
            row_max[h] = hrx_fa_split_max_cols(row_max[h]);
            const float old_m = m[h];
            m[h] = fmaxf(m[h], row_max[h]);
            const float old_scale = expf(old_m - m[h]);
            l[h] *= old_scale;
#pragma unroll
            for (int d = 0; d < VEC_PER_THREAD; ++d) {
                out[h][d] = hrx_fa_split_scale4(out[h][d], old_scale);
            }
        }

#pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const long long kv_col = jb + ci * 4 + col_tid;
            if (kv_col >= split_end) {
                continue;
            }
            float p[HEADS_PER_RG];
#pragma unroll
            for (int h = 0; h < HEADS_PER_RG; ++h) {
                p[h] = expf(scores[h][ci] - m[h]);
                l[h] += p[h];
            }
            const char * v_row = v_head + kv_col * c.v_nb1;
#pragma unroll
            for (int d = 0; d < VEC_PER_THREAD; ++d) {
                const int vec_index = d * D_SPLIT + static_cast<int>(d_tid);
                const int byte_offset_f16 = vec_index * 4 * static_cast<int>(sizeof(__half));
                const float4 vv = hrx_fa_split_load_f16x4(v_row + byte_offset_f16);
#pragma unroll
                for (int h = 0; h < HEADS_PER_RG; ++h) {
                    out[h][d] = hrx_fa_split_f4_madd(out[h][d], p[h], vv);
                }
            }
        }
    }

#pragma unroll
    for (int h = 0; h < HEADS_PER_RG; ++h) {
        l[h] = hrx_fa_split_sum_cols(l[h]);
#pragma unroll
        for (int d = 0; d < VEC_PER_THREAD; ++d) {
            out[h][d] = hrx_fa_split_sum_cols4(out[h][d]);
        }
    }

    if (col_tid == 0) {
        const size_t partial_count = static_cast<size_t>(c.S) * static_cast<size_t>(c.N) *
            static_cast<size_t>(c.H) * SPLITS;
        float * scratch_o = scratch;
        float * scratch_l = scratch_o + partial_count * D;
        float * scratch_m = scratch_l + partial_count;
#pragma unroll
        for (int h = 0; h < HEADS_PER_RG; ++h) {
            if (!valid[h]) {
                continue;
            }
            const size_t base = (((static_cast<size_t>(seq) * c.N + static_cast<size_t>(token)) *
                c.H + static_cast<size_t>(head[h])) * SPLITS + static_cast<size_t>(split));
            if (d_tid == 0) {
                scratch_l[base] = l[h];
                scratch_m[base] = m[h];
            }
#pragma unroll
            for (int d = 0; d < VEC_PER_THREAD; ++d) {
                const int vec_index = d * D_SPLIT + static_cast<int>(d_tid);
                *reinterpret_cast<float4 *>(scratch_o + base * D + vec_index * 4) = out[h][d];
            }
        }
    }
}

// D=256 / GQA-8 (2 query heads per row_group). Used by the head-dim-256 decode path.
extern "C" __global__ __launch_bounds__(128) void hrx_flash_attn_ext_f32_f16_decode_gqa8_split(
        const float * q,
        const __half * k,
        const __half * v,
        const __half * mask,
        const float * sinks,
        float * dst,
        float * scratch,
        hrx_flash_attn_ext_f32_f16_decode_constants c) {
    hrx_flash_attn_ext_f32_f16_decode_split_impl<256, 8>(q, k, v, mask, sinks, dst, scratch, c);
}

// D=128 / GQA-4 (1 query head per row_group). Used by the head-dim-128 decode path (Llama-3.x).
extern "C" __global__ __launch_bounds__(128) void hrx_flash_attn_ext_f32_f16_decode_gqa4_split(
        const float * q,
        const __half * k,
        const __half * v,
        const __half * mask,
        const float * sinks,
        float * dst,
        float * scratch,
        hrx_flash_attn_ext_f32_f16_decode_constants c) {
    hrx_flash_attn_ext_f32_f16_decode_split_impl<128, 4>(q, k, v, mask, sinks, dst, scratch, c);
}
