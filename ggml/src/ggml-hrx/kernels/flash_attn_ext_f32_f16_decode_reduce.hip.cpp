#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct hrx_flash_attn_ext_f32_f16_decode_reduce_constants {
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

extern "C" __global__ __launch_bounds__(256) void hrx_flash_attn_ext_f32_f16_decode_reduce(
        const float * scratch,
        const float * sinks,
        float * dst,
        hrx_flash_attn_ext_f32_f16_decode_reduce_constants c) {
    constexpr int SPLITS = 8;
    const long long D = c.D;

    const long long head = __builtin_amdgcn_workgroup_id_x();
    const long long token = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (head >= c.H || token >= c.N || seq >= c.S) {
        return;
    }

    __shared__ float split_scales[SPLITS];
    __shared__ float inv_l_shared;

    const size_t partial_count = static_cast<size_t>(c.S) * static_cast<size_t>(c.N) *
        static_cast<size_t>(c.H) * SPLITS;
    const float * scratch_o = scratch;
    const float * scratch_l = scratch_o + partial_count * D;
    const float * scratch_m = scratch_l + partial_count;
    const size_t base = (((static_cast<size_t>(seq) * c.N + static_cast<size_t>(token)) *
        c.H + static_cast<size_t>(head)) * SPLITS);

    if (tid == 0) {
        float m = -FLT_MAX * 0.5f;
#pragma unroll
        for (int split = 0; split < SPLITS; ++split) {
            m = fmaxf(m, scratch_m[base + split]);
        }

        float l = 0.0f;
#pragma unroll
        for (int split = 0; split < SPLITS; ++split) {
            const float scale = expf(scratch_m[base + split] - m);
            split_scales[split] = scale;
            l += scale * scratch_l[base + split];
        }

        if (c.has_sinks) {
            const float sink = sinks[head];
            if (sink > m) {
                const float rescale = expf(m - sink);
                l = l * rescale + 1.0f;
#pragma unroll
                for (int split = 0; split < SPLITS; ++split) {
                    split_scales[split] *= rescale;
                }
            } else {
                l += expf(sink - m);
            }
        }

        inv_l_shared = l == 0.0f ? 0.0f : 1.0f / l;
    }
    __syncthreads();

    const float inv_l = inv_l_shared;
    char * dst_head = reinterpret_cast<char *>(dst) + head * c.dst_nb1 + token * c.dst_nb2 + seq * c.dst_nb3;
    for (long long d = tid; d < D; d += 256) {
        float acc = 0.0f;
#pragma unroll
        for (int split = 0; split < SPLITS; ++split) {
            acc += split_scales[split] * scratch_o[(base + split) * D + static_cast<size_t>(d)];
        }
        *reinterpret_cast<float *>(dst_head + d * static_cast<long long>(sizeof(float))) = acc * inv_l;
    }
}
