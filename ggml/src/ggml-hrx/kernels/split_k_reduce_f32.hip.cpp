#include <hip/hip_runtime.h>

struct hrx_split_k_reduce_f32_constants {
    long long n;
    long long split;
};

extern "C" __global__ void hrx_split_k_reduce_f32(
        const float * src,
        float * dst,
        hrx_split_k_reduce_f32_constants c) {
    const long long idx =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) *
            static_cast<long long>(__builtin_amdgcn_workgroup_size_x()) +
        static_cast<long long>(__builtin_amdgcn_workitem_id_x());
    if (idx >= c.n) {
        return;
    }

    float sum = 0.0f;
    for (long long s = 0; s < c.split; ++s) {
        sum += src[s * c.n + idx];
    }
    dst[idx] = sum;
}
