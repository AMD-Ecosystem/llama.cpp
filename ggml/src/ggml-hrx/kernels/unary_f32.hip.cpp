#include <hip/hip_runtime.h>

static __device__ __forceinline__ float hrx_sigmoid_value_f32(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

static __device__ __forceinline__ float hrx_softplus_value_f32(float x) {
    return x > 20.0f ? x : __logf(1.0f + __expf(x));
}

static __device__ __forceinline__ float hrx_gelu_value_f32(float x) {
    constexpr float gelu_coef_a = 0.044715f;
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + gelu_coef_a * x * x)));
}

extern "C" __global__ void hrx_silu_f32(const float * src, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        const float x = src[idx];
        dst[idx] = x * hrx_sigmoid_value_f32(x);
    }
}

extern "C" __global__ void hrx_sigmoid_f32(const float * src, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = hrx_sigmoid_value_f32(src[idx]);
    }
}

extern "C" __global__ void hrx_tanh_f32(const float * src, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = tanhf(src[idx]);
    }
}

extern "C" __global__ void hrx_sigmoid_mul_f32(
        const float * src0, const float * src1, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = hrx_sigmoid_value_f32(src0[idx]) * src1[idx];
    }
}

extern "C" __global__ void hrx_softplus_f32(const float * src, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = hrx_softplus_value_f32(src[idx]);
    }
}

extern "C" __global__ void hrx_swiglu_f32(
        const float * src0, const float * src1, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        const float gate = src0[idx];
        dst[idx] = gate * hrx_sigmoid_value_f32(gate) * src1[idx];
    }
}

extern "C" __global__ void hrx_geglu_f32(
        const float * src0, const float * src1, float * dst, long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = hrx_gelu_value_f32(src0[idx]) * src1[idx];
    }
}
