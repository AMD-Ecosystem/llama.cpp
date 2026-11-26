#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10,
    MMVQ_PARAMETERS_RDNA3_5,
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA3_5;
#elif defined(RDNA2) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA3_5;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        cc = ggml_cuda_highest_compiled_arch(cc);
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // RDNA3.5 (gfx1151): with several columns the kernel is bound by loading and converting the q8_1 columns,
        // which two rows per block share. One warp per block, as before, so each row is summed in the same order
        // and the results do not change. Q4_0 and Q8_0, and Q4_K up to 4 columns, were not faster with two rows.
        if (ncols_dst == 1) {
            return 1;
        }
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
                return 1;
            case GGML_TYPE_Q4_K:
                return ncols_dst <= 4 ? 1 : 2;
            default:
                return 2;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    int             row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }
    if constexpr (has_fusion) {
        // the blocks after x2_block0 compute the second matrix (one src1, one column: its rows are the dst stride)
        if (fusion.x2 && blockIdx.x >= fusion.x2_block0) {
            vx = fusion.x2;
            dst = fusion.x2_dst;
            stride_col_dst = fusion.x2_nrows;
            row0 = rows_per_cuda_block*(blockIdx.x - fusion.x2_block0);
        }
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    ggml_cuda_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.x2 != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                        nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                ggml_cuda_mm_fusion_args_device fusion_x2 = fusion;
                if (fusion.x2) {
                    // the second matrix's blocks follow the first's
                    fusion_x2.x2_block0 = dims.first.x;
                    dims.first.x += calc_launch_params<type>(c_ncols_dst, fusion.x2_nrows, 1, 1, warp_size, table_id,
                                                             c_small_k, c_halve_iters).first.x;
                }
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion_x2, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x2) {
            GGML_ASSERT(!ids && !fusion->gate && !fusion->shared_up && fusion->x2_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->x2->type == src0->type && fusion->x2->ne[0] == ne00 && fusion->x2->nb[1] == nb01);
            GGML_ASSERT(ggml_is_matrix(fusion->x2) && ggml_is_contiguous(fusion->x2));
            GGML_ASSERT(fusion->x2_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->x2_dst));
            GGML_ASSERT(fusion->x2_dst->ne[0] == fusion->x2->ne[1] && ggml_nrows(fusion->x2_dst) == 1);
            fusion_local.x2       = fusion->x2->data;
            fusion_local.x2_dst   = (float *) fusion->x2_dst->data;
            fusion_local.x2_nrows = fusion->x2->ne[1];
        }

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    // src1 quantized once per graph evaluation: an earlier consumer (or the producing norm) may have made the copy
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    const char * src1_q8_1 = ctx.q8_1_cache.find(src1);
    if (src1_q8_1 == nullptr) {
        char * buf = ctx.q8_1_cache.reserve(nbytes_src1_q8_1);
        const bool cached = buf != nullptr;
        if (!cached) {
            buf = src1_q8_1_pool.alloc(nbytes_src1_q8_1);
        }
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, buf, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        if (cached) {
            ctx.q8_1_cache.commit(src1);
        }
        src1_q8_1 = buf;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

// The hyper-connection pre-mix of qwen4exp fused with the MMVQ product that makes its gate, for one token: warp w of
// block b computes the hc gate rows of position i = 32*b + w, each as mul_mat_vec_q does with one warp per row and
// block (the RDNA3.5 parameters), then dst[i] = scale * sum_s x[i, s] * sigmoid(gate[s*n_embd + i]) as dsv4_hc_pre_f32
// does. The 32 positions of a block make one block of MMVQ's q8_1 copy of dst.
// With quantize_y the warps of the block first quantize y_f32 (ne10 values, at most 32 blocks) to q8_1 in shared memory
// as quantize_q8_1 does.
template <ggml_type type, bool quantize_y>
__launch_bounds__(QK8_1*QK8_1, 1)
static __global__ void mul_mat_vec_q_hc_pre(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const float * __restrict__ y_f32,
        const float * __restrict__ x, float * __restrict__ dst, block_q8_1 * __restrict__ q8_1, const int ncols_x,
        const int ne10, const int stride_row_x, const int n_embd, const int hc, const float scale,
        const float * __restrict__ inj_w, const float * __restrict__ inj_x, float * __restrict__ inj_dst, const int inj_ncols2,
        const int inj_stride_row) {
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr             = get_vdr_mmvq(type);
    constexpr int warp_size       = QK8_1;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    // the injection product goes first, block 0, as its rows take longer than a block of positions
    const int b                = inj_dst != nullptr ? (int) blockIdx.x - 1 : (int) blockIdx.x;
    const int tid              = threadIdx.x;
    const int i                = b*warp_size + threadIdx.y;
    const int blocks_per_row_x = ncols_x / qk;

    if (b < 0) {
        // the extra block: the hc rows of the injection product, each by 8 warps as mul_mat_vec_f sums a row with 256
        // threads (F32 weights)
        constexpr int inj_block_size = 256;
        const int r  = threadIdx.y / (inj_block_size/warp_size);
        const int iw = threadIdx.y % (inj_block_size/warp_size);
        __shared__ float buf_iw[warp_size][warp_size];
        buf_iw[threadIdx.y][tid] = 0.0f;
        __syncthreads();
        float sumf = 0.0f;
        if (r < hc) {
            const float2 * x2 = (const float2 *) (inj_w + (int64_t) r*inj_stride_row);
            const float2 * y2 = (const float2 *) inj_x;
            for (int col2 = iw*warp_size + tid; col2 < inj_ncols2; col2 += inj_block_size) {
                const float2 tmpx = x2[col2];
                const float2 tmpy = y2[col2];
                ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
            }
        }
        sumf = warp_reduce_sum<warp_size>(sumf);
        buf_iw[r][iw] = sumf;
        __syncthreads();
        if (iw == 0 && r < hc) {
            sumf = buf_iw[r][tid];
            sumf = warp_reduce_sum<warp_size>(sumf);
            if (tid == 0) {
                inj_dst[r] = sumf;
            }
        }
        return;
    }

    __shared__ block_q8_1 y_shared[quantize_y ? warp_size : 1];
    if constexpr (quantize_y) {
        if (threadIdx.y < ncols_x / QK8_1) {
            const int   k     = threadIdx.y*QK8_1 + tid;
            const float xi    = k < ne10 ? y_f32[k] : 0.0f;
            const float amax  = warp_reduce_max<QK8_1>(fabsf(xi));
            const float d     = amax / 127.0f;
            const int8_t q    = amax == 0.0f ? 0 : roundf(xi / d);
            const int   sum_q = warp_reduce_sum<QK8_1>((int) q);
            y_shared[threadIdx.y].qs[tid] = q;
            if (tid == 0) {
                y_shared[threadIdx.y].ds = make_half2(d, d * sum_q);
            }
        }
        __syncthreads();
    }
    // not through y, whose restrict lets the compiler move its loads before the barrier
    const block_q8_1 * y_row = quantize_y ? y_shared : y;

    float sum = 0.0f;
    for (int ih = 0; ih < hc; ++ih) {
        const int row = ih*n_embd + i;
        float gate = 0.0f;
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (tid % (qi/vdr));
            gate += vec_dot_q_cuda(vx, &y_row[kby], row*stride_row_x + kbx, kqs);
        }
        gate = warp_reduce_sum<warp_size>(gate);

        const float xv = x[ih*n_embd + i];
        const float wv = 1.0f / (1.0f + expf(-gate));
        sum += xv * wv;
    }

    const float v = scale * sum;
    __shared__ float v_shared[warp_size];
    if (tid == 0) {
        dst[i] = v;
        v_shared[threadIdx.y] = v;
    }
    if (q8_1 == nullptr) {
        return;
    }
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }

    const float  vq    = v_shared[tid];
    const float  amax  = warp_reduce_max<QK8_1>(fabsf(vq));
    const float  d     = amax / 127.0f;
    const int8_t q     = amax == 0.0f ? 0 : roundf(vq / d);
    const int    sum_q = warp_reduce_sum<QK8_1>((int) q);
    block_q8_1 & yb = q8_1[b];
    yb.qs[tid] = q;
    if (tid == 0) {
        yb.ds = make_half2(d, d * sum_q);
    }
}

// the q8_1 copy of src1 (one row) as ggml_cuda_mul_mat_vec_q makes it once per graph evaluation, see ggml_cuda_q8_1_cache
static const char * ggml_cuda_mmvq_src1_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
                                             ggml_cuda_pool_alloc<char> & pool) {
    const char * src1_q8_1 = ctx.q8_1_cache.find(src1);
    if (src1_q8_1 != nullptr) {
        return src1_q8_1;
    }
    const int64_t ne10             = src1->ne[0];
    const int64_t ne10_padded      = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  nbytes_src1_q8_1 = ne10_padded * sizeof(block_q8_1)/QK8_1;
    char * buf = ctx.q8_1_cache.reserve(nbytes_src1_q8_1);
    const bool cached = buf != nullptr;
    if (!cached) {
        buf = pool.alloc(nbytes_src1_q8_1);
    }
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    quantize_row_q8_1_cuda((const float *) src1->data, buf, ne10, s11, s12, s13, ne10_padded, 1, 1, 1, ctx.stream());
    if (cached) {
        ctx.q8_1_cache.commit(src1);
    }
    return buf;
}

// silu(s*(row . y) + b), the MMVQ product for one token followed by SCALE -> SILU, each row summed as mul_mat_vec_q does
// with one warp per row and block (the RDNA3.5 parameters for one column)
template <ggml_type type>
__launch_bounds__(QK8_1, 1)
static __global__ void mul_mat_vec_q_scale_silu(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        float * __restrict__ dst_copy, const int ncols_x, const int stride_row_x, const float s, const float b) {
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr             = get_vdr_mmvq(type);
    constexpr int warp_size       = QK8_1;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid              = threadIdx.x;
    const int row              = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp = 0.0f;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
        tmp += vec_dot_q_cuda(vx, &y[kby], row*stride_row_x + kbx, kqs);
    }
    tmp = warp_reduce_sum<warp_size>(tmp);

    if (tid == 0) {
        const float v = ggml_cuda_op_silu_single(s * tmp + b);
        dst[row] = v;
        if (dst_copy != nullptr) {
            dst_copy[row] = v;
        }
    }
}

void ggml_cuda_mul_mat_vec_q_scale_silu(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * scale_node, ggml_tensor * unary_node, float * dst_copy) {
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    const char * src1_q8_1 = ggml_cuda_mmvq_src1_q8_1(ctx, src1, src1_q8_1_pool);

    float s;
    float b;
    memcpy(&s, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) scale_node->op_params + 1, sizeof(float));

    ctx.q8_1_cache.invalidate(unary_node);
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_scale_silu<GGML_TYPE_Q8_0><<<src0->ne[1], QK8_1, 0, ctx.stream()>>>(
                src0->data, (const block_q8_1 *) src1_q8_1, (float *) unary_node->data, dst_copy, src0->ne[0],
                src0->nb[1] / ggml_type_size(src0->type), s, b);
            break;
        default:
            GGML_ABORT("unsupported type for the fused MMVQ + scale + silu");
    }
}

void ggml_cuda_mul_mat_vec_q_hc_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                    const float * src1_copy, ggml_tensor * hc_pre, const bool write_q8_1,
                                    const ggml_tensor * inject, float * inject_dst) {
    const ggml_tensor * x = hc_pre->src[0];
    cudaStream_t stream = ctx.stream();

    // A short src1 without a q8_1 copy is quantized in the kernel, from src1_copy: the output may overlap src1, whose
    // last reader the fused product is, and the blocks that finish first would overwrite it for the later ones.
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    const char * src1_q8_1  = ctx.q8_1_cache.find(src1);
    const bool   quantize_y = src1_q8_1 == nullptr && src1_copy != nullptr && src1->ne[0] <= QK8_1*QK8_1 &&
        src0->ne[0] <= QK8_1*QK8_1;
    if (src1_q8_1 == nullptr && !quantize_y) {
        src1_q8_1 = ggml_cuda_mmvq_src1_q8_1(ctx, src1, src1_q8_1_pool);
    }

    const int64_t n_embd = x->ne[0];
    const int     hc     = x->ne[1];
    const float   scale  = ggml_get_op_params_f32(hc_pre, 0);

    ctx.q8_1_cache.invalidate(hc_pre);
    block_q8_1 * q8_1 = write_q8_1 ? (block_q8_1 *) ctx.q8_1_cache.reserve(n_embd/QK8_1 * sizeof(block_q8_1)) : nullptr;

    // with the injection product, one more block (the first) computes it, see mul_mat_vec_q_hc_pre
    const float * inj_w          = inject_dst ? (const float *) inject->src[0]->data : nullptr;
    const float * inj_x          = inject_dst ? (const float *) inject->src[1]->data : nullptr;
    const int     inj_ncols2     = inject_dst ? inject->src[0]->ne[0] / 2 : 0;
    const int     inj_stride_row = inject_dst ? inject->src[0]->nb[1] / sizeof(float) : 0;

    const dim3 grid_dims(n_embd/QK8_1 + (inject_dst ? 1 : 0), 1, 1);
    const dim3 block_dims(QK8_1, QK8_1, 1);
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            if (quantize_y) {
                mul_mat_vec_q_hc_pre<GGML_TYPE_Q8_0, true><<<grid_dims, block_dims, 0, stream>>>(
                    src0->data, nullptr, src1_copy, (const float *) x->data, (float *) hc_pre->data, q8_1,
                    src0->ne[0], src1->ne[0], src0->nb[1] / ggml_type_size(src0->type), n_embd, hc, scale,
                    inj_w, inj_x, inject_dst, inj_ncols2, inj_stride_row);
            } else {
                mul_mat_vec_q_hc_pre<GGML_TYPE_Q8_0, false><<<grid_dims, block_dims, 0, stream>>>(
                    src0->data, (const block_q8_1 *) src1_q8_1, nullptr, (const float *) x->data, (float *) hc_pre->data,
                    q8_1, src0->ne[0], src1->ne[0], src0->nb[1] / ggml_type_size(src0->type), n_embd, hc, scale,
                    inj_w, inj_x, inject_dst, inj_ncols2, inj_stride_row);
            }
            break;
        default:
            GGML_ABORT("unsupported type for the fused hyper-connection pre-mix");
    }

    if (q8_1) {
        ctx.q8_1_cache.commit(hc_pre);
    }
}

bool ggml_cuda_mmvq_moe_glu_shared_supported(ggml_type type, ggml_type type_s) {
    return type_s == GGML_TYPE_Q6_K && (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ2_S);
}

// up . y and gate . y for one row with the GLU, as mul_mat_vec_q computes them with one warp per row and block
template <ggml_type type>
static __device__ __forceinline__ float mmvq_moe_glu_row(
        const void * __restrict__ vup, const void * __restrict__ vgate, const block_q8_1 * __restrict__ y,
        const int kbx_offset, const int ncols_x, const ggml_glu_op glu_op, const float glu_limit) {
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr             = get_vdr_mmvq(type);
    constexpr int warp_size       = QK8_1;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid              = threadIdx.x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp      = 0.0f;
    float tmp_gate = 0.0f;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
        tmp      += vec_dot_q_cuda(vup,   &y[kby], kbx_offset + kbx, kqs);
        tmp_gate += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + kbx, kqs);
    }
    tmp      = warp_reduce_sum<warp_size>(tmp);
    tmp_gate = warp_reduce_sum<warp_size>(tmp_gate);

    // the zero biases mul_mat_vec_q adds with the fusion
    float result     = tmp + 0.0f;
    float gate_value = tmp_gate + 0.0f;
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
            result *= ggml_cuda_op_silu_single(gate_value);
            break;
        case GGML_GLU_OP_GEGLU:
            result *= ggml_cuda_op_gelu_single(gate_value);
            break;
        case GGML_GLU_OP_SWIGLU_OAI:
            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
            break;
        case GGML_GLU_OP_SWIGLU_CLAMP:
            result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
            break;
        default:
            result = result * gate_value;
            break;
    }
    return result;
}

// The routed experts' gate and up products with the GLU for one token and, in the first nblocks blocks (so that they
// overlap with the routed ones), those of the shared expert, whose weights may have another type. Warp w of block
// e*nblocks + b computes row 32*b + w of expert e - 1 (e = 0: the shared expert, see mmvq_moe_glu_row).
// The 32 rows of a block make one block of MMVQ's q8_1 copy of the result, whose rows are padded to nblocks_padded
// blocks: the blocks past the rows are zero, as quantize_q8_1 makes them, and written by the blocks with rows.
template <ggml_type type, ggml_type type_s>
__launch_bounds__(QK8_1*QK8_1, 1)
static __global__ void mul_mat_vec_q_moe_glu_shared(
        const void * __restrict__ up, const void * __restrict__ gate, const void * __restrict__ up_s,
        const void * __restrict__ gate_s, const block_q8_1 * __restrict__ y, const int32_t * __restrict__ ids,
        float * __restrict__ dst, float * __restrict__ dst_s, block_q8_1 * __restrict__ q8_1,
        block_q8_1 * __restrict__ q8_1_s, const int ncols_x, const int nrows, const int stride_row_x,
        const int stride_channel_x, const int stride_channel_dst, const int stride_row_x_s, const int nblocks,
        const int nblocks_padded, const ggml_glu_op glu_op, const float glu_limit, const float * __restrict__ sg_w, const float * __restrict__ sg_x,
        const int sg_ncols2, const float * __restrict__ weights, float * __restrict__ scratch, const int n_used) {
    if (scratch != nullptr && blockIdx.x == gridDim.x - 1) {
        // the extra last block: the shared expert's gate (F32 weights), with 8 warps as mul_mat_vec_f sums a row with
        // 256 threads, and copies of the routing for the fused down stage (see ggml_cuda_mul_mat_vec_q_moe_down_shared),
        // whose output may overlap the routing tensors
        constexpr int sg_block_size = 256;
        const int tid = threadIdx.x;
        const int iw  = threadIdx.y;
        __shared__ float buf_iw[QK8_1];
        if (iw == 0) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
        float sumf = 0.0f;
        if (iw < sg_block_size/QK8_1) {
            const float2 * x2 = (const float2 *) sg_w;
            const float2 * y2 = (const float2 *) sg_x;
            for (int col2 = iw*QK8_1 + tid; col2 < sg_ncols2; col2 += sg_block_size) {
                const float2 tmpx = x2[col2];
                const float2 tmpy = y2[col2];
                ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
            }
        }
        sumf = warp_reduce_sum<QK8_1>(sumf);
        if (iw < sg_block_size/QK8_1) {
            buf_iw[iw] = sumf;
        }
        __syncthreads();
        if (iw == 0) {
            sumf = buf_iw[tid];
            sumf = warp_reduce_sum<QK8_1>(sumf);
            if (tid == 0) {
                scratch[0] = sumf;
            }
        } else if (iw == 1 && tid < n_used) {
            scratch[1 + tid] = weights[tid];
            ((int *) scratch)[QK8_1 + tid] = ids[tid];
        }
        return;
    }

    const int  b      = blockIdx.x % nblocks;
    const bool shared = blockIdx.x < nblocks;
    const int  e      = (int) (blockIdx.x / nblocks) - 1;
    const int  row    = b*QK8_1 + threadIdx.y;

    float v = 0.0f;
    if (row < nrows) {
        if (shared) {
            v = mmvq_moe_glu_row<type_s>(up_s, gate_s, y, row*stride_row_x_s, ncols_x, glu_op, glu_limit);
        } else {
            v = mmvq_moe_glu_row<type>(up, gate, y, ids[e]*stride_channel_x + row*stride_row_x, ncols_x, glu_op, glu_limit);
        }
        if (threadIdx.x == 0) {
            if (shared) {
                dst_s[row] = v;
            } else {
                dst[e*stride_channel_dst + row] = v;
            }
        }
    }

    block_q8_1 * q = shared ? q8_1_s : (q8_1 ? q8_1 + e*nblocks_padded : nullptr);
    if (q == nullptr) {
        return;
    }
    __shared__ float v_shared[QK8_1];
    if (threadIdx.x == 0) {
        v_shared[threadIdx.y] = v;
    }
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }
    const int    tid   = threadIdx.x;
    const float  vq    = v_shared[tid];
    const float  amax  = warp_reduce_max<QK8_1>(fabsf(vq));
    const float  d     = amax / 127.0f;
    const int8_t qv    = amax == 0.0f ? 0 : roundf(vq / d);
    const int    sum_q = warp_reduce_sum<QK8_1>((int) qv);
    block_q8_1 & yb = q[b];
    yb.qs[tid] = qv;
    if (tid == 0) {
        yb.ds = make_half2(d, d * sum_q);
    }
    for (int bp = nblocks + b; bp < nblocks_padded; bp += nblocks) {
        q[bp].qs[tid] = 0;
        if (tid == 0) {
            q[bp].ds = make_half2(0.0f, 0.0f);
        }
    }
}

void ggml_cuda_mul_mat_vec_q_moe_glu_shared(ggml_backend_cuda_context & ctx, const ggml_tensor * routed,
                                            const ggml_tensor * shared, const bool write_q8_1, const bool write_q8_1_s,
                                            const ggml_tensor * shared_gate, const ggml_tensor * weights, float * scratch) {
    const ggml_tensor * gate_w   = routed->src[0]->src[0];
    const ggml_tensor * up_mm    = routed->src[1];
    const ggml_tensor * up_w     = up_mm->src[0];
    const ggml_tensor * src1     = up_mm->src[1];
    const ggml_tensor * ids      = up_mm->src[2];
    const ggml_tensor * gate_w_s = shared->src[0]->src[0];
    const ggml_tensor * up_w_s   = shared->src[1]->src[0];

    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    const char * src1_q8_1 = ggml_cuda_mmvq_src1_q8_1(ctx, src1, src1_q8_1_pool);

    const int ncols_x        = up_w->ne[0];
    const int nrows          = up_w->ne[1];
    const int n_used         = routed->ne[1];
    const int nblocks        = (nrows + QK8_1 - 1) / QK8_1;
    const int nblocks_padded = GGML_PAD(nrows, MATRIX_ROW_PADDING) / QK8_1;

    // the q8_1 copies MMVQ reads for the down products, rows padded to MATRIX_ROW_PADDING (see ggml_cuda_q8_1_cache)
    ctx.q8_1_cache.invalidate(routed);
    ctx.q8_1_cache.invalidate(shared);
    block_q8_1 * q8_1 = nullptr;
    if (write_q8_1) {
        q8_1 = (block_q8_1 *) ctx.q8_1_cache.reserve(n_used * nblocks_padded * sizeof(block_q8_1));
        if (q8_1) {
            ctx.q8_1_cache.commit(routed);
        }
    }
    block_q8_1 * q8_1_s = nullptr;
    if (write_q8_1_s) {
        q8_1_s = (block_q8_1 *) ctx.q8_1_cache.reserve(nblocks_padded * sizeof(block_q8_1));
        if (q8_1_s) {
            ctx.q8_1_cache.commit(shared);
        }
    }

    const ggml_glu_op glu_op    = ggml_get_glu_op(routed);
    const float       glu_limit = ggml_get_op_params_f32(routed, 3);

    // with scratch, one more block computes the shared expert's gate and copies the routing there
    const float * sg_w      = scratch ? (const float *) shared_gate->src[0]->data : nullptr;
    const float * sg_x      = scratch ? (const float *) shared_gate->src[1]->data : nullptr;
    const int     sg_ncols2 = scratch ? shared_gate->src[0]->ne[0] / 2 : 0;

    const dim3 grid_dims(nblocks*(n_used + 1) + (scratch ? 1 : 0), 1, 1);
    const dim3 block_dims(QK8_1, QK8_1, 1);
#define MMVQ_MOE_GLU_SHARED_CASE(T, TS) \
    if (up_w->type == T && up_w_s->type == TS) { \
        mul_mat_vec_q_moe_glu_shared<T, TS><<<grid_dims, block_dims, 0, ctx.stream()>>>( \
            up_w->data, gate_w->data, up_w_s->data, gate_w_s->data, (const block_q8_1 *) src1_q8_1, \
            (const int32_t *) ids->data, (float *) routed->data, (float *) shared->data, q8_1, q8_1_s, ncols_x, nrows, \
            up_w->nb[1] / ggml_type_size(T), up_w->nb[2] / ggml_type_size(T), routed->nb[1] / sizeof(float), \
            up_w_s->nb[1] / ggml_type_size(TS), nblocks, nblocks_padded, glu_op, glu_limit, sg_w, sg_x, sg_ncols2, \
            scratch ? (const float *) weights->data : nullptr, scratch, n_used); \
        return; \
    }
    MMVQ_MOE_GLU_SHARED_CASE(GGML_TYPE_IQ3_XXS, GGML_TYPE_Q6_K)
    MMVQ_MOE_GLU_SHARED_CASE(GGML_TYPE_IQ4_XS,  GGML_TYPE_Q6_K)
    MMVQ_MOE_GLU_SHARED_CASE(GGML_TYPE_IQ2_S,   GGML_TYPE_Q6_K)
#undef MMVQ_MOE_GLU_SHARED_CASE
    GGML_ABORT("unsupported types for the fused routed + shared expert gate/up");
}

bool ggml_cuda_mmvq_moe_down_shared_supported(ggml_type type, ggml_type type_s) {
    return type_s == GGML_TYPE_Q8_0 && (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q8_0 ||
        type == GGML_TYPE_Q4_0);
}

// row . y for one row, as mul_mat_vec_q computes it with one warp per row and block
template <ggml_type type>
static __device__ __forceinline__ float mmvq_dot_row(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int kbx_offset, const int ncols_x) {
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr             = get_vdr_mmvq(type);
    constexpr int warp_size       = QK8_1;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid              = threadIdx.x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp = 0.0f;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
        tmp += vec_dot_q_cuda(vx, &y[kby], kbx_offset + kbx, kqs);
    }
    return warp_reduce_sum<warp_size>(tmp);
}

bool ggml_cuda_mmvq_norm_gate_supported(ggml_type type) {
    return type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0;
}

// as op_sigmoid in unary.cu
static __device__ __forceinline__ float mmvq_norm_gate_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// dst = sigmoid(x . y) * (the RMS norm of the row of 128 values of nx holding each row, times nw), the MMVQ product for
// one token (one warp per row, see mmvq_dot_row) followed by the gated output norm of a gated delta net as
// rms_norm_f32<32, 1, true, true> and unary_gated_op_q8_1_kernel compute it, with MMVQ's q8_1 copy of dst: the 32 rows
// of a block make one of its blocks.
template <ggml_type type>
__launch_bounds__(QK8_1*QK8_1, 1)
static __global__ void mul_mat_vec_q_norm_gate(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const float * __restrict__ nx,
        const float * __restrict__ nw, float * __restrict__ dst, block_q8_1 * __restrict__ q8_1, const int ncols_x,
        const int stride_row_x, const float eps) {
    constexpr int ncols_norm = 128;
    const int tid = threadIdx.x;
    const int row = blockIdx.x*QK8_1 + threadIdx.y;

    const float z = mmvq_dot_row<type>(vx, y, row*stride_row_x, ncols_x);

    const int    h  = row / ncols_norm;
    const float4 xs = ((const float4 *) (nx + h*ncols_norm))[tid];
    float tmp = 0.0f;
    tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    tmp = warp_reduce_sum(tmp);
    const float scale = rsqrtf(tmp / ncols_norm + eps);

    __shared__ float v_shared[QK8_1];
    if (tid == 0) {
        const int c = row % ncols_norm;
        float n = scale*nx[h*ncols_norm + c];
        n *= nw[c];
        const float v = mmvq_norm_gate_sigmoid(z) * n;
        dst[row] = v;
        v_shared[threadIdx.y] = v;
    }
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }
    const float  v     = v_shared[tid];
    const float  amax  = warp_reduce_max<QK8_1>(fabsf(v));
    const float  d     = amax / 127.0f;
    const int8_t q     = amax == 0.0f ? 0 : roundf(v / d);
    const int    sum_q = warp_reduce_sum<QK8_1>((int) q);
    block_q8_1 & yb = q8_1[blockIdx.x];
    yb.qs[tid] = q;
    if (tid == 0) {
        yb.ds = make_half2(d, d * sum_q);
    }
}

void ggml_cuda_mul_mat_vec_q_norm_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * rms,
                                       const ggml_tensor * norm_w, ggml_tensor * dst, const ggml_tensor * q8_1_tensor) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];

    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    const char * src1_q8_1 = ggml_cuda_mmvq_src1_q8_1(ctx, src1, src1_q8_1_pool);

    // the q8_1 copy MMVQ reads through q8_1_tensor, or one nothing reads if the cache has no room
    const size_t nbytes_q8_1 = ggml_nrows(q8_1_tensor) * q8_1_tensor->ne[0]/QK8_1 * sizeof(block_q8_1);
    ctx.q8_1_cache.invalidate(dst);
    ctx.q8_1_cache.invalidate(q8_1_tensor);
    ggml_cuda_pool_alloc<char> q8_1_pool(ctx.pool());
    block_q8_1 * q8_1 = (block_q8_1 *) ctx.q8_1_cache.reserve(nbytes_q8_1);
    const bool cached = q8_1 != nullptr;
    if (!cached) {
        q8_1 = (block_q8_1 *) q8_1_pool.alloc(nbytes_q8_1);
    }

    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));

    const int  nrows = src0->ne[1];
    const dim3 grid_dims(nrows / QK8_1, 1, 1);
    const dim3 block_dims(QK8_1, QK8_1, 1);
    switch (src0->type) {
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_norm_gate<GGML_TYPE_Q6_K><<<grid_dims, block_dims, 0, ctx.stream()>>>(
                src0->data, (const block_q8_1 *) src1_q8_1, (const float *) rms->src[0]->data, (const float *) norm_w->data,
                (float *) dst->data, q8_1, src0->ne[0], src0->nb[1] / ggml_type_size(GGML_TYPE_Q6_K), eps);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_norm_gate<GGML_TYPE_Q8_0><<<grid_dims, block_dims, 0, ctx.stream()>>>(
                src0->data, (const block_q8_1 *) src1_q8_1, (const float *) rms->src[0]->data, (const float *) norm_w->data,
                (float *) dst->data, q8_1, src0->ne[0], src0->nb[1] / ggml_type_size(GGML_TYPE_Q8_0), eps);
            break;
        default:
            GGML_ABORT("unsupported type for the fused gated norm");
    }
    if (cached) {
        ctx.q8_1_cache.commit(q8_1_tensor);
    }
}

// the weighted sum of the experts' rows as moe_weighted_reduction_f32 computes it
static __device__ __forceinline__ float moe_weighted_sum(
        const float * experts, const float * expert_scale, const float * weights, const int n_expert_used) {
    const float first_scale = expert_scale != nullptr ? expert_scale[0] : 1.0f;
    float       sum         = (experts[0] * first_scale) * weights[0];
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const float scale = expert_scale != nullptr ? expert_scale[expert] : 1.0f;
        sum += (experts[expert] * scale) * weights[expert];
    }
    return sum;
}

static __device__ __forceinline__ float moe_shared_gate_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// a + x*sigmoid(g) as unary_mul_add_kernel computes it, the product rounded before the sum
static __device__ __forceinline__ float moe_shared_gated_add(const float a, const float x, const float g) {
#ifdef GGML_USE_HIP
#pragma clang fp contract(off)
    const float m = x * moe_shared_gate_sigmoid(g);
    return a + m;
#else
    return __fadd_rn(a, __fmul_rn(x, moe_shared_gate_sigmoid(g)));
#endif // GGML_USE_HIP
}

// The down stage of the routed experts and the shared expert for one token: block r computes row r, warp e < n_used
// that of routed expert ids[e] and warp n_used that of the shared expert, then dst[r] = the routing-weighted sum of the
// routed rows + shared row*sigmoid(gate). The routing and the gate come from scratch, see mul_mat_vec_q_moe_glu_shared.
template <ggml_type type, ggml_type type_s>
__launch_bounds__(QK8_1*QK8_1, 1)
static __global__ void mul_mat_vec_q_moe_down_shared(
        const void * __restrict__ down, const void * __restrict__ down_s, const block_q8_1 * __restrict__ y,
        const block_q8_1 * __restrict__ y_s, const float * __restrict__ scratch, const float * __restrict__ expert_scale,
        float * __restrict__ dst, const int ncols_x, const int stride_row_x, const int stride_channel_x,
        const int stride_row_x_s, const int stride_channel_y, const int n_used) {
    const int row = blockIdx.x;
    const int w   = threadIdx.y;

    float v;
    if (w < n_used) {
        const int e = ((const int *) scratch)[QK8_1 + w];
        v = mmvq_dot_row<type>(down, y + w*stride_channel_y, e*stride_channel_x + row*stride_row_x, ncols_x);
    } else {
        v = mmvq_dot_row<type_s>(down_s, y_s, row*stride_row_x_s, ncols_x);
    }

    __shared__ float rows[QK8_1];
    if (threadIdx.x == 0) {
        rows[w] = v;
    }
    __syncthreads();
    if (w == 0 && threadIdx.x == 0) {
        const float sum = moe_weighted_sum(rows, expert_scale, scratch + 1, n_used);
        dst[row] = moe_shared_gated_add(sum, rows[n_used], scratch[0]);
    }
}

void ggml_cuda_mul_mat_vec_q_moe_down_shared(ggml_backend_cuda_context & ctx, const ggml_tensor * down,
                                             const ggml_tensor * shared_down, ggml_tensor * dst, const float * scratch,
                                             const char * y_q8_1, const char * y_s_q8_1) {
    const ggml_tensor * w   = down->src[0];
    const ggml_tensor * w_s = shared_down->src[0];
    const int ncols_x = w->ne[0];
    const int n_used  = down->ne[1];
    const int stride_channel_y = GGML_PAD(ncols_x, MATRIX_ROW_PADDING) / QK8_1;

    ctx.q8_1_cache.invalidate(dst);
    const dim3 grid_dims(w->ne[1], 1, 1);
    const dim3 block_dims(QK8_1, n_used + 1, 1);
#define MMVQ_MOE_DOWN_SHARED_CASE(T, TS) \
    if (w->type == T && w_s->type == TS) { \
        mul_mat_vec_q_moe_down_shared<T, TS><<<grid_dims, block_dims, 0, ctx.stream()>>>( \
            w->data, w_s->data, (const block_q8_1 *) y_q8_1, (const block_q8_1 *) y_s_q8_1, scratch, nullptr, \
            (float *) dst->data, ncols_x, w->nb[1] / ggml_type_size(T), w->nb[2] / ggml_type_size(T), \
            w_s->nb[1] / ggml_type_size(TS), stride_channel_y, n_used); \
        return; \
    }
    MMVQ_MOE_DOWN_SHARED_CASE(GGML_TYPE_Q5_0,   GGML_TYPE_Q8_0)
    MMVQ_MOE_DOWN_SHARED_CASE(GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0)
    MMVQ_MOE_DOWN_SHARED_CASE(GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0)
    MMVQ_MOE_DOWN_SHARED_CASE(GGML_TYPE_Q4_0,   GGML_TYPE_Q8_0)
#undef MMVQ_MOE_DOWN_SHARED_CASE
    GGML_ABORT("unsupported types for the fused routed + shared expert down stage");
}
