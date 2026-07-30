#pragma once

// Autotuning entry point for the MMQ (mul_mat_q) kernel, built only with -DGGML_MMQ_TUNE=ON.
//
// Launches mul_mat_q with a caller-chosen tile width instead of the one mul_mat_q_case would
// pick, for dense and MoE shapes alike. mmq_y and the warp count are compile-time constants
// (GGML_CUDA_MMQ_Y_RDNA3_5 / GGML_CUDA_MMQ_NWARPS_RDNA3_5 in mmq.cuh); they are reported back so
// results from builds with different values can be merged.

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// mmq_x value meaning "let the production heuristic choose" - used as the baseline measurement.
#define GGML_MMQ_TUNE_MMQ_X_AUTO (-1)

struct ggml_mmq_tune_case {
    enum ggml_type type;      // weight type of src0
    int64_t        n;         // src0 rows (output columns)
    int64_t        k;         // reduction dimension
    int64_t        m;         // tokens
    int            n_experts; // 0 = dense MUL_MAT, >0 = MoE MUL_MAT_ID
    int            top_k;     // experts per token (MoE only)
    float          zipf_s;    // routing skew (MoE only); 0 = uniform
    int            nwarmup;
    int            niter;
};

struct ggml_mmq_tune_point {
    int    mmq_x;
    bool   valid;     // false when the config is rejected (granularity or shared memory)
    double us_median;
    double us_min;
    double checksum;  // position-weighted sum of |dst|; configs that disagree computed different results
};

// Runs one case over the given mmq_x list, writing n_mmq_x results into out.
// Buffers are allocated once for the whole list. Returns false if the case itself is unusable.
GGML_BACKEND_API bool ggml_mmq_tune_sweep(
        const struct ggml_mmq_tune_case * tcase,
        const int * mmq_x_list, int n_mmq_x,
        struct ggml_mmq_tune_point * out);

GGML_BACKEND_API int ggml_mmq_tune_mmq_y(void);
GGML_BACKEND_API int ggml_mmq_tune_nwarps(void);

#ifdef __cplusplus
}
#endif
