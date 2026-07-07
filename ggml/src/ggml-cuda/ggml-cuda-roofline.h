#pragma once

// Per-op GPU roofline profiling for the HIP/ROCm backend (see ggml-cuda-roofline.cpp).
// Compiled only when GGML_HIP_ROOFLINE is defined and activated at runtime by the
// environment variable GGML_ROOFLINE_OUT=<path.json>. Every entry point is a no-op
// unless that variable is set.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load and configure rocprofiler-sdk when GGML_ROOFLINE_OUT is set. Call once during
// backend registration, before any GPU stream is created. Idempotent; no-op otherwise.
void ggml_cuda_roofline_init(void);

// Record the GPU architecture (e.g. "gfx1151") stored in the report. No-op unless active.
void ggml_cuda_roofline_set_device(const char * arch);

// Discard everything captured so far (drains pending records first). Call after a
// warmup run so the report covers only the measured run. No-op unless active.
void ggml_cuda_roofline_reset(void);

// Tag the GPU kernels launched for this op so their device time is attributed to it.
// Call once per op, before its kernel(s) are dispatched. No-op unless active.
void ggml_cuda_roofline_begin_op(const struct ggml_tensor * node);

// Override the record of the op tagged by the last begin_op so it covers a fused span of
// node_count nodes (cgraph->nodes[node_idx .. node_idx+node_count-1]): lists every fused op
// and reports the fused group's HBM traffic with intermediates discarded. Call right after a
// successful fusion, before advancing the loop. No-op unless active.
void ggml_cuda_roofline_fuse_ops(const struct ggml_cgraph * cgraph, int node_idx, int node_count);

#ifdef __cplusplus
}
#endif
