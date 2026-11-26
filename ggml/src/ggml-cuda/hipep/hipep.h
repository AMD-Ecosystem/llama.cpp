#pragma once

// ggml-hip with hip-ep (GGML_HIP_HIPEP): each split runs by the code libhipep_backend compiles of it,
// and what it does not compile, or cannot run yet, by ggml-hip. GGML_HIP_HIPEP=0 turns it off.

#include "ggml-backend-impl.h"

#include <functional>

bool ggml_cuda_hipep_enabled();

// ggml_backend_reads_split_outputs_t: the scheduler marks what each split hands on
bool ggml_cuda_hipep_reads_split_outputs();

// runs `cgraph`, its compiled runs by hip-ep and the rest by `run_rest`, after `stream` has done its work
// false: none of it is compiled, the caller runs it all; `state` is the backend's, made on first use
bool ggml_cuda_hipep_graph_compute(int device, void * stream, void ** state, ggml_cgraph * cgraph,
                                   const std::function<void(ggml_cgraph *)> & run_rest);

// compiles the splits of a graph the scheduler reserves, before the first step runs one
void ggml_cuda_hipep_precompile(int device, void ** state, ggml_cgraph * cgraph,
                                const ggml_backend_graph_optimize_params * params);

void ggml_cuda_hipep_free(void * state);
