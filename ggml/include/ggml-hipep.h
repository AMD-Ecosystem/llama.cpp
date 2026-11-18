#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_HIPEP_NAME "HIPEP"

// A device HIPEP<n> over each ROCm device: the splits the scheduler gives it
// are compiled and run by libhipep_backend (hip-ep); what it does not
// compile, ggml-hip runs.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hipep_reg(void);

#ifdef  __cplusplus
}
#endif
