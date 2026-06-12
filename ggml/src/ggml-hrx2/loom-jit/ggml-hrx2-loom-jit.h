// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef GGML_HRX2_LOOM_JIT_H_
#define GGML_HRX2_LOOM_JIT_H_

#ifdef GGML_HRX2_USE_HRX_LOOM_JIT

#include "hrx_loom_jit.h"

#define ggml_hrx2_loom_jit_amdgpu_t hrx_loom_jit_amdgpu_t
#define ggml_hrx2_loom_jit_source_format_t hrx_loom_jit_source_format_t
#define GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_TEXT HRX_LOOM_JIT_SOURCE_FORMAT_TEXT
#define GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_BYTECODE HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE
#define ggml_hrx2_loom_jit_amdgpu_options_t hrx_loom_jit_amdgpu_options_t
#define ggml_hrx2_loom_jit_config_binding_t hrx_loom_jit_config_binding_t
#define ggml_hrx2_loom_jit_compile_options_t hrx_loom_jit_compile_options_t
#define ggml_hrx2_loom_jit_compile_result_t hrx_loom_jit_compile_result_t
#define ggml_hrx2_loom_jit_amdgpu_create hrx_loom_jit_amdgpu_create
#define ggml_hrx2_loom_jit_amdgpu_release hrx_loom_jit_amdgpu_release
#define ggml_hrx2_loom_jit_amdgpu_compile hrx_loom_jit_amdgpu_compile
#define ggml_hrx2_loom_jit_compile_result_deinitialize hrx_loom_jit_compile_result_deinitialize

#else

#include "hrx_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_hrx2_loom_jit_amdgpu_s* ggml_hrx2_loom_jit_amdgpu_t;

typedef enum ggml_hrx2_loom_jit_source_format_t {
  GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_TEXT = 0,
  GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_BYTECODE = 1,
} ggml_hrx2_loom_jit_source_format_t;

typedef struct ggml_hrx2_loom_jit_amdgpu_options_t {
  size_t structure_size;
  const char* processor;
  const char* identifier;
} ggml_hrx2_loom_jit_amdgpu_options_t;

typedef struct ggml_hrx2_loom_jit_config_binding_t {
  const char* key;
  const char* value;
} ggml_hrx2_loom_jit_config_binding_t;

typedef struct ggml_hrx2_loom_jit_compile_options_t {
  size_t structure_size;
  const void* source_data;
  size_t source_size;
  ggml_hrx2_loom_jit_source_format_t source_format;
  const char* source_identifier;
  const char* root_symbol;
  const char* module_name;
  const char* artifact_identifier;
  const ggml_hrx2_loom_jit_config_binding_t* config_bindings;
  size_t config_binding_count;
} ggml_hrx2_loom_jit_compile_options_t;

typedef struct ggml_hrx2_loom_jit_compile_result_t {
  void* hsaco_data;
  size_t hsaco_size;
  char* manifest_json;
  size_t manifest_json_size;
  char* compile_report_json;
  size_t compile_report_json_size;
} ggml_hrx2_loom_jit_compile_result_t;

hrx_status_t ggml_hrx2_loom_jit_amdgpu_create(
    const ggml_hrx2_loom_jit_amdgpu_options_t* options,
    ggml_hrx2_loom_jit_amdgpu_t* out_jit);

void ggml_hrx2_loom_jit_amdgpu_release(ggml_hrx2_loom_jit_amdgpu_t jit);

hrx_status_t ggml_hrx2_loom_jit_amdgpu_compile(
    ggml_hrx2_loom_jit_amdgpu_t jit,
    const ggml_hrx2_loom_jit_compile_options_t* options,
    ggml_hrx2_loom_jit_compile_result_t* out_result);

void ggml_hrx2_loom_jit_compile_result_deinitialize(
    ggml_hrx2_loom_jit_compile_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GGML_HRX2_USE_HRX_LOOM_JIT

#endif  // GGML_HRX2_LOOM_JIT_H_
