#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_backend_hrx_test_case {
    std::string id;
    std::string op;
    std::string family;
    std::string expected_route_id;
    std::string src0_type;
    std::string src1_type;
    std::string dst_type;
    int64_t k = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    float tolerance = 0.0f;
    uint32_t repeat = 1;
};

struct ggml_backend_hrx_test_route_record {
    size_t dispatch_count = 0;
    size_t jit_compile_count = 0;
    size_t jit_cache_hit_count = 0;
};

GGML_BACKEND_API std::vector<ggml_backend_hrx_test_case> ggml_backend_hrx_test_cases(
    ggml_backend_dev_t dev,
    const std::string & family);

GGML_BACKEND_API void ggml_backend_hrx_test_reset_dispatch_record();

GGML_BACKEND_API ggml_backend_hrx_test_route_record ggml_backend_hrx_test_get_route_record(
    const std::string & route_id);
