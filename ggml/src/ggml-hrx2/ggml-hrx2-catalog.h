#pragma once

#include "hrx_runtime.h"

#include <cstdint>
#include <memory>
#include <string>

struct ggml_backend_hrx2_device_info {
    hrx_device_t device = nullptr;
    const char * architecture = nullptr;
};

struct ggml_backend_hrx2_kernel_route {
    std::string id;
    std::string source_id;
    std::string root_symbol;
    std::string export_name;
    std::string artifact_format;
    uint32_t    binding_count = 0;
    uint32_t    parameter_count = 0;
    uint32_t    constant_byte_length = 0;
    uint32_t    workgroup_size[3] = { 1, 1, 1 };
};

struct ggml_backend_hrx2_provider {
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    ggml_backend_hrx2_kernel_route route;

    ggml_backend_hrx2_provider() = default;
    ggml_backend_hrx2_provider(const ggml_backend_hrx2_provider &) = delete;
    ggml_backend_hrx2_provider & operator=(const ggml_backend_hrx2_provider &) = delete;
    ~ggml_backend_hrx2_provider();
};

std::unique_ptr<ggml_backend_hrx2_provider> ggml_backend_hrx2_load_provider(
        const ggml_backend_hrx2_device_info & device,
        const char * route_id);
