#pragma once

#include "hrx_runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_hrx2_device_info {
    hrx_device_t device = nullptr;
    const char * architecture = nullptr;
};

struct ggml_backend_hrx2_catalog;

struct ggml_backend_hrx2_catalog_deleter {
    void operator()(ggml_backend_hrx2_catalog * catalog) const;
};

using ggml_backend_hrx2_catalog_ptr = std::unique_ptr<ggml_backend_hrx2_catalog, ggml_backend_hrx2_catalog_deleter>;

struct ggml_backend_hrx2_config_binding {
    std::string key;
    std::string value;
};

struct ggml_backend_hrx2_config_binding_spec {
    std::string key;
    std::string value;
    std::string value_source;
};

struct ggml_backend_hrx2_kernel_route {
    std::string id;
    std::string family;
    std::string op;
    std::string target_key;
    std::string source_id;
    std::string artifact_id;
    std::string root_symbol;
    std::string export_name;
    std::string loader_format;
    int32_t     priority = 0;
    uint32_t    binding_count = 0;
    uint32_t    parameter_count = 0;
    uint32_t    constant_byte_length = 0;
    uint32_t    workgroup_size[3] = { 1, 1, 1 };
    uint32_t    rows_per_workgroup = 1;
    uint32_t    cols_per_workgroup = 1;
    uint32_t    ncols_min = 0;
    uint32_t    ncols_max = 0;
    uint32_t    n_dims_min = 0;
    uint32_t    n_dims_max = 0;
    uint32_t    nrows_min = 0;
    uint32_t    nrows_max = 0;
    uint32_t    k_min = 0;
    uint32_t    k_max = 0;
    uint32_t    rows_min = 0;
    uint32_t    rows_max = 0;
    uint32_t    cols_min = 0;
    uint32_t    cols_max = 0;
    int8_t      k_pow2_guard = 0;
    int8_t      all_pot_guard = 0;
    uint32_t    k_multiple_of_guard = 0;
    uint32_t    cols_multiple_of_guard = 0;
    uint32_t    ncols_multiple_of_guard = 0;
    bool        pointwise_src0_row_stride_eq_ncols = false;
    bool        pointwise_src1_row_stride_eq_ncols = false;
    bool        pointwise_src1_row_stride_eq_zero = false;
    bool        pointwise_src1_ncols_eq_ncols = false;
    std::string supports_mode;
    std::string supports_glu_op;
    std::string supports_layout;
    std::string specialization_mode;
    std::vector<ggml_backend_hrx2_config_binding_spec> config_bindings;
};

struct ggml_backend_hrx2_provider {
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    ggml_backend_hrx2_kernel_route route;
    std::string cache_key;
    std::string manifest_json;
    std::string compile_report_json;

    ggml_backend_hrx2_provider() = default;
    ggml_backend_hrx2_provider(const ggml_backend_hrx2_provider &) = delete;
    ggml_backend_hrx2_provider & operator=(const ggml_backend_hrx2_provider &) = delete;
    ~ggml_backend_hrx2_provider();
};

ggml_backend_hrx2_catalog_ptr ggml_backend_hrx2_load_catalog();

const ggml_backend_hrx2_kernel_route * ggml_backend_hrx2_catalog_find_route(
        const ggml_backend_hrx2_catalog & catalog,
        const char * route_id);

void ggml_backend_hrx2_catalog_find_routes(
        const ggml_backend_hrx2_catalog & catalog,
        const char * family,
        const char * op,
        std::vector<const ggml_backend_hrx2_kernel_route *> * out_routes);

std::unique_ptr<ggml_backend_hrx2_provider> ggml_backend_hrx2_load_provider(
        const ggml_backend_hrx2_device_info & device,
        const ggml_backend_hrx2_catalog & catalog,
        const ggml_backend_hrx2_kernel_route & route,
        const std::vector<ggml_backend_hrx2_config_binding> & config_bindings,
        const std::string & cache_key);
