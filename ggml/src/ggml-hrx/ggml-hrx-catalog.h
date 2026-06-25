#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend_hrx_catalog_artifact {
    std::string id;
    std::string path;
    std::string format;
    std::vector<uint8_t> data;
};

struct ggml_backend_hrx_catalog_binding {
    std::string key;
    std::string value;
    std::string source;
};

struct ggml_backend_hrx_catalog_constraint {
    std::string source;
    std::string eq_source;
    bool has_eq_value = false;
    int64_t eq_value = 0;
};

struct ggml_backend_hrx_catalog_route {
    std::string id;
    std::string family;
    std::string op;
    std::string target_key;
    std::string source_id;
    std::string artifact_id;
    std::string root_symbol;
    std::string export_name;
    std::map<std::string, int64_t> shape_min;
    std::map<std::string, int64_t> shape_max;
    std::map<std::string, int64_t> shape_multiple_of;
    std::map<std::string, std::string> supports;
    std::vector<ggml_backend_hrx_catalog_constraint> constraints;
    std::vector<ggml_backend_hrx_catalog_binding> bindings;
    std::vector<std::string> workload_argument_sources;
    uint32_t binding_count = 0;
    uint32_t parameter_count = 0;
    uint32_t constant_byte_length = 0;
};

struct ggml_backend_hrx_catalog_problem {
    std::string op;
    std::string target_key;
    std::map<std::string, std::string> supports;
    std::map<std::string, int64_t> shape;
    std::map<std::string, int64_t> facts;
};

struct ggml_backend_hrx_catalog_test_case {
    std::string id;
    std::string target_key;
    std::string family;
    std::string op;
    std::string expected_route_id;
    std::map<std::string, std::string> supports;
    std::map<std::string, int64_t> shape;
    float tolerance = 0.0f;
    uint32_t repeat = 1;
};

struct ggml_backend_hrx_catalog {
    std::string catalog_id;
    std::string source;
    size_t source_count = 0;
    size_t artifact_count = 0;
    size_t family_count = 0;
    size_t route_count = 0;
    size_t fusion_count = 0;
    size_t test_case_count = 0;
    std::vector<ggml_backend_hrx_catalog_artifact> artifacts;
    std::vector<ggml_backend_hrx_catalog_route> routes;
    std::vector<ggml_backend_hrx_catalog_test_case> test_cases;
    std::unordered_map<std::string, size_t> artifact_by_id;
    std::unordered_map<std::string, std::vector<size_t>> routes_by_target_op;
    std::unordered_map<std::string, std::vector<size_t>> test_cases_by_target_family;
};

struct ggml_backend_hrx_catalog_deleter {
    void operator()(ggml_backend_hrx_catalog * catalog) const;
};

using ggml_backend_hrx_catalog_ptr =
    std::unique_ptr<ggml_backend_hrx_catalog, ggml_backend_hrx_catalog_deleter>;

ggml_backend_hrx_catalog_ptr ggml_backend_hrx_load_catalog(
        const char * catalog_dir,
        std::string * out_error);

const ggml_backend_hrx_catalog_artifact * ggml_backend_hrx_catalog_find_artifact(
        const ggml_backend_hrx_catalog & catalog,
        const std::string & artifact_id);

const ggml_backend_hrx_catalog_route * ggml_backend_hrx_catalog_find_route(
        const ggml_backend_hrx_catalog & catalog,
        const ggml_backend_hrx_catalog_problem & problem);

bool ggml_backend_hrx_catalog_make_config_bindings(
        const ggml_backend_hrx_catalog_route & route,
        const ggml_backend_hrx_catalog_problem & problem,
        std::vector<ggml_backend_hrx_catalog_binding> * out_bindings,
        std::string * out_error);
