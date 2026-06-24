#pragma once

#include <cstddef>
#include <memory>
#include <string>

struct ggml_backend_hrx_catalog {
    std::string catalog_id;
    std::string source;
    size_t source_count = 0;
    size_t artifact_count = 0;
    size_t family_count = 0;
    size_t route_count = 0;
    size_t fusion_count = 0;
};

struct ggml_backend_hrx_catalog_deleter {
    void operator()(ggml_backend_hrx_catalog * catalog) const;
};

using ggml_backend_hrx_catalog_ptr =
    std::unique_ptr<ggml_backend_hrx_catalog, ggml_backend_hrx_catalog_deleter>;

ggml_backend_hrx_catalog_ptr ggml_backend_hrx_load_catalog(
        const char * catalog_dir,
        std::string * out_error);
