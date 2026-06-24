#include "ggml-hrx-catalog.h"

#include "hrx_embedded_catalog.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>
#include <new>
#include <string>

namespace {

static bool ggml_backend_hrx_read_text_file(const std::string & path, std::string * out_text) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    *out_text = std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return true;
}

static std::string ggml_backend_hrx_join_path(const std::string & base, const char * leaf) {
    if (base.empty()) {
        return leaf;
    }
    if (base.back() == '/') {
        return base + leaf;
    }
    return base + "/" + leaf;
}

static bool ggml_backend_hrx_parse_catalog_json(
        const std::string & text,
        const std::string & source,
        ggml_backend_hrx_catalog * catalog,
        std::string * out_error) {
    nlohmann::json value;
    try {
        value = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception & e) {
        if (out_error) {
            *out_error = "failed to parse " + source + ": " + e.what();
        }
        return false;
    }

    auto require = [&](bool condition, const char * message) {
        if (!condition && out_error && out_error->empty()) {
            *out_error = source + ": " + message;
        }
        return condition;
    };

    if (!require(value.is_object(), "catalog must be an object") ||
        !require(value.value("schema", "") == "ggml-hrx-catalog-v1", "schema must be ggml-hrx-catalog-v1") ||
        !require(value.contains("catalog_id") && value["catalog_id"].is_string(), "catalog_id must be a string") ||
        !require(value.contains("sources") && value["sources"].is_object(), "sources must be an object") ||
        !require(value.contains("artifacts") && value["artifacts"].is_object(), "artifacts must be an object") ||
        !require(value.contains("families") && value["families"].is_array(), "families must be an array") ||
        !require(value.contains("routes") && value["routes"].is_array(), "routes must be an array")) {
        return false;
    }
    if (value.contains("fusions") && !require(value["fusions"].is_array(), "fusions must be an array")) {
        return false;
    }

    catalog->catalog_id = value["catalog_id"].get<std::string>();
    catalog->source = source;
    catalog->source_count = value["sources"].size();
    catalog->artifact_count = value["artifacts"].size();
    catalog->family_count = value["families"].size();
    catalog->route_count = value["routes"].size();
    catalog->fusion_count = value.value("fusions", nlohmann::json::array()).size();
    return true;
}

} // namespace

void ggml_backend_hrx_catalog_deleter::operator()(ggml_backend_hrx_catalog * catalog) const {
    delete catalog;
}

ggml_backend_hrx_catalog_ptr ggml_backend_hrx_load_catalog(
        const char * catalog_dir,
        std::string * out_error) {
    if (out_error) {
        out_error->clear();
    }

    std::string text;
    std::string source = "embedded catalog";
    if (catalog_dir && catalog_dir[0]) {
        source = ggml_backend_hrx_join_path(catalog_dir, "catalog.json");
        if (!ggml_backend_hrx_read_text_file(source, &text)) {
            if (out_error) {
                *out_error = "failed to read HRX catalog: " + source;
            }
            return nullptr;
        }
    } else {
        text = ggml_hrx_embedded_catalog_json();
    }

    ggml_backend_hrx_catalog_ptr catalog(new (std::nothrow) ggml_backend_hrx_catalog());
    if (!catalog) {
        if (out_error) {
            *out_error = "failed to allocate HRX catalog";
        }
        return nullptr;
    }
    if (!ggml_backend_hrx_parse_catalog_json(text, source, catalog.get(), out_error)) {
        return nullptr;
    }
    return catalog;
}
