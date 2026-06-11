#include "ggml-hrx2-catalog.h"

#include "ggml-impl.h"
#include "hrx2_embedded_catalog.h"
#include "hrx_loom_jit.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>

namespace {

static bool ggml_hrx2_catalog_check(hrx_status_t status, const char * expression, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length = 0;
    if (hrx_status_is_ok(hrx_status_to_string(status, &message, &length)) && message) {
        GGML_LOG_ERROR("HRX2: %s failed at %s:%d: %.*s\n", expression, file, line, (int) length, message);
        hrx_status_free_message(message);
    } else {
        GGML_LOG_ERROR("HRX2: %s failed at %s:%d\n", expression, file, line);
    }
    hrx_status_ignore(status);
    return false;
}

#define GGML_HRX2_CATALOG_CHECK(expr) ggml_hrx2_catalog_check((expr), #expr, __FILE__, __LINE__)

static std::string ggml_backend_hrx2_read_text_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

static std::string ggml_backend_hrx2_join_path(const std::string & base, const std::string & relative) {
    if (base.empty()) {
        return relative;
    }
    if (base.back() == '/') {
        return base + relative;
    }
    return base + "/" + relative;
}

static bool ggml_backend_hrx2_load_catalog(
        nlohmann::json * out_catalog,
        std::unordered_map<std::string, std::string> * out_sources) {
    std::string catalog_text = ggml_hrx2_embedded_catalog_json();
    std::string catalog_dir;
    if (const char * env = std::getenv("GGML_HRX2_CATALOG_DIR")) {
        catalog_dir = env;
        const std::string catalog_path = ggml_backend_hrx2_join_path(catalog_dir, "catalog.json");
        const std::string disk_catalog = ggml_backend_hrx2_read_text_file(catalog_path);
        if (disk_catalog.empty()) {
            GGML_LOG_ERROR("HRX2: GGML_HRX2_CATALOG_DIR is set but %s could not be read\n", catalog_path.c_str());
            return false;
        }
        catalog_text = disk_catalog;
    }

    try {
        *out_catalog = nlohmann::json::parse(catalog_text);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("HRX2: failed to parse catalog JSON: %s\n", e.what());
        return false;
    }

    out_sources->clear();
    const ggml_hrx2_embedded_source * embedded_sources = ggml_hrx2_embedded_sources();
    for (size_t i = 0; i < ggml_hrx2_embedded_source_count(); ++i) {
        (*out_sources)[embedded_sources[i].id] = std::string(embedded_sources[i].text, embedded_sources[i].text_size);
    }

    if (!catalog_dir.empty() && out_catalog->contains("sources")) {
        for (const auto & item : (*out_catalog)["sources"].items()) {
            const std::string source_id = item.key();
            const auto & source = item.value();
            if (source.contains("text")) {
                (*out_sources)[source_id] = source["text"].get<std::string>();
            } else if (source.contains("path")) {
                const std::string source_path = ggml_backend_hrx2_join_path(catalog_dir, source["path"].get<std::string>());
                const std::string source_text = ggml_backend_hrx2_read_text_file(source_path);
                if (source_text.empty()) {
                    GGML_LOG_ERROR("HRX2: source %s could not be read from %s\n", source_id.c_str(), source_path.c_str());
                    return false;
                }
                (*out_sources)[source_id] = source_text;
            }
        }
    }

    return true;
}

static bool ggml_backend_hrx2_route_from_catalog(
        const nlohmann::json & catalog,
        const char * route_id,
        ggml_backend_hrx2_kernel_route * route) {
    if (!catalog.contains("kernels") || !catalog["kernels"].is_array()) {
        GGML_LOG_ERROR("HRX2: catalog has no kernels array\n");
        return false;
    }
    for (const auto & kernel : catalog["kernels"]) {
        if (kernel.value("id", std::string()) != route_id) {
            continue;
        }
        route->id = kernel.value("id", std::string());
        route->source_id = kernel.value("source_id", std::string());
        route->root_symbol = kernel.value("root_symbol", std::string());
        route->export_name = kernel.value("export_name", std::string());
        route->artifact_format = kernel.value("artifact_format", std::string("amdgpu-hsaco"));
        const auto & abi = kernel["abi"];
        route->binding_count = abi.value("binding_count", 0);
        route->parameter_count = abi.value("parameter_count", 0);
        route->constant_byte_length = abi.value("constant_byte_length", 0);
        const auto & workgroup = kernel["dispatch"]["workgroup_size"];
        for (int i = 0; i < 3; ++i) {
            route->workgroup_size[i] = workgroup.at(i).get<uint32_t>();
        }
        return !route->source_id.empty() && !route->root_symbol.empty() && !route->export_name.empty();
    }
    GGML_LOG_ERROR("HRX2: route %s not found in catalog\n", route_id);
    return false;
}

static bool ggml_backend_hrx2_compile_route(
        const ggml_backend_hrx2_device_info & device,
        const ggml_backend_hrx2_kernel_route & route,
        const std::string & source,
        ggml_backend_hrx2_provider * provider) {
    const char * architecture = device.architecture ? device.architecture : "";
    hrx_loom_jit_amdgpu_t jit = nullptr;
    hrx_loom_jit_amdgpu_options_t jit_options = {
        /* .structure_size = */ sizeof(hrx_loom_jit_amdgpu_options_t),
        /* .processor      = */ architecture,
        /* .identifier     = */ "ggml-hrx2",
    };
    if (!GGML_HRX2_CATALOG_CHECK(hrx_loom_jit_amdgpu_create(&jit_options, &jit))) {
        return false;
    }

    hrx_loom_jit_compile_result_t result = {};
    hrx_loom_jit_compile_options_t compile_options = {
        /* .structure_size      = */ sizeof(hrx_loom_jit_compile_options_t),
        /* .source_data         = */ source.data(),
        /* .source_size         = */ source.size(),
        /* .source_format       = */ HRX_LOOM_JIT_SOURCE_FORMAT_TEXT,
        /* .source_identifier   = */ route.id.c_str(),
        /* .root_symbol         = */ route.root_symbol.c_str(),
        /* .module_name         = */ "ggml_hrx2",
        /* .artifact_identifier = */ route.export_name.c_str(),
    };
    const bool compiled = GGML_HRX2_CATALOG_CHECK(hrx_loom_jit_amdgpu_compile(jit, &compile_options, &result));
    hrx_loom_jit_amdgpu_release(jit);
    if (!compiled) {
        return false;
    }

    hrx_executable_t executable = nullptr;
    const size_t hsaco_size = result.hsaco_size;
    const char * loader_format = route.artifact_format.empty() || route.artifact_format == "amdgpu-hsaco" ?
        nullptr :
        route.artifact_format.c_str();
    const bool loaded = GGML_HRX2_CATALOG_CHECK(hrx_executable_load_data(
        device.device,
        result.hsaco_data,
        hsaco_size,
        loader_format,
        &executable));
    hrx_loom_jit_compile_result_deinitialize(&result);
    if (!loaded) {
        return false;
    }

    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    const bool abi_ok =
        GGML_HRX2_CATALOG_CHECK(hrx_executable_lookup_export_by_name(executable, route.export_name.c_str(), &export_ordinal)) &&
        GGML_HRX2_CATALOG_CHECK(hrx_executable_export_info(executable, export_ordinal, &export_info)) &&
        export_info.binding_count == route.binding_count &&
        export_info.parameter_count == route.parameter_count &&
        export_info.constant_byte_length == route.constant_byte_length;
    if (!abi_ok) {
        GGML_LOG_ERROR(
            "HRX2: route %s ABI mismatch: bindings=%u/%u parameters=%u/%u constants=%u/%u workgroup=%ux%ux%u\n",
            route.id.c_str(),
            export_info.binding_count, route.binding_count,
            export_info.parameter_count, route.parameter_count,
            export_info.constant_byte_length, route.constant_byte_length,
            export_info.workgroup_size[0], export_info.workgroup_size[1], export_info.workgroup_size[2]);
        hrx_executable_release(executable);
        return false;
    }

    provider->executable = executable;
    provider->export_ordinal = export_ordinal;
    provider->export_info = export_info;
    provider->route = route;
    GGML_LOG_INFO(
        "HRX2: JIT compiled %s for %s (%zu bytes HSACO)\n",
        route.export_name.c_str(), architecture, hsaco_size);
    return true;
}

} // namespace

ggml_backend_hrx2_provider::~ggml_backend_hrx2_provider() {
    if (executable) {
        hrx_executable_release(executable);
    }
}

std::unique_ptr<ggml_backend_hrx2_provider> ggml_backend_hrx2_load_provider(
        const ggml_backend_hrx2_device_info & device,
        const char * route_id) {
    nlohmann::json catalog;
    std::unordered_map<std::string, std::string> sources;
    if (!ggml_backend_hrx2_load_catalog(&catalog, &sources)) {
        return nullptr;
    }

    ggml_backend_hrx2_kernel_route route;
    if (!ggml_backend_hrx2_route_from_catalog(catalog, route_id, &route)) {
        return nullptr;
    }

    auto source = sources.find(route.source_id);
    if (source == sources.end() || source->second.empty()) {
        GGML_LOG_ERROR("HRX2: source %s not available\n", route.source_id.c_str());
        return nullptr;
    }

    auto provider = std::make_unique<ggml_backend_hrx2_provider>();
    if (!ggml_backend_hrx2_compile_route(device, route, source->second, provider.get())) {
        return nullptr;
    }
    return provider;
}
