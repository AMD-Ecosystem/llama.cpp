#include "ggml-hrx2-catalog.h"

#include "ggml-impl.h"
#include "hrx2_embedded_catalog.h"
#include "loom-jit/ggml-hrx2-loom-jit.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend_hrx2_catalog {
    struct source {
        std::string id;
        std::string path;
        std::string text;
    };

    struct artifact {
        std::string id;
        std::string path;
        std::string format;
        std::vector<unsigned char> data;
    };

    std::string catalog_id;
    std::unordered_map<std::string, source> sources;
    std::unordered_map<std::string, artifact> artifacts;
    std::vector<ggml_backend_hrx2_kernel_route> routes;
};

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

static std::vector<unsigned char> ggml_backend_hrx2_read_binary_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
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

static std::string ggml_backend_hrx2_safe_filename(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char c : value) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == '-') {
            result += static_cast<char>(c);
        } else {
            result += '_';
        }
    }
    if (result.empty()) {
        return "provider";
    }
    if (result.size() <= 180) {
        return result;
    }
    char suffix[32] = {};
    std::snprintf(suffix, sizeof(suffix), "_%016zx", std::hash<std::string>{}(value));
    return result.substr(0, 180 - std::strlen(suffix)) + suffix;
}

static void ggml_backend_hrx2_write_text_file(const std::filesystem::path & path, const std::string & text) {
    std::ofstream output(path, std::ios::binary);
    if (output) {
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
}

static void ggml_backend_hrx2_write_binary_file(const std::filesystem::path & path, const void * data, size_t size) {
    std::ofstream output(path, std::ios::binary);
    if (output && data && size != 0) {
        output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    }
}

static void ggml_backend_hrx2_dump_provider_evidence(
        const ggml_backend_hrx2_kernel_route & route,
        const std::vector<ggml_backend_hrx2_config_binding> & config_bindings,
        const std::string & cache_key,
        const char * architecture,
        const char * source_identifier,
        const char * source_format,
        const void * hsaco_data,
        size_t hsaco_size,
        const ggml_backend_hrx2_provider & provider) {
    const char * evidence_dir_env = std::getenv("GGML_HRX2_EVIDENCE_DIR");
    if (!evidence_dir_env || evidence_dir_env[0] == '\0') {
        return;
    }

    const std::filesystem::path provider_dir =
        std::filesystem::path(evidence_dir_env) / ggml_backend_hrx2_safe_filename(cache_key);
    std::error_code ec;
    std::filesystem::create_directories(provider_dir, ec);
    if (ec) {
        GGML_LOG_ERROR("HRX2: failed to create evidence dir %s: %s\n", provider_dir.string().c_str(), ec.message().c_str());
        return;
    }

    if (!provider.compile_report_json.empty()) {
        ggml_backend_hrx2_write_text_file(provider_dir / "compile_report.json", provider.compile_report_json);
    }
    if (!provider.manifest_json.empty()) {
        ggml_backend_hrx2_write_text_file(provider_dir / "manifest.json", provider.manifest_json);
    }
    ggml_backend_hrx2_write_binary_file(provider_dir / "kernel.hsaco", hsaco_data, hsaco_size);

    nlohmann::json metadata = {
        { "route_id", route.id },
        { "family", route.family },
        { "op", route.op },
        { "target_key", architecture ? architecture : "" },
        { "cache_key", cache_key },
        { "root_symbol", route.root_symbol },
        { "export_name", route.export_name },
        { "source_identifier", source_identifier ? source_identifier : "" },
        { "source_format", source_format ? source_format : "" },
        { "hsaco_size", hsaco_size },
        { "compile_report_bytes", provider.compile_report_json.size() },
        { "manifest_bytes", provider.manifest_json.size() },
    };
    metadata["config_bindings"] = nlohmann::json::array();
    for (const auto & binding : config_bindings) {
        metadata["config_bindings"].push_back({
            { "key", binding.key },
            { "value", binding.value },
        });
    }
    ggml_backend_hrx2_write_text_file(provider_dir / "provider.json", metadata.dump(2) + "\n");
}

static uint32_t ggml_backend_hrx2_json_u32(const nlohmann::json & object, const char * key, uint32_t default_value = 0) {
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) {
        return default_value;
    }
    return it->get<uint32_t>();
}

static bool ggml_backend_hrx2_route_from_json(
        const nlohmann::json & route_json,
        ggml_backend_hrx2_kernel_route * route) {
    route->id = route_json.value("id", std::string());
    route->family = route_json.value("family", std::string());
    route->op = route_json.value("op", std::string());
    route->target_key = route_json.value("target_key", std::string());
    route->source_id = route_json.value("source_id", std::string());
    route->artifact_id = route_json.value("artifact_id", std::string());
    route->root_symbol = route_json.value("root_symbol", std::string());
    route->export_name = route_json.value("export_name", std::string());
    route->loader_format = route_json.value("loader_format", std::string("amdgpu-hsaco"));
    route->priority = route_json.value("priority", 0);

    const auto & abi = route_json.at("abi");
    route->binding_count = ggml_backend_hrx2_json_u32(abi, "binding_count");
    route->parameter_count = ggml_backend_hrx2_json_u32(abi, "parameter_count");
    route->constant_byte_length = ggml_backend_hrx2_json_u32(abi, "constant_byte_length");

    const auto & dispatch = route_json.at("dispatch");
    const auto & workgroup = dispatch.at("workgroup_size");
    for (int i = 0; i < 3; ++i) {
        route->workgroup_size[i] = workgroup.at(i).get<uint32_t>();
    }
    route->rows_per_workgroup = ggml_backend_hrx2_json_u32(dispatch, "rows_per_workgroup", 1);
    route->cols_per_workgroup = ggml_backend_hrx2_json_u32(dispatch, "cols_per_workgroup", 1);

    const auto & shape_domain = route_json.value("shape_domain", nlohmann::json::object());
    route->ncols_min = ggml_backend_hrx2_json_u32(shape_domain, "ncols_min");
    route->ncols_max = ggml_backend_hrx2_json_u32(shape_domain, "ncols_max");
    route->n_dims_min = ggml_backend_hrx2_json_u32(shape_domain, "n_dims_min");
    route->n_dims_max = ggml_backend_hrx2_json_u32(shape_domain, "n_dims_max");
    route->nrows_min = ggml_backend_hrx2_json_u32(shape_domain, "nrows_min");
    route->nrows_max = ggml_backend_hrx2_json_u32(shape_domain, "nrows_max");
    route->k_min = ggml_backend_hrx2_json_u32(shape_domain, "k_min");
    route->k_max = ggml_backend_hrx2_json_u32(shape_domain, "k_max");
    route->rows_min = ggml_backend_hrx2_json_u32(shape_domain, "rows_min");
    route->rows_max = ggml_backend_hrx2_json_u32(shape_domain, "rows_max");
    route->cols_min = ggml_backend_hrx2_json_u32(shape_domain, "cols_min");
    route->cols_max = ggml_backend_hrx2_json_u32(shape_domain, "cols_max");
    const auto & shape_guards = route_json.value("shape_guards", nlohmann::json::object());
    if (shape_guards.contains("k_pow2")) {
        route->k_pow2_guard = shape_guards.at("k_pow2").get<bool>() ? 1 : -1;
    }
    if (shape_guards.contains("all_pot")) {
        route->all_pot_guard = shape_guards.at("all_pot").get<bool>() ? 1 : -1;
    }
    if (shape_guards.contains("k_multiple_of")) {
        route->k_multiple_of_guard = shape_guards.at("k_multiple_of").get<uint32_t>();
    }
    if (shape_guards.contains("rows_multiple_of")) {
        route->rows_multiple_of_guard = shape_guards.at("rows_multiple_of").get<uint32_t>();
    }
    if (shape_guards.contains("cols_multiple_of")) {
        route->cols_multiple_of_guard = shape_guards.at("cols_multiple_of").get<uint32_t>();
    }
    if (shape_guards.contains("ncols_multiple_of")) {
        route->ncols_multiple_of_guard = shape_guards.at("ncols_multiple_of").get<uint32_t>();
    }
    route->pointwise_src0_row_stride_eq_ncols =
        shape_guards.value("pointwise_src0_row_stride_eq_ncols", false);
    route->pointwise_src1_row_stride_eq_ncols =
        shape_guards.value("pointwise_src1_row_stride_eq_ncols", false);
    route->pointwise_src1_row_stride_eq_zero =
        shape_guards.value("pointwise_src1_row_stride_eq_zero", false);
    route->pointwise_src1_ncols_eq_ncols =
        shape_guards.value("pointwise_src1_ncols_eq_ncols", false);

    const auto & supports = route_json.value("supports", nlohmann::json::object());
    route->supports_mode = supports.value("mode", std::string());
    route->supports_glu_op = supports.value("glu_op", std::string());
    route->supports_layout = supports.value("layout", std::string());

    const auto & specialization = route_json.value("specialization", nlohmann::json::object());
    route->specialization_mode = specialization.value("mode", std::string());
    route->config_bindings.clear();
    for (const auto & binding_json : specialization.value("bindings", nlohmann::json::array())) {
        ggml_backend_hrx2_config_binding_spec binding;
        binding.key = binding_json.value("key", std::string());
        binding.value = binding_json.value("value", std::string());
        binding.value_source = binding_json.value("source", std::string());
        if (binding.key.empty() || (binding.value.empty() && binding.value_source.empty())) {
            GGML_LOG_ERROR("HRX2: route %s has invalid specialization binding\n", route->id.c_str());
            return false;
        }
        route->config_bindings.push_back(std::move(binding));
    }

    return !route->id.empty() &&
           !route->source_id.empty() &&
           !route->root_symbol.empty() &&
           !route->export_name.empty() &&
           route->binding_count > 0;
}

static bool ggml_backend_hrx2_load_catalog_json(
        const nlohmann::json & catalog_json,
        const std::string & catalog_dir,
        ggml_backend_hrx2_catalog * catalog) {
    if (catalog_json.value("schema", std::string()) != "ggml-hrx2-catalog-v1") {
        GGML_LOG_ERROR("HRX2: unsupported catalog schema\n");
        return false;
    }

    catalog->catalog_id = catalog_json.value("catalog_id", std::string());
    catalog->sources.clear();
    catalog->artifacts.clear();
    catalog->routes.clear();

    const ggml_hrx2_embedded_source * embedded_sources = ggml_hrx2_embedded_sources();
    for (size_t i = 0; i < ggml_hrx2_embedded_source_count(); ++i) {
        catalog->sources[embedded_sources[i].id] = {
            /* .id   = */ embedded_sources[i].id,
            /* .path = */ embedded_sources[i].path,
            /* .text = */ std::string(embedded_sources[i].text, embedded_sources[i].text_size),
        };
    }

    const ggml_hrx2_embedded_artifact * embedded_artifacts = ggml_hrx2_embedded_artifacts();
    for (size_t i = 0; i < ggml_hrx2_embedded_artifact_count(); ++i) {
        auto & artifact = catalog->artifacts[embedded_artifacts[i].id];
        artifact.id = embedded_artifacts[i].id;
        artifact.path = embedded_artifacts[i].path;
        artifact.data.assign(embedded_artifacts[i].data, embedded_artifacts[i].data + embedded_artifacts[i].data_size);
    }

    for (const auto & item : catalog_json.at("sources").items()) {
        const std::string source_id = item.key();
        const auto & source_json = item.value();
        auto & source = catalog->sources[source_id];
        source.id = source_id;
        source.path = source_json.value("path", std::string());
        if (!catalog_dir.empty()) {
            const std::string source_text = ggml_backend_hrx2_read_text_file(ggml_backend_hrx2_join_path(catalog_dir, source.path));
            if (source_text.empty()) {
                GGML_LOG_ERROR("HRX2: source %s could not be read from catalog dir\n", source_id.c_str());
                return false;
            }
            source.text = source_text;
        }
    }

    for (const auto & item : catalog_json.at("artifacts").items()) {
        const std::string artifact_id = item.key();
        const auto & artifact_json = item.value();
        auto & artifact = catalog->artifacts[artifact_id];
        artifact.id = artifact_id;
        artifact.path = artifact_json.value("path", std::string());
        artifact.format = artifact_json.value("format", std::string());
        if (!catalog_dir.empty()) {
            const std::vector<unsigned char> artifact_data =
                ggml_backend_hrx2_read_binary_file(ggml_backend_hrx2_join_path(catalog_dir, artifact.path));
            if (!artifact_data.empty()) {
                artifact.data = artifact_data;
            }
        }
    }

    for (const auto & route_json : catalog_json.at("routes")) {
        ggml_backend_hrx2_kernel_route route;
        if (!ggml_backend_hrx2_route_from_json(route_json, &route)) {
            GGML_LOG_ERROR("HRX2: invalid route entry in catalog\n");
            return false;
        }
        if (catalog->sources.find(route.source_id) == catalog->sources.end()) {
            GGML_LOG_ERROR("HRX2: route %s references unknown source %s\n", route.id.c_str(), route.source_id.c_str());
            return false;
        }
        if (!route.artifact_id.empty() && catalog->artifacts.find(route.artifact_id) == catalog->artifacts.end()) {
            GGML_LOG_ERROR("HRX2: route %s references unknown artifact %s\n", route.id.c_str(), route.artifact_id.c_str());
            return false;
        }
        catalog->routes.push_back(std::move(route));
    }
    return true;
}

static bool ggml_backend_hrx2_compile_route(
        const ggml_backend_hrx2_device_info & device,
        const ggml_backend_hrx2_kernel_route & route,
        const std::vector<ggml_backend_hrx2_config_binding> & config_bindings,
        const std::string & cache_key,
        const void * source_data,
        size_t source_size,
        ggml_hrx2_loom_jit_source_format_t source_format,
        const char * source_identifier,
        ggml_backend_hrx2_provider * provider) {
    const char * architecture = device.architecture ? device.architecture : "";
    if (!route.target_key.empty() && route.target_key != architecture) {
        GGML_LOG_ERROR(
            "HRX2: route %s targets %s but device architecture is %s\n",
            route.id.c_str(), route.target_key.c_str(), architecture);
        return false;
    }

    ggml_hrx2_loom_jit_amdgpu_t jit = nullptr;
    ggml_hrx2_loom_jit_amdgpu_options_t jit_options = {
        /* .structure_size = */ sizeof(ggml_hrx2_loom_jit_amdgpu_options_t),
        /* .processor      = */ architecture,
        /* .identifier     = */ "ggml-hrx2",
    };
    if (!GGML_HRX2_CATALOG_CHECK(ggml_hrx2_loom_jit_amdgpu_create(&jit_options, &jit))) {
        return false;
    }

    ggml_hrx2_loom_jit_compile_result_t result = {};
    std::vector<ggml_hrx2_loom_jit_config_binding_t> jit_config_bindings;
    jit_config_bindings.reserve(config_bindings.size());
    for (const auto & binding : config_bindings) {
        jit_config_bindings.push_back({
            /* .key   = */ binding.key.c_str(),
            /* .value = */ binding.value.c_str(),
        });
    }
    ggml_hrx2_loom_jit_compile_options_t compile_options = {
        /* .structure_size        = */ sizeof(ggml_hrx2_loom_jit_compile_options_t),
        /* .source_data           = */ source_data,
        /* .source_size           = */ source_size,
        /* .source_format         = */ source_format,
        /* .source_identifier     = */ source_identifier,
        /* .root_symbol           = */ route.root_symbol.c_str(),
        /* .module_name           = */ "ggml_hrx2",
        /* .artifact_identifier   = */ route.export_name.c_str(),
        /* .config_bindings       = */ jit_config_bindings.empty() ? nullptr : jit_config_bindings.data(),
        /* .config_binding_count  = */ jit_config_bindings.size(),
    };
    const bool compiled = GGML_HRX2_CATALOG_CHECK(ggml_hrx2_loom_jit_amdgpu_compile(jit, &compile_options, &result));
    ggml_hrx2_loom_jit_amdgpu_release(jit);
    if (!compiled) {
        return false;
    }

    hrx_executable_t executable = nullptr;
    const size_t hsaco_size = result.hsaco_size;
    const char * loader_format = route.loader_format.empty() || route.loader_format == "amdgpu-hsaco" ?
        nullptr :
        route.loader_format.c_str();
    const bool loaded = GGML_HRX2_CATALOG_CHECK(hrx_executable_load_data(
        device.device,
        result.hsaco_data,
        hsaco_size,
        loader_format,
        &executable));
    provider->manifest_json = result.manifest_json ?
        std::string(result.manifest_json, result.manifest_json_size) :
        std::string();
    provider->compile_report_json = result.compile_report_json ?
        std::string(result.compile_report_json, result.compile_report_json_size) :
        std::string();
    if (!loaded) {
        ggml_hrx2_loom_jit_compile_result_deinitialize(&result);
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
        ggml_hrx2_loom_jit_compile_result_deinitialize(&result);
        return false;
    }

    provider->executable = executable;
    provider->export_ordinal = export_ordinal;
    provider->export_info = export_info;
    provider->route = route;
    provider->cache_key = cache_key;
    ggml_backend_hrx2_dump_provider_evidence(
        route,
        config_bindings,
        cache_key,
        architecture,
        source_identifier,
        source_format == GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_BYTECODE ? "loom-bytecode" : "loom-text",
        result.hsaco_data,
        hsaco_size,
        *provider);
    ggml_hrx2_loom_jit_compile_result_deinitialize(&result);
    GGML_LOG_INFO(
        "HRX2: JIT compiled route=%s cache_key=%s export=%s target=%s source=%s configs=%zu hsaco=%zu bytes\n",
        route.id.c_str(),
        cache_key.c_str(),
        route.export_name.c_str(),
        architecture,
        source_format == GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_BYTECODE ? "Loom bytecode" : "Loom text",
        config_bindings.size(),
        hsaco_size);
    return true;
}

static bool ggml_backend_hrx2_load_hsaco_route(
        const ggml_backend_hrx2_device_info & device,
        const ggml_backend_hrx2_kernel_route & route,
        const std::vector<ggml_backend_hrx2_config_binding> & config_bindings,
        const std::string & cache_key,
        const void * hsaco_data,
        size_t hsaco_size,
        const char * source_identifier,
        ggml_backend_hrx2_provider * provider) {
    const char * architecture = device.architecture ? device.architecture : "";
    if (!route.target_key.empty() && route.target_key != architecture) {
        GGML_LOG_ERROR(
            "HRX2: route %s targets %s but device architecture is %s\n",
            route.id.c_str(), route.target_key.c_str(), architecture);
        return false;
    }
    if (!hsaco_data || hsaco_size == 0) {
        GGML_LOG_ERROR("HRX2: route %s has empty HSACO artifact\n", route.id.c_str());
        return false;
    }

    hrx_executable_t executable = nullptr;
    const char * loader_format = route.loader_format.empty() || route.loader_format == "amdgpu-hsaco" ?
        nullptr :
        route.loader_format.c_str();
    if (!GGML_HRX2_CATALOG_CHECK(hrx_executable_load_data(
            device.device,
            hsaco_data,
            hsaco_size,
            loader_format,
            &executable))) {
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
    provider->cache_key = cache_key;
    ggml_backend_hrx2_dump_provider_evidence(
        route,
        config_bindings,
        cache_key,
        architecture,
        source_identifier,
        "amdgpu-hsaco",
        hsaco_data,
        hsaco_size,
        *provider);
    GGML_LOG_INFO(
        "HRX2: loaded HSACO route=%s cache_key=%s export=%s target=%s source=%s hsaco=%zu bytes\n",
        route.id.c_str(),
        cache_key.c_str(),
        route.export_name.c_str(),
        architecture,
        source_identifier ? source_identifier : "",
        hsaco_size);
    return true;
}

} // namespace

ggml_backend_hrx2_provider::~ggml_backend_hrx2_provider() {
    if (executable) {
        hrx_executable_release(executable);
    }
}

void ggml_backend_hrx2_catalog_deleter::operator()(ggml_backend_hrx2_catalog * catalog) const {
    delete catalog;
}

ggml_backend_hrx2_catalog_ptr ggml_backend_hrx2_load_catalog() {
    std::string catalog_text = ggml_hrx2_embedded_catalog_json();
    std::string catalog_dir;
    if (const char * env = std::getenv("GGML_HRX2_CATALOG_DIR")) {
        catalog_dir = env;
        const std::string catalog_path = ggml_backend_hrx2_join_path(catalog_dir, "catalog.json");
        const std::string disk_catalog = ggml_backend_hrx2_read_text_file(catalog_path);
        if (disk_catalog.empty()) {
            GGML_LOG_ERROR("HRX2: GGML_HRX2_CATALOG_DIR is set but %s could not be read\n", catalog_path.c_str());
            return nullptr;
        }
        catalog_text = disk_catalog;
    }

    nlohmann::json catalog_json;
    try {
        catalog_json = nlohmann::json::parse(catalog_text);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("HRX2: failed to parse catalog JSON: %s\n", e.what());
        return nullptr;
    }

    ggml_backend_hrx2_catalog_ptr catalog(new ggml_backend_hrx2_catalog());
    if (!ggml_backend_hrx2_load_catalog_json(catalog_json, catalog_dir, catalog.get())) {
        return nullptr;
    }
    return catalog;
}

const ggml_backend_hrx2_kernel_route * ggml_backend_hrx2_catalog_find_route(
        const ggml_backend_hrx2_catalog & catalog,
        const char * route_id) {
    for (const auto & route : catalog.routes) {
        if (route.id == route_id) {
            return &route;
        }
    }
    GGML_LOG_ERROR("HRX2: route %s not found in catalog\n", route_id);
    return nullptr;
}

void ggml_backend_hrx2_catalog_find_routes(
        const ggml_backend_hrx2_catalog & catalog,
        const char * family,
        const char * op,
        std::vector<const ggml_backend_hrx2_kernel_route *> * out_routes) {
    if (!out_routes) {
        return;
    }
    out_routes->clear();
    for (const auto & route : catalog.routes) {
        const bool family_matches = !family || route.family == family;
        const bool op_matches = !op || route.op == op;
        if (family_matches && op_matches) {
            out_routes->push_back(&route);
        }
    }
}

std::unique_ptr<ggml_backend_hrx2_provider> ggml_backend_hrx2_load_provider(
        const ggml_backend_hrx2_device_info & device,
        const ggml_backend_hrx2_catalog & catalog,
        const ggml_backend_hrx2_kernel_route & route,
        const std::vector<ggml_backend_hrx2_config_binding> & config_bindings,
        const std::string & cache_key) {
    auto provider = std::make_unique<ggml_backend_hrx2_provider>();

    if (!route.artifact_id.empty()) {
        const auto artifact_it = catalog.artifacts.find(route.artifact_id);
        if (artifact_it != catalog.artifacts.end() &&
            artifact_it->second.format == "loom-bytecode" &&
            !artifact_it->second.data.empty()) {
            if (ggml_backend_hrx2_compile_route(
                    device,
                    route,
                    config_bindings,
                    cache_key,
                    artifact_it->second.data.data(),
                    artifact_it->second.data.size(),
                    GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_BYTECODE,
                    route.artifact_id.c_str(),
                    provider.get())) {
                return provider;
            }
            return nullptr;
        } else if (artifact_it != catalog.artifacts.end() &&
            artifact_it->second.format == "amdgpu-hsaco" &&
            !artifact_it->second.data.empty()) {
            if (ggml_backend_hrx2_load_hsaco_route(
                    device,
                    route,
                    config_bindings,
                    cache_key,
                    artifact_it->second.data.data(),
                    artifact_it->second.data.size(),
                    route.artifact_id.c_str(),
                    provider.get())) {
                return provider;
            }
            return nullptr;
        }
    }

    const auto source_it = catalog.sources.find(route.source_id);
    if (source_it == catalog.sources.end() || source_it->second.text.empty()) {
        GGML_LOG_ERROR("HRX2: source %s not available\n", route.source_id.c_str());
        return nullptr;
    }

    if (!ggml_backend_hrx2_compile_route(
            device,
            route,
            config_bindings,
            cache_key,
            source_it->second.text.data(),
            source_it->second.text.size(),
            GGML_HRX2_LOOM_JIT_SOURCE_FORMAT_TEXT,
            route.source_id.c_str(),
            provider.get())) {
        return nullptr;
    }
    return provider;
}
