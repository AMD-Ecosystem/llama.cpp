#include "ggml-hrx2.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "hrx_runtime.h"
#include "hrx_loom_jit.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

static constexpr size_t    GGML_HRX2_ALIGNMENT     = 256;
static constexpr uintptr_t GGML_HRX2_FAKE_PTR_BASE = 0x200000000ull;

static const char * k_hrx2_embedded_catalog_json = R"json(
{
  "schema_version": 1,
  "catalog_id": "hrx2-phase0",
  "sources": {
    "rms_norm_f32": {
      "path": "kernels/rms_norm_f32.loom"
    }
  },
  "kernels": [
    {
      "id": "rms_norm_f32_contiguous",
      "op": "RMS_NORM",
      "source_id": "rms_norm_f32",
      "root_symbol": "@hrx2_rms_norm_f32",
      "export_name": "hrx2_rms_norm_f32",
      "artifact_format": "amdgpu-hsaco",
      "abi": {
        "binding_count": 2,
        "parameter_count": 13,
        "constant_byte_length": 44
      },
      "dispatch": {
        "workgroup_size": [512, 1, 1]
      },
      "supports": {
        "src_type": "F32",
        "dst_type": "F32",
        "layout": "contiguous"
      }
    }
  ]
}
)json";

static const char * k_hrx2_rms_norm_f32_loom = R"loom(
kernel.def @hrx2_rms_norm_f32(%ncols: index, %nrows: index) {
  %unit = index.constant 1 : index
  %workgroup_size = index.constant 512 : index
  kernel.launch.config workgroups(%nrows, %unit, %unit) workgroup_size(%workgroup_size, %unit, %unit) : index
} launch(
    %ncols: index,
    %nrows: index,
    %ne1: index,
    %ne2: index,
    %src_nb1: index,
    %src_nb2: index,
    %src_nb3: index,
    %dst_nb1: index,
    %dst_nb2: index,
    %dst_nb3: index,
    %eps: f32,
    %src: buffer,
    %dst: buffer) {
  %base = index.constant 0 : offset
  %workgroup_size = index.constant 512 : index
  %ncols_bounded = index.assume %ncols [range(%ncols, 1, 65536)] : index
  %nrows_bounded = index.assume %nrows [range(%nrows, 1, 1048576)] : index
  %row0 = kernel.workgroup.id<x> : index
  %row = index.assume %row0 [range(%row0, 0, 1048575)] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, 511)] : index
  %zero_f32 = scalar.constant 0.0 : f32

  %src_global = buffer.assume.memory_space<global> %src : buffer
  %dst_global = buffer.assume.memory_space<global> %dst : buffer
  %src_noalias, %dst_noalias = buffer.assume.noalias %src_global, %dst_global : buffer, buffer
  %src_view = buffer.view %src_noalias[%base] : buffer -> view<[%nrows_bounded]x[%ncols_bounded]xf32, #dense>
  %dst_view = buffer.view %dst_noalias[%base] : buffer -> view<[%nrows_bounded]x[%ncols_bounded]xf32, #dense>

  %sum = scf.for %col = [%lane to %ncols_bounded step %workgroup_size](%acc = %zero_f32 : f32) -> (f32) {
    %value = view.load %src_view[%row, %col] : view<[%nrows_bounded]x[%ncols_bounded]xf32, #dense> -> f32
    %square = scalar.mulf<reassoc|nnan|ninf|nsz> %value, %value : f32
    %next = scalar.addf<reassoc|nnan|ninf|nsz> %acc, %square : f32
    scf.yield %next : f32
  }

  %row_sum = kernel.workgroup.reduce<addf> %sum : f32
  %ncols_i32 = index.cast %ncols_bounded : index to i32
  %ncols_f32 = scalar.sitofp %ncols_i32 : i32 to f32
  %mean = scalar.divf<nnan|ninf|nsz|arcp> %row_sum, %ncols_f32 : f32
  %mean_eps = scalar.addf<nnan|ninf|nsz> %mean, %eps : f32
  %scale = scalar.rsqrtf<nnan|ninf|nsz|afn> %mean_eps : f32

  scf.for %col = [%lane to %ncols_bounded step %workgroup_size] {
    %value = view.load %src_view[%row, %col] : view<[%nrows_bounded]x[%ncols_bounded]xf32, #dense> -> f32
    %result = scalar.mulf<nnan|ninf|nsz> %value, %scale : f32
    view.store %result, %dst_view[%row, %col] : f32, view<[%nrows_bounded]x[%ncols_bounded]xf32, #dense>
  }
  kernel.return
}
)loom";

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

    ~ggml_backend_hrx2_provider() {
        if (executable) {
            hrx_executable_release(executable);
        }
    }
};

struct ggml_backend_hrx2_device_context;

struct ggml_backend_hrx2_buffer_type_context {
    ggml_backend_hrx2_device_context * device_context = nullptr;
    std::string name;
};

struct ggml_backend_hrx2_buffer_context {
    ggml_backend_hrx2_device_context * device_context = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * base = nullptr;
};

struct ggml_backend_hrx2_device_context {
    hrx_device_t device = nullptr;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    std::unique_ptr<ggml_backend_hrx2_provider> rms_norm;
    ggml_backend_buffer_type buffer_type = {};
    ggml_backend_hrx2_buffer_type_context buffer_type_context = {};
};

struct ggml_backend_hrx2_context {
    ggml_backend_hrx2_device_context * device_context = nullptr;
    hrx_stream_t stream = nullptr;
    std::string name;
};

struct ggml_backend_hrx2_reg_context {
    bool gpu_initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx2_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx2_reg_context() {
        for (auto & device_context : device_contexts) {
            if (device_context->device) {
                hrx_device_release(device_context->device);
            }
        }
        if (gpu_initialized) {
            hrx_status_ignore(hrx_gpu_shutdown());
        }
    }
};

struct ggml_backend_hrx2_rms_norm_constants {
    uint32_t ncols;
    uint32_t nrows;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t src_nb1;
    uint32_t src_nb2;
    uint32_t src_nb3;
    uint32_t dst_nb1;
    uint32_t dst_nb2;
    uint32_t dst_nb3;
    float eps;
};

static_assert(sizeof(ggml_backend_hrx2_rms_norm_constants) == 44);

static bool ggml_hrx2_check(hrx_status_t status, const char * expression, const char * file, int line) {
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

#define GGML_HRX2_CHECK(expr) ggml_hrx2_check((expr), #expr, __FILE__, __LINE__)

static ggml_backend_hrx2_device_context * ggml_backend_hrx2_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_hrx2_device_context *>(dev->context);
}

static ggml_backend_hrx2_context * ggml_backend_hrx2_get_context(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx2_context *>(backend->context);
}

static ggml_backend_hrx2_buffer_context * ggml_backend_hrx2_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx2_buffer_context *>(buffer->context);
}

static ggml_backend_hrx2_buffer_type_context * ggml_backend_hrx2_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx2_buffer_type_context *>(buft->context);
}

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
    std::string catalog_text = k_hrx2_embedded_catalog_json;
    std::string catalog_dir;
    if (const char * env = std::getenv("GGML_HRX2_CATALOG_DIR")) {
        catalog_dir = env;
        const std::string disk_catalog = ggml_backend_hrx2_read_text_file(ggml_backend_hrx2_join_path(catalog_dir, "catalog.json"));
        if (disk_catalog.empty()) {
            GGML_LOG_ERROR("HRX2: GGML_HRX2_CATALOG_DIR is set but catalog.json could not be read\n");
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
    (*out_sources)["rms_norm_f32"] = k_hrx2_rms_norm_f32_loom;

    if (!catalog_dir.empty() && out_catalog->contains("sources")) {
        for (const auto & item : (*out_catalog)["sources"].items()) {
            const std::string source_id = item.key();
            const auto & source = item.value();
            if (source.contains("text")) {
                (*out_sources)[source_id] = source["text"].get<std::string>();
            } else if (source.contains("path")) {
                const std::string source_text = ggml_backend_hrx2_read_text_file(
                    ggml_backend_hrx2_join_path(catalog_dir, source["path"].get<std::string>()));
                if (source_text.empty()) {
                    GGML_LOG_ERROR("HRX2: source %s could not be read from dev catalog\n", source_id.c_str());
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

static size_t ggml_backend_hrx2_total_memory(hrx_device_t device) {
    uint64_t memory_total = 0;
    if (!GGML_HRX2_CHECK(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory_total, sizeof(memory_total)))) {
        return 0;
    }
    return static_cast<size_t>(memory_total);
}

static std::string ggml_backend_hrx2_device_string_property(hrx_device_t device, hrx_device_property_t property) {
    std::array<char, 128> value = {};
    if (!GGML_HRX2_CHECK(hrx_device_get_property(device, property, value.data(), value.size()))) {
        return {};
    }
    return std::string(value.data());
}

static std::string ggml_backend_hrx2_device_description(hrx_device_t device) {
    std::string name = ggml_backend_hrx2_device_string_property(device, HRX_DEVICE_PROPERTY_NAME);
    std::string arch = ggml_backend_hrx2_device_string_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);
    if (name.empty()) {
        name = "HRX GPU";
    }
    if (!arch.empty()) {
        name += " (";
        name += arch;
        name += ")";
    }
    return name;
}

static bool ggml_backend_hrx2_compile_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route & route,
        const std::string & source,
        ggml_backend_hrx2_provider * provider) {
    hrx_loom_jit_amdgpu_t jit = nullptr;
    hrx_loom_jit_amdgpu_options_t jit_options = {
        /* .structure_size = */ sizeof(hrx_loom_jit_amdgpu_options_t),
        /* .processor      = */ device_context->architecture.c_str(),
        /* .identifier     = */ "ggml-hrx2",
    };
    if (!GGML_HRX2_CHECK(hrx_loom_jit_amdgpu_create(&jit_options, &jit))) {
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
    const bool compiled = GGML_HRX2_CHECK(hrx_loom_jit_amdgpu_compile(jit, &compile_options, &result));
    hrx_loom_jit_amdgpu_release(jit);
    if (!compiled) {
        return false;
    }

    hrx_executable_t executable = nullptr;
    const size_t hsaco_size = result.hsaco_size;
    const char * loader_format = route.artifact_format.empty() || route.artifact_format == "amdgpu-hsaco" ?
        nullptr :
        route.artifact_format.c_str();
    const bool loaded = GGML_HRX2_CHECK(hrx_executable_load_data(
        device_context->device,
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
        GGML_HRX2_CHECK(hrx_executable_lookup_export_by_name(executable, route.export_name.c_str(), &export_ordinal)) &&
        GGML_HRX2_CHECK(hrx_executable_export_info(executable, export_ordinal, &export_info)) &&
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
        "HRX2: JIT compiled %s for %s (%zu bytes HSACO, sgpr/vgpr available in compile report)\n",
        route.export_name.c_str(), device_context->architecture.c_str(), hsaco_size);
    return true;
}

static std::unique_ptr<ggml_backend_hrx2_provider> ggml_backend_hrx2_load_rms_norm_provider(
        ggml_backend_hrx2_device_context * device_context) {
    nlohmann::json catalog;
    std::unordered_map<std::string, std::string> sources;
    if (!ggml_backend_hrx2_load_catalog(&catalog, &sources)) {
        return nullptr;
    }

    ggml_backend_hrx2_kernel_route route;
    if (!ggml_backend_hrx2_route_from_catalog(catalog, "rms_norm_f32_contiguous", &route)) {
        return nullptr;
    }

    auto source = sources.find(route.source_id);
    if (source == sources.end() || source->second.empty()) {
        GGML_LOG_ERROR("HRX2: source %s not available\n", route.source_id.c_str());
        return nullptr;
    }

    auto provider = std::make_unique<ggml_backend_hrx2_provider>();
    if (!ggml_backend_hrx2_compile_route(device_context, route, source->second, provider.get())) {
        return nullptr;
    }
    return provider;
}

static void * ggml_backend_hrx2_buffer_get_base(ggml_backend_buffer_t buffer);

static size_t ggml_backend_hrx2_tensor_offset(const ggml_backend_hrx2_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static bool ggml_backend_hrx2_tensor_buffer_ref(const ggml_tensor * tensor, hrx_buffer_ref_t * out_ref) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx2_buffer_get_base) {
        return false;
    }

    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    const size_t offset = ggml_backend_hrx2_tensor_offset(context, tensor);
    const size_t length = ggml_nbytes(tensor);
    if (!context->buffer || offset > buffer->size || length > buffer->size - offset) {
        return false;
    }

    *out_ref = {
        /* .buffer = */ context->buffer,
        /* .offset = */ offset,
        /* .length = */ length,
    };
    return true;
}

static const char * ggml_backend_hrx2_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_hrx2_get_buft_context(buft)->name.c_str();
}

static void ggml_backend_hrx2_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    if (context->buffer) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * ggml_backend_hrx2_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx2_get_buffer_context(buffer)->base;
}

static void ggml_backend_hrx2_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    const size_t buffer_offset = ggml_backend_hrx2_tensor_offset(context, tensor) + offset;
    if (size != 0) {
        (void) GGML_HRX2_CHECK(hrx_queue_fill(
            context->device_context->device, 0, nullptr, nullptr, context->buffer, buffer_offset, size, &value, sizeof(value)));
    }
}

static void ggml_backend_hrx2_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    const size_t buffer_offset = ggml_backend_hrx2_tensor_offset(context, tensor) + offset;
    if (size != 0) {
        (void) GGML_HRX2_CHECK(hrx_synchronous_h2d(context->device_context->device, data, context->buffer, buffer_offset, size));
    }
}

static void ggml_backend_hrx2_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    const size_t buffer_offset = ggml_backend_hrx2_tensor_offset(context, tensor) + offset;
    if (size != 0) {
        (void) GGML_HRX2_CHECK(hrx_synchronous_d2h(context->device_context->device, context->buffer, buffer_offset, data, size));
    }
}

static void ggml_backend_hrx2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    if (buffer->size != 0) {
        (void) GGML_HRX2_CHECK(hrx_queue_fill(
            context->device_context->device, 0, nullptr, nullptr, context->buffer, 0, buffer->size, &value, sizeof(value)));
    }
}

static const ggml_backend_buffer_i ggml_backend_hrx2_buffer_i = {
    /* .free_buffer   = */ ggml_backend_hrx2_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx2_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx2_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx2_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx2_buffer_get_tensor,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ ggml_backend_hrx2_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_hrx2_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_hrx2_get_buft_context(buft);

    hrx_buffer_params_t params = {
        /* .type           = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
        /* .access         = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT,
        /* .queue_affinity = */ 0,
    };
    hrx_buffer_t hrx_buffer = nullptr;
    if (size != 0 && !GGML_HRX2_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(buft_context->device_context->device), params, size, &hrx_buffer))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx2_buffer_context {
        /* .device_context = */ buft_context->device_context,
        /* .buffer         = */ hrx_buffer,
        /* .base           = */ reinterpret_cast<uint8_t *>(GGML_HRX2_FAKE_PTR_BASE),
    };
    if (!context) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_hrx2_buffer_i, context, size);
    if (!buffer) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        delete context;
    }
    return buffer;
}

static size_t ggml_backend_hrx2_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX2_ALIGNMENT;
}

static size_t ggml_backend_hrx2_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * context = ggml_backend_hrx2_get_buft_context(buft)->device_context;
    return context->memory_total ? context->memory_total : std::numeric_limits<size_t>::max();
}

static const ggml_backend_buffer_type_i ggml_backend_hrx2_buffer_type_i = {
    /* .get_name       = */ ggml_backend_hrx2_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hrx2_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hrx2_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hrx2_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_hrx2_device_buffer_type(ggml_backend_dev_t dev) {
    return &ggml_backend_hrx2_get_device_context(dev)->buffer_type;
}

static bool ggml_backend_hrx2_supports_rms_norm(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const bool known_non_overlapping =
        !src0 || !src0->data || !op->data ||
        static_cast<const uint8_t *>(src0->data) + ggml_nbytes(src0) <= static_cast<const uint8_t *>(op->data) ||
        static_cast<const uint8_t *>(op->data) + ggml_nbytes(op) <= static_cast<const uint8_t *>(src0->data);
    return device_context->rms_norm &&
           op->op == GGML_OP_RMS_NORM &&
           src0 &&
           op->view_src == nullptr &&
           op->type == GGML_TYPE_F32 &&
           src0->type == GGML_TYPE_F32 &&
           src0->ne[0] > 0 &&
           src0->ne[0] <= 65536 &&
           ggml_nrows(src0) > 0 &&
           ggml_nrows(src0) <= 1048576 &&
           ggml_are_same_shape(src0, op) &&
           known_non_overlapping &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static ggml_status ggml_backend_hrx2_dispatch_rms_norm(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: RMS_NORM tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(eps));
    ggml_backend_hrx2_rms_norm_constants constants = {
        /* .ncols   = */ static_cast<uint32_t>(src0->ne[0]),
        /* .nrows   = */ static_cast<uint32_t>(ggml_nrows(src0)),
        /* .ne1     = */ static_cast<uint32_t>(src0->ne[1]),
        /* .ne2     = */ static_cast<uint32_t>(src0->ne[2]),
        /* .src_nb1 = */ static_cast<uint32_t>(src0->nb[1]),
        /* .src_nb2 = */ static_cast<uint32_t>(src0->nb[2]),
        /* .src_nb3 = */ static_cast<uint32_t>(src0->nb[3]),
        /* .dst_nb1 = */ static_cast<uint32_t>(dst->nb[1]),
        /* .dst_nb2 = */ static_cast<uint32_t>(dst->nb[2]),
        /* .dst_nb3 = */ static_cast<uint32_t>(dst->nb[3]),
        /* .eps     = */ eps,
    };

    const auto & provider = *context->device_context->rms_norm;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { constants.nrows, 1, 1 },
        /* .workgroup_size  = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : provider.route.workgroup_size[0],
            1,
            1,
        },
        /* .subgroup_size   = */ 0,
    };

    if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static const char * ggml_backend_hrx2_get_name(ggml_backend_t backend) {
    return ggml_backend_hrx2_get_context(backend)->name.c_str();
}

static void ggml_backend_hrx2_free(ggml_backend_t backend) {
    auto * context = ggml_backend_hrx2_get_context(backend);
    if (context->stream) {
        hrx_stream_release(context->stream);
    }
    delete context;
    delete backend;
}

static void ggml_backend_hrx2_synchronize(ggml_backend_t backend) {
    auto * context = ggml_backend_hrx2_get_context(backend);
    if (context->stream) {
        (void) GGML_HRX2_CHECK(hrx_stream_synchronize(context->stream));
    }
}

static enum ggml_status ggml_backend_hrx2_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = ggml_backend_hrx2_get_context(backend);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_RMS_NORM:
                if (!ggml_backend_hrx2_supports_rms_norm(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported RMS_NORM shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_rms_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            default:
                GGML_LOG_ERROR("HRX2: unsupported op %s\n", ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }
    if (!GGML_HRX2_CHECK(hrx_stream_flush(context->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_hrx2_i = {
    /* .get_name           = */ ggml_backend_hrx2_get_name,
    /* .free               = */ ggml_backend_hrx2_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ ggml_backend_hrx2_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_hrx2_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static ggml_guid_t ggml_backend_hrx2_guid(void) {
    static ggml_guid guid = { 0x82, 0x48, 0x52, 0x58, 0x32, 0x2d, 0x4c, 0x4f, 0x4f, 0x4d, 0x2d, 0x4a, 0x49, 0x54, 0x00, 0x01 };
    return &guid;
}

static const char * ggml_backend_hrx2_device_get_name(ggml_backend_dev_t dev) {
    return ggml_backend_hrx2_get_device_context(dev)->name.c_str();
}

static const char * ggml_backend_hrx2_device_get_description(ggml_backend_dev_t dev) {
    return ggml_backend_hrx2_get_device_context(dev)->description.c_str();
}

static void ggml_backend_hrx2_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = ggml_backend_hrx2_get_device_context(dev);
    *free = context->memory_total;
    *total = context->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx2_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx2_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_hrx2_device_get_name(dev);
    props->description = ggml_backend_hrx2_device_get_description(dev);
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    ggml_backend_hrx2_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_hrx2_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * device_context = ggml_backend_hrx2_get_device_context(dev);
    hrx_stream_t stream = nullptr;
    if (!GGML_HRX2_CHECK(hrx_stream_create(device_context->device, 0, &stream))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx2_context {
        /* .device_context = */ device_context,
        /* .stream         = */ stream,
        /* .name           = */ device_context->name,
    };
    if (!context) {
        hrx_stream_release(stream);
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_hrx2_guid(),
        /* .iface   = */ ggml_backend_hrx2_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    if (!backend) {
        hrx_stream_release(stream);
        delete context;
        return nullptr;
    }
    return backend;
}

static bool ggml_backend_hrx2_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_RMS_NORM:
            return ggml_backend_hrx2_supports_rms_norm(ggml_backend_hrx2_get_device_context(dev), op);
        default:
            return false;
    }
}

static bool ggml_backend_hrx2_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_hrx2_buffer_type_get_name) {
        return false;
    }
    return ggml_backend_hrx2_get_buft_context(buft)->device_context == ggml_backend_hrx2_get_device_context(dev);
}

static const ggml_backend_device_i ggml_backend_hrx2_device_i = {
    /* .get_name             = */ ggml_backend_hrx2_device_get_name,
    /* .get_description      = */ ggml_backend_hrx2_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx2_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx2_device_get_type,
    /* .get_props            = */ ggml_backend_hrx2_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx2_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx2_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx2_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_backend_hrx2_reg_context * ggml_backend_hrx2_get_reg_context(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_hrx2_reg_context *>(reg->context);
}

static const char * ggml_backend_hrx2_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX2_NAME;
}

static size_t ggml_backend_hrx2_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_hrx2_get_reg_context(reg)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * context = ggml_backend_hrx2_get_reg_context(reg);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * ggml_backend_hrx2_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hrx2_reg_i = {
    /* .get_name         = */ ggml_backend_hrx2_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx2_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx2_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hrx2_reg_get_proc_address,
};

static std::unique_ptr<ggml_backend_hrx2_reg_context> ggml_backend_hrx2_create_reg_context() {
    auto context = std::make_unique<ggml_backend_hrx2_reg_context>();

    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->gpu_initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        GGML_HRX2_CHECK(status);
        return context;
    }

    int device_count = 0;
    if (!GGML_HRX2_CHECK(hrx_gpu_device_count(&device_count)) || device_count <= 0) {
        return context;
    }

    context->device_contexts.reserve(device_count);
    context->devices.reserve(device_count);
    for (int i = 0; i < device_count; ++i) {
        hrx_device_t device = nullptr;
        if (!GGML_HRX2_CHECK(hrx_gpu_device_get(i, &device)) || !device) {
            continue;
        }
        hrx_device_retain(device);

        auto device_context = std::make_unique<ggml_backend_hrx2_device_context>();
        device_context->device = device;
        device_context->name = std::string(GGML_HRX2_NAME) + std::to_string(i);
        device_context->description = ggml_backend_hrx2_device_description(device);
        device_context->architecture = ggml_backend_hrx2_device_string_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);
        device_context->memory_total = ggml_backend_hrx2_total_memory(device);
        device_context->rms_norm = ggml_backend_hrx2_load_rms_norm_provider(device_context.get());
        device_context->buffer_type_context = {
            /* .device_context = */ device_context.get(),
            /* .name           = */ device_context->name,
        };
        device_context->buffer_type = {
            /* .iface   = */ ggml_backend_hrx2_buffer_type_i,
            /* .device  = */ nullptr,
            /* .context = */ &device_context->buffer_type_context,
        };

        context->device_contexts.emplace_back(std::move(device_context));
        context->devices.push_back({
            /* .iface   = */ ggml_backend_hrx2_device_i,
            /* .reg     = */ nullptr,
            /* .context = */ context->device_contexts.back().get(),
        });
        context->device_contexts.back()->buffer_type.device = &context->devices.back();
    }

    return context;
}

} // namespace

ggml_backend_t ggml_backend_hrx2_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid HRX2 device index %zu\n", __func__, dev_num);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, dev_num), nullptr);
}

bool ggml_backend_is_hrx2(ggml_backend_t backend) {
    return backend != nullptr && backend->guid == ggml_backend_hrx2_guid();
}

int ggml_backend_hrx2_get_device_count(void) {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_hrx2_reg()));
}

void ggml_backend_hrx2_get_device_description(int device, char * description, size_t description_size) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        if (description_size) {
            description[0] = '\0';
        }
        return;
    }
    const char * desc = ggml_backend_dev_description(ggml_backend_reg_dev_get(reg, device));
    std::snprintf(description, description_size, "%s", desc ? desc : "");
}

void ggml_backend_hrx2_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        *free = 0;
        *total = 0;
        return;
    }
    ggml_backend_dev_memory(ggml_backend_reg_dev_get(reg, device), free, total);
}

ggml_backend_buffer_type_t ggml_backend_hrx2_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_reg_t ggml_backend_hrx2_reg(void) {
    static std::unique_ptr<ggml_backend_hrx2_reg_context> context = ggml_backend_hrx2_create_reg_context();
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_hrx2_reg_i,
        /* .context     = */ context.get(),
    };
    if (context) {
        for (auto & device : context->devices) {
            device.reg = &reg;
        }
    }
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx2_reg)
