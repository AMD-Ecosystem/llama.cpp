#include "ggml-hrx2.h"

#include "ggml-hrx2-catalog.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "hrx_runtime.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

static constexpr size_t    GGML_HRX2_ALIGNMENT     = 256;
static constexpr uintptr_t GGML_HRX2_FAKE_PTR_BASE = 0x200000000ull;

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
    ggml_backend_hrx2_catalog_ptr catalog;
    std::vector<const ggml_backend_hrx2_kernel_route *> rms_norm_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q8_0_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_f32_f32_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q4_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_id_q4_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q5_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_id_q5_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q6_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_id_q6_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_f16_f32_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> cont_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> swiglu_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> set_rows_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> add_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> div_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> scale_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> clamp_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> sum_rows_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> get_rows_moe_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> argsort_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> rope_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> soft_max_routes;
    std::unordered_map<std::string, std::unique_ptr<ggml_backend_hrx2_provider>> providers;
    std::unordered_set<std::string> provider_failures;
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
            device_context->providers.clear();
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

struct ggml_backend_hrx2_mul_mat_constants {
    uint32_t k;
    uint32_t rows;
    uint32_t cols;
};

static_assert(sizeof(ggml_backend_hrx2_mul_mat_constants) == 12);

struct ggml_backend_hrx2_rope_constants {
    float freq_base;
    float freq_scale;
    float attn_factor;
};

static_assert(sizeof(ggml_backend_hrx2_rope_constants) == 12);

struct ggml_backend_hrx2_soft_max_constants {
    float scale;
};

static_assert(sizeof(ggml_backend_hrx2_soft_max_constants) == 4);

struct ggml_backend_hrx2_mul_mat_shape {
    uint32_t k = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct ggml_backend_hrx2_mul_mat_id_shape {
    uint32_t k = 0;
    uint32_t rows = 0;
    uint32_t nexperts = 0;
    uint32_t nselected = 0;
    uint32_t ntokens = 0;
    uint32_t src1_selected_stride = 0;
    uint32_t src1_token_stride = 0;
    uint32_t idx_token_stride = 0;
    uint32_t dst_token_stride = 0;
};

struct ggml_backend_hrx2_mul_mat_f16_shape {
    uint32_t k = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t dst_ne2 = 0;
    uint32_t dst_ne3 = 0;
    uint32_t src0_ne2 = 0;
    uint32_t src0_ne3 = 0;
    uint32_t src0_stride_row = 0;
    uint32_t src0_stride_ne2 = 0;
    uint32_t src0_stride_ne3 = 0;
    uint32_t src1_stride_col = 0;
    uint32_t src1_stride_ne2 = 0;
    uint32_t src1_stride_ne3 = 0;
    uint32_t dst_stride_col = 0;
    uint32_t dst_stride_ne2 = 0;
    uint32_t dst_stride_ne3 = 0;
};

struct ggml_backend_hrx2_rms_norm_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
};

struct ggml_backend_hrx2_set_rows_shape {
    uint32_t nc = 0;
    uint32_t nr = 0;
    uint32_t ne02 = 0;
    uint32_t ne03 = 0;
    uint32_t ne1 = 0;
    uint32_t ne11 = 0;
    uint32_t ne12 = 0;
    uint32_t src0_nb1 = 0;
    uint32_t src0_nb2 = 0;
    uint32_t src0_nb3 = 0;
    uint32_t idx_nb0 = 0;
    uint32_t idx_nb1 = 0;
    uint32_t idx_nb2 = 0;
    uint32_t dst_nb1 = 0;
    uint32_t dst_nb2 = 0;
    uint32_t dst_nb3 = 0;
};

struct ggml_backend_hrx2_pointwise_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t src0_row_stride = 0;
    uint32_t src1_row_stride = 0;
    uint32_t src1_ncols = 0;
};

struct ggml_backend_hrx2_sum_rows_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t src0_row_stride = 0;
};

struct ggml_backend_hrx2_get_rows_moe_shape {
    uint32_t nexperts = 0;
    uint32_t nselected = 0;
    uint32_t ntokens = 0;
    uint32_t src0_token_stride = 0;
    uint32_t idx_token_stride = 0;
    uint32_t dst_token_stride = 0;
};

struct ggml_backend_hrx2_argsort_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
};

struct ggml_backend_hrx2_rope_shape {
    uint32_t ncols = 0;
    uint32_t n_dims = 0;
    uint32_t mode = 0;
    uint32_t nheads = 0;
    uint32_t ntokens = 0;
    uint32_t nrows = 0;
    uint32_t src0_head_stride = 0;
    uint32_t src0_token_stride = 0;
    uint32_t dst_head_stride = 0;
    uint32_t dst_token_stride = 0;
    uint32_t pos_token_stride = 0;
};

struct ggml_backend_hrx2_soft_max_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t ne01 = 0;
    uint32_t ne02 = 0;
    uint32_t mask_nb1 = 1;
    uint32_t mask_nb2 = 1;
    uint32_t mask_nb3 = 1;
    uint32_t mask_ne1 = 1;
    uint32_t mask_ne2 = 1;
    uint32_t mask_ne3 = 1;
    bool has_mask = false;
};

struct ggml_backend_hrx2_cont_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t ne1 = 0;
    uint32_t ne2 = 0;
    uint32_t src_nb1 = 0;
    uint32_t src_nb2 = 0;
    uint32_t src_nb3 = 0;
};

struct ggml_backend_hrx2_swiglu_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    enum ggml_glu_op glu_op = GGML_GLU_OP_SWIGLU;
    bool split_sources = false;
};

struct ggml_backend_hrx2_provider_plan {
    const ggml_backend_hrx2_kernel_route * route = nullptr;
    std::string cache_key;
    std::vector<ggml_backend_hrx2_config_binding> config_bindings;
};

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

static bool ggml_backend_hrx2_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static const char * ggml_backend_hrx2_glu_op_key(enum ggml_glu_op glu_op) {
    switch (glu_op) {
        case GGML_GLU_OP_REGLU: return "REGLU";
        case GGML_GLU_OP_GEGLU: return "GEGLU";
        case GGML_GLU_OP_SWIGLU: return "SWIGLU";
        case GGML_GLU_OP_SWIGLU_OAI: return "SWIGLU_OAI";
        case GGML_GLU_OP_GEGLU_ERF: return "GEGLU_ERF";
        case GGML_GLU_OP_GEGLU_QUICK: return "GEGLU_QUICK";
        case GGML_GLU_OP_COUNT: break;
    }
    return "UNKNOWN";
}

static std::string ggml_backend_hrx2_json_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"':  escaped += "\\\""; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[7] = {};
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    escaped += buffer;
                } else {
                    escaped += static_cast<char>(c);
                }
                break;
        }
    }
    return escaped;
}

static std::string ggml_backend_hrx2_json_kv(const char * key, const std::string & value) {
    std::string result = "\"";
    result += key;
    result += "\":\"";
    result += ggml_backend_hrx2_json_escape(value);
    result += "\"";
    return result;
}

static std::string ggml_backend_hrx2_json_kv(const char * key, uint64_t value) {
    std::string result = "\"";
    result += key;
    result += "\":";
    result += std::to_string(value);
    return result;
}

static void ggml_backend_hrx2_trace_event(const char * event, const std::string & fields_json) {
    const char * trace_path = std::getenv("GGML_HRX2_TRACE_JSONL");
    const bool trace_log = ggml_backend_hrx2_env_enabled("GGML_HRX2_TRACE_ROUTES");
    if ((!trace_path || trace_path[0] == '\0') && !trace_log) {
        return;
    }

    std::string line = "{\"event\":\"";
    line += event ? event : "";
    line += "\"";
    if (!fields_json.empty()) {
        line += ",";
        line += fields_json;
    }
    line += "}";

    if (trace_path && trace_path[0] != '\0') {
        std::ofstream output(trace_path, std::ios::out | std::ios::app);
        if (output) {
            output << line << "\n";
        }
    }
    if (trace_log) {
        GGML_LOG_INFO("HRX2_TRACE: %s\n", line.c_str());
    }
}

static ggml_backend_hrx2_device_context * ggml_backend_hrx2_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_hrx2_device_context *>(dev->context);
}

static ggml_backend_hrx2_context * ggml_backend_hrx2_get_context(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx2_context *>(backend->context);
}

static bool ggml_backend_hrx2_route_available(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route) {
    return route && (route->target_key.empty() || route->target_key == device_context->architecture);
}

static std::string ggml_backend_hrx2_base_cache_key(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route) {
    std::string cache_key = route ? route->id : std::string();
    cache_key += "|target=";
    cache_key += device_context ? device_context->architecture : std::string();
    return cache_key;
}

static ggml_backend_hrx2_provider * ggml_backend_hrx2_get_provider(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const std::vector<ggml_backend_hrx2_config_binding> & config_bindings,
        const std::string & cache_key) {
    if (!ggml_backend_hrx2_route_available(device_context, route) || !device_context->catalog) {
        return nullptr;
    }

    auto existing = device_context->providers.find(cache_key);
    if (existing != device_context->providers.end()) {
        ggml_backend_hrx2_trace_event(
            "provider_cache",
            ggml_backend_hrx2_json_kv("status", "hit") + "," +
            ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
            ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", cache_key));
        return existing->second.get();
    }
    if (device_context->provider_failures.find(cache_key) != device_context->provider_failures.end()) {
        ggml_backend_hrx2_trace_event(
            "provider_cache",
            ggml_backend_hrx2_json_kv("status", "failed_memo") + "," +
            ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
            ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", cache_key));
        return nullptr;
    }

    ggml_backend_hrx2_trace_event(
        "provider_cache",
        ggml_backend_hrx2_json_kv("status", "miss") + "," +
        ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
        ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
        ggml_backend_hrx2_json_kv("cache_key", cache_key));

    const ggml_backend_hrx2_device_info jit_device = {
        /* .device       = */ device_context->device,
        /* .architecture = */ device_context->architecture.c_str(),
    };
    auto provider = ggml_backend_hrx2_load_provider(jit_device, *device_context->catalog, *route, config_bindings, cache_key);
    if (!provider) {
        ggml_backend_hrx2_trace_event(
            "provider_compile",
            ggml_backend_hrx2_json_kv("status", "failed") + "," +
            ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
            ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", cache_key));
        device_context->provider_failures.insert(cache_key);
        return nullptr;
    }
    ggml_backend_hrx2_provider * provider_ptr = provider.get();
    ggml_backend_hrx2_trace_event(
        "provider_compile",
        ggml_backend_hrx2_json_kv("status", "success") + "," +
        ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
        ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
        ggml_backend_hrx2_json_kv("cache_key", cache_key) + "," +
        ggml_backend_hrx2_json_kv("compile_report_bytes", static_cast<uint64_t>(provider_ptr->compile_report_json.size())) + "," +
        ggml_backend_hrx2_json_kv("manifest_bytes", static_cast<uint64_t>(provider_ptr->manifest_json.size())));
    device_context->providers.emplace(cache_key, std::move(provider));
    return provider_ptr;
}

static ggml_backend_hrx2_buffer_context * ggml_backend_hrx2_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx2_buffer_context *>(buffer->context);
}

static ggml_backend_hrx2_buffer_type_context * ggml_backend_hrx2_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx2_buffer_type_context *>(buft->context);
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
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_RMS_NORM &&
           src0 &&
           op->view_src == nullptr &&
           op->type == GGML_TYPE_F32 &&
           src0->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_q8_0(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q8_0 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_f32_f32(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k > 0 && rows > 0 && cols > 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_q4_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q4_K ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_id_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op,
        ggml_type src0_type) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    if (op->op != GGML_OP_MUL_MAT_ID ||
        !src0 ||
        !src1 ||
        !src2 ||
        op->view_src != nullptr ||
        src0->type != src0_type ||
        src1->type != GGML_TYPE_F32 ||
        src2->type != GGML_TYPE_I32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t nexperts = src0->ne[2];
    const int64_t nselected = src2->ne[0];
    const int64_t ntokens = op->ne[2];
    const int64_t block_size = ggml_blck_size(src0->type);
    return k > 0 &&
           rows > 0 &&
           nexperts > 0 &&
           nselected > 0 &&
           ntokens > 0 &&
           src0->ne[0] == src1->ne[0] &&
           src0->ne[3] == 1 &&
           (src1->ne[1] == 1 || src1->ne[1] == nselected) &&
           src1->ne[2] == ntokens &&
           src1->ne[3] == 1 &&
           src2->ne[1] == ntokens &&
           src2->ne[2] == 1 &&
           src2->ne[3] == 1 &&
           op->ne[0] == rows &&
           op->ne[1] == nselected &&
           op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           src1->nb[0] == sizeof(float) &&
           src1->nb[1] % sizeof(float) == 0 &&
           src1->nb[2] % sizeof(float) == 0 &&
           src2->nb[0] == sizeof(int32_t) &&
           src2->nb[1] % sizeof(int32_t) == 0 &&
           op->nb[0] == sizeof(float) &&
           op->nb[1] == static_cast<size_t>(rows) * sizeof(float) &&
           op->nb[2] % sizeof(float) == 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           nexperts <= std::numeric_limits<uint32_t>::max() &&
           nselected <= std::numeric_limits<uint32_t>::max() &&
           ntokens <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[1] / sizeof(float) <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[2] / sizeof(float) <= std::numeric_limits<uint32_t>::max() &&
           src2->nb[1] / sizeof(int32_t) <= std::numeric_limits<uint32_t>::max() &&
           op->nb[2] / sizeof(float) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q4_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k(device_context, op, GGML_TYPE_Q4_K);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q5_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k(device_context, op, GGML_TYPE_Q5_K);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q6_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k(device_context, op, GGML_TYPE_Q6_K);
}

static bool ggml_backend_hrx2_supports_mul_mat_q6_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q6_K ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_q5_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q5_K ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_f16_f32_batched(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F16 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        src0->nb[0] != ggml_type_size(GGML_TYPE_F16) ||
        src1->nb[0] != sizeof(float) ||
        op->nb[0] != sizeof(float)) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    if (k <= 0 ||
        rows <= 0 ||
        cols <= 0 ||
        src0->ne[0] != src1->ne[0] ||
        op->ne[0] != src0->ne[1] ||
        op->ne[1] != src1->ne[1] ||
        op->ne[2] != src1->ne[2] ||
        op->ne[3] != src1->ne[3] ||
        src0->ne[2] <= 0 ||
        src0->ne[3] <= 0 ||
        op->ne[2] <= 0 ||
        op->ne[3] <= 0 ||
        op->ne[2] % src0->ne[2] != 0 ||
        op->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    return k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max() &&
           op->ne[2] <= std::numeric_limits<uint32_t>::max() &&
           op->ne[3] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[2] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[3] <= std::numeric_limits<uint32_t>::max() &&
           src0->nb[1] <= std::numeric_limits<uint32_t>::max() &&
           src0->nb[2] <= std::numeric_limits<uint32_t>::max() &&
           src0->nb[3] <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[1] <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[2] <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[3] <= std::numeric_limits<uint32_t>::max() &&
           op->nb[1] <= std::numeric_limits<uint32_t>::max() &&
           op->nb[2] <= std::numeric_limits<uint32_t>::max() &&
           op->nb[3] <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_u32(int64_t value, uint32_t * out_value) {
    if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out_value = static_cast<uint32_t>(value);
    return true;
}

static bool ggml_backend_hrx2_u32_size(size_t value, uint32_t * out_value) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out_value = static_cast<uint32_t>(value);
    return true;
}

static bool ggml_backend_hrx2_flat_row_stride_f32(
        const ggml_tensor * tensor,
        uint32_t * out_row_stride) {
    if (!tensor ||
        tensor->type != GGML_TYPE_F32 ||
        tensor->nb[0] != sizeof(float) ||
        tensor->ne[0] <= 0 ||
        tensor->nb[1] % sizeof(float) != 0) {
        return false;
    }
    const size_t row_stride = tensor->nb[1] / sizeof(float);
    if (row_stride < static_cast<size_t>(tensor->ne[0]) ||
        !ggml_backend_hrx2_u32_size(row_stride, out_row_stride)) {
        return false;
    }
    if (tensor->ne[2] > 1) {
        if (tensor->nb[2] % sizeof(float) != 0 ||
            tensor->nb[2] / sizeof(float) != row_stride * static_cast<size_t>(tensor->ne[1])) {
            return false;
        }
    }
    if (tensor->ne[3] > 1) {
        if (tensor->nb[3] % sizeof(float) != 0 ||
            tensor->nb[3] / sizeof(float) != row_stride * static_cast<size_t>(tensor->ne[1]) * static_cast<size_t>(tensor->ne[2])) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx2_supports_set_rows(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    return op->op == GGML_OP_SET_ROWS &&
           src0 &&
           src1 &&
           src2 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I64 &&
           src2->type == op->type &&
           (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
           src0->ne[0] == op->ne[0] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           op->ne[1] > 0 &&
           src1->ne[0] == src0->ne[1] &&
           src0->ne[2] % src1->ne[1] == 0 &&
           src0->ne[3] % src1->ne[2] == 0 &&
           src1->ne[3] == 1 &&
           ggml_is_contiguous_rows(src0) &&
           ggml_is_contiguous_rows(op);
}

static bool ggml_backend_hrx2_supports_pointwise_binary(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if ((op->op != GGML_OP_ADD && op->op != GGML_OP_MUL && op->op != GGML_OP_DIV) ||
        !src0 || !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(src0, op) ||
        !ggml_is_contiguous(op) ||
        src0->ne[0] <= 0 ||
        ggml_nrows(src0) <= 0 ||
        src0->ne[0] > std::numeric_limits<uint32_t>::max() ||
        ggml_nrows(src0) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) > 1073741824ULL) {
        return false;
    }

    uint32_t src0_row_stride = 0;
    uint32_t src1_row_stride = 0;
    if (!ggml_backend_hrx2_flat_row_stride_f32(src0, &src0_row_stride)) {
        return false;
    }

    const int64_t nrows = ggml_nrows(src0);
    const bool same_shape_row_strided =
        ggml_are_same_shape(src1, src0) &&
        ggml_backend_hrx2_flat_row_stride_f32(src1, &src1_row_stride);
    const bool row_broadcast =
        src1->ne[0] == src0->ne[0] &&
        ggml_nrows(src1) == 1 &&
        src1->nb[0] == sizeof(float);
    const bool col_broadcast =
        src1->ne[0] == 1 &&
        ggml_nrows(src1) == nrows &&
        ggml_backend_hrx2_flat_row_stride_f32(src1, &src1_row_stride);
    const bool scalar_broadcast =
        src1->ne[0] == 1 &&
        ggml_nrows(src1) == 1 &&
        src1->nb[0] == sizeof(float);
    return same_shape_row_strided || row_broadcast || col_broadcast || scalar_broadcast;
}

static bool ggml_backend_hrx2_supports_scale(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_SCALE &&
           src0 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_clamp(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_CLAMP &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_sum_rows(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    uint32_t src0_row_stride = 0;
    return op->op == GGML_OP_SUM_ROWS &&
           src0 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           op->ne[0] == 1 &&
           op->ne[1] == src0->ne[1] &&
           op->ne[2] == src0->ne[2] &&
           op->ne[3] == src0->ne[3] &&
           ggml_backend_hrx2_flat_row_stride_f32(src0, &src0_row_stride) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_get_rows_moe_weights(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return op->op == GGML_OP_GET_ROWS &&
           src0 &&
           src1 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == 1 &&
           src0->ne[1] == 128 &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           src1->ne[0] == op->ne[1] &&
           src1->ne[1] == op->ne[2] &&
           src1->ne[2] == op->ne[3] &&
           src1->ne[3] == 1 &&
           op->ne[0] == 1 &&
           op->ne[1] == 8 &&
           op->ne[2] >= 1 &&
           op->ne[3] == 1 &&
           src0->nb[0] == sizeof(float) &&
           src0->nb[1] == sizeof(float) &&
           src0->nb[2] % sizeof(float) == 0 &&
           src1->nb[0] == sizeof(int32_t) &&
           src1->nb[1] % sizeof(int32_t) == 0 &&
           op->nb[0] == sizeof(float) &&
           op->nb[1] == sizeof(float) &&
           op->nb[2] % sizeof(float) == 0 &&
           src0->nb[2] / sizeof(float) >= static_cast<size_t>(src0->ne[1]) &&
           src1->nb[1] / sizeof(int32_t) >= static_cast<size_t>(src1->ne[0]) &&
           op->nb[2] / sizeof(float) >= static_cast<size_t>(op->ne[1]) &&
           op->ne[2] <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_cont(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_CONT &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->nb[0] == sizeof(float) &&
           ggml_is_contiguous(op) &&
           ggml_nelements(src0) == ggml_nelements(op) &&
           ggml_row_size(src0->type, src0->ne[0]) * ggml_nrows(src0) == ggml_nbytes(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_rope_f32(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    const int32_t n_dims = ggml_get_op_params_i32(op, 1);
    const int32_t mode = ggml_get_op_params_i32(op, 2);
    if (op->op != GGML_OP_ROPE ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_I32 ||
        op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(src0, op) ||
        (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) ||
        ggml_get_op_params_f32(op, 7) != 0.0f ||
        n_dims <= 0 ||
        n_dims > src0->ne[0] ||
        (n_dims % 2) != 0 ||
        (!src2 && n_dims != src0->ne[0]) ||
        src0->ne[0] <= 0 ||
        (src0->ne[0] % 2) != 0 ||
        src0->ne[1] <= 0 ||
        src0->ne[2] <= 0 ||
        src0->ne[3] != 1 ||
        src1->ne[0] != src0->ne[2] ||
        src1->ne[1] != 1 ||
        src1->ne[2] != 1 ||
        src1->ne[3] != 1 ||
        src0->nb[0] != sizeof(float) ||
        op->nb[0] != sizeof(float) ||
        src1->nb[0] != sizeof(int32_t) ||
        src0->nb[1] % sizeof(float) != 0 ||
        src0->nb[2] % sizeof(float) != 0 ||
        op->nb[1] % sizeof(float) != 0 ||
        op->nb[2] % sizeof(float) != 0 ||
        src1->nb[0] % sizeof(int32_t) != 0 ||
        src1->nb[1] % sizeof(int32_t) != 0) {
        return false;
    }

    if (src2) {
        if (src2->type != GGML_TYPE_F32 ||
            src2->ne[0] != n_dims / 2 ||
            src2->ne[1] != 1 ||
            src2->ne[2] != 1 ||
            src2->ne[3] != 1 ||
            src2->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(src2)) {
            return false;
        }
    }

    const uint64_t total_pairs =
        static_cast<uint64_t>(src0->ne[0] / 2) *
        static_cast<uint64_t>(src0->ne[1]) *
        static_cast<uint64_t>(src0->ne[2]);
    return src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[1] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[2] <= std::numeric_limits<uint32_t>::max() &&
           total_pairs <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_soft_max_f32(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    if (op->op != GGML_OP_SOFT_MAX ||
        !src0 ||
        src2 != nullptr ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(src0, op) ||
        !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(op) ||
        ggml_get_op_params_f32(op, 1) != 0.0f ||
        src0->ne[0] <= 0) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) ||
            src1->ne[0] != src0->ne[0] ||
            src1->ne[1] < src0->ne[1] ||
            src1->ne[2] <= 0 ||
            src1->ne[3] <= 0 ||
            (src0->ne[2] % src1->ne[2]) != 0 ||
            (src0->ne[3] % src1->ne[3]) != 0 ||
            src1->nb[1] % sizeof(float) != 0 ||
            src1->nb[2] % sizeof(float) != 0 ||
            src1->nb[3] % sizeof(float) != 0) {
            return false;
        }
    }

    return src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_swiglu(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const enum ggml_glu_op glu_op = ggml_get_glu_op(op);
    if (op->op != GGML_OP_GLU ||
        !src0 ||
        op->view_src != nullptr ||
        (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) ||
        ggml_get_op_params_i32(op, 1) != 0 ||
        src0->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        src0->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(op) ||
        op->ne[0] <= 0 ||
        ggml_nrows(op) <= 0 ||
        op->ne[0] > std::numeric_limits<uint32_t>::max() ||
        ggml_nrows(op) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (src1) {
        if (src1->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) ||
            !ggml_are_same_shape(src0, op) ||
            !ggml_are_same_shape(src1, op)) {
            return false;
        }
    } else if (src0->ne[0] != op->ne[0] * 2 ||
               ggml_nrows(src0) != ggml_nrows(op)) {
        return false;
    }
    const uint64_t total = static_cast<uint64_t>(op->ne[0]) * static_cast<uint64_t>(ggml_nrows(op));
    return total <= 536870912ULL;
}

static bool ggml_backend_hrx2_is_pow2_i64(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static bool ggml_backend_hrx2_supports_argsort_f32_i32_desc(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_ARGSORT &&
           src0 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_I32 &&
           ggml_get_op_params_i32(op, 0) == GGML_SORT_ORDER_DESC &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           src0->ne[0] <= 256 &&
           ggml_nrows(src0) > 0 &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_mul_mat_q8_0_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q8_0(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_f32_f32_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_f32_f32(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_q4_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q4_k(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_id_k_shape(
        const ggml_tensor * op,
        ggml_type src0_type,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    if (!ggml_backend_hrx2_supports_mul_mat_id_k(nullptr, op, src0_type)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    out_shape->k = static_cast<uint32_t>(src0->ne[0]);
    out_shape->rows = static_cast<uint32_t>(src0->ne[1]);
    out_shape->nexperts = static_cast<uint32_t>(src0->ne[2]);
    out_shape->nselected = static_cast<uint32_t>(src2->ne[0]);
    out_shape->ntokens = static_cast<uint32_t>(op->ne[2]);
    out_shape->src1_selected_stride = src1->ne[1] == 1 ? 0 : static_cast<uint32_t>(src1->nb[1] / sizeof(float));
    out_shape->src1_token_stride = static_cast<uint32_t>(src1->nb[2] / sizeof(float));
    out_shape->idx_token_stride = static_cast<uint32_t>(src2->nb[1] / sizeof(int32_t));
    out_shape->dst_token_stride = static_cast<uint32_t>(op->nb[2] / sizeof(float));
    return true;
}

static bool ggml_backend_hrx2_mul_mat_id_q4_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    return ggml_backend_hrx2_mul_mat_id_k_shape(op, GGML_TYPE_Q4_K, out_shape);
}

static bool ggml_backend_hrx2_mul_mat_id_q5_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    return ggml_backend_hrx2_mul_mat_id_k_shape(op, GGML_TYPE_Q5_K, out_shape);
}

static bool ggml_backend_hrx2_mul_mat_id_q6_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    return ggml_backend_hrx2_mul_mat_id_k_shape(op, GGML_TYPE_Q6_K, out_shape);
}

static bool ggml_backend_hrx2_mul_mat_q6_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q6_k(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_q5_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q5_k(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_extract_mul_mat_f16_f32_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_f16_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_f16_f32_batched(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_mul_mat_f16_shape shape = {};
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.k) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.rows) ||
        !ggml_backend_hrx2_u32(src1->ne[1], &shape.cols) ||
        !ggml_backend_hrx2_u32(op->ne[2], &shape.dst_ne2) ||
        !ggml_backend_hrx2_u32(op->ne[3], &shape.dst_ne3) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.src0_ne2) ||
        !ggml_backend_hrx2_u32(src0->ne[3], &shape.src0_ne3) ||
        !ggml_backend_hrx2_u32_size(src0->nb[1], &shape.src0_stride_row) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2], &shape.src0_stride_ne2) ||
        !ggml_backend_hrx2_u32_size(src0->nb[3], &shape.src0_stride_ne3) ||
        !ggml_backend_hrx2_u32_size(src1->nb[1], &shape.src1_stride_col) ||
        !ggml_backend_hrx2_u32_size(src1->nb[2], &shape.src1_stride_ne2) ||
        !ggml_backend_hrx2_u32_size(src1->nb[3], &shape.src1_stride_ne3) ||
        !ggml_backend_hrx2_u32_size(op->nb[1], &shape.dst_stride_col) ||
        !ggml_backend_hrx2_u32_size(op->nb[2], &shape.dst_stride_ne2) ||
        !ggml_backend_hrx2_u32_size(op->nb[3], &shape.dst_stride_ne3)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_rms_norm_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_rms_norm_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_rms_norm(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    *out_shape = {
        /* .ncols = */ static_cast<uint32_t>(src0->ne[0]),
        /* .nrows = */ static_cast<uint32_t>(ggml_nrows(src0)),
    };
    return true;
}

static bool ggml_backend_hrx2_extract_set_rows_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_set_rows_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_set_rows(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_set_rows_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.nc) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.nr) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ne02) ||
        !ggml_backend_hrx2_u32(src0->ne[3], &shape.ne03) ||
        !ggml_backend_hrx2_u32(op->ne[1], &shape.ne1) ||
        !ggml_backend_hrx2_u32(src1->ne[1], &shape.ne11) ||
        !ggml_backend_hrx2_u32(src1->ne[2], &shape.ne12) ||
        src0->nb[1] % ggml_type_size(src0->type) != 0 ||
        src0->nb[2] % ggml_type_size(src0->type) != 0 ||
        src0->nb[3] % ggml_type_size(src0->type) != 0 ||
        src1->nb[0] % sizeof(int64_t) != 0 ||
        src1->nb[1] % sizeof(int64_t) != 0 ||
        src1->nb[2] % sizeof(int64_t) != 0 ||
        op->nb[1] % ggml_type_size(op->type) != 0 ||
        op->nb[2] % ggml_type_size(op->type) != 0 ||
        op->nb[3] % ggml_type_size(op->type) != 0 ||
        !ggml_backend_hrx2_u32_size(src0->nb[1] / ggml_type_size(src0->type), &shape.src0_nb1) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / ggml_type_size(src0->type), &shape.src0_nb2) ||
        !ggml_backend_hrx2_u32_size(src0->nb[3] / ggml_type_size(src0->type), &shape.src0_nb3) ||
        !ggml_backend_hrx2_u32_size(src1->nb[0] / sizeof(int64_t), &shape.idx_nb0) ||
        !ggml_backend_hrx2_u32_size(src1->nb[1] / sizeof(int64_t), &shape.idx_nb1) ||
        !ggml_backend_hrx2_u32_size(src1->nb[2] / sizeof(int64_t), &shape.idx_nb2) ||
        !ggml_backend_hrx2_u32_size(op->nb[1] / ggml_type_size(op->type), &shape.dst_nb1) ||
        !ggml_backend_hrx2_u32_size(op->nb[2] / ggml_type_size(op->type), &shape.dst_nb2) ||
        !ggml_backend_hrx2_u32_size(op->nb[3] / ggml_type_size(op->type), &shape.dst_nb3)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_pointwise_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_pointwise_shape * out_shape) {
    if (!out_shape) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_pointwise_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows)) {
        return false;
    }
    if (op->op == GGML_OP_ADD || op->op == GGML_OP_MUL || op->op == GGML_OP_DIV) {
        if (!ggml_backend_hrx2_supports_pointwise_binary(nullptr, op)) {
            return false;
        }
        if (!ggml_backend_hrx2_flat_row_stride_f32(src0, &shape.src0_row_stride)) {
            return false;
        }
        if (ggml_are_same_shape(src1, src0)) {
            if (!ggml_backend_hrx2_flat_row_stride_f32(src1, &shape.src1_row_stride) ||
                !ggml_backend_hrx2_u32(src1->ne[0], &shape.src1_ncols)) {
                return false;
            }
        } else if (src1->ne[0] == 1 && ggml_nrows(src1) == ggml_nrows(src0)) {
            if (!ggml_backend_hrx2_flat_row_stride_f32(src1, &shape.src1_row_stride) ||
                !ggml_backend_hrx2_u32(src1->ne[0], &shape.src1_ncols)) {
                return false;
            }
        } else if ((src1->ne[0] == src0->ne[0] || src1->ne[0] == 1) && ggml_nrows(src1) == 1) {
            shape.src1_row_stride = 0;
            if (!ggml_backend_hrx2_u32(src1->ne[0], &shape.src1_ncols)) {
                return false;
            }
        } else {
            return false;
        }
    } else if (op->op == GGML_OP_SCALE) {
        if (!ggml_backend_hrx2_supports_scale(nullptr, op)) {
            return false;
        }
        shape.src0_row_stride = shape.ncols;
        shape.src1_row_stride = 0;
        shape.src1_ncols = 1;
    } else if (op->op == GGML_OP_CLAMP) {
        if (!ggml_backend_hrx2_supports_clamp(nullptr, op)) {
            return false;
        }
        shape.src0_row_stride = shape.ncols;
        shape.src1_row_stride = 0;
        shape.src1_ncols = 1;
    } else {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_sum_rows_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_sum_rows_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_sum_rows(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    ggml_backend_hrx2_sum_rows_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows) ||
        !ggml_backend_hrx2_flat_row_stride_f32(src0, &shape.src0_row_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_get_rows_moe_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_get_rows_moe_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_get_rows_moe_weights(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_get_rows_moe_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[1], &shape.nexperts) ||
        !ggml_backend_hrx2_u32(op->ne[1], &shape.nselected) ||
        !ggml_backend_hrx2_u32(op->ne[2], &shape.ntokens) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / sizeof(float), &shape.src0_token_stride) ||
        !ggml_backend_hrx2_u32_size(src1->nb[1] / sizeof(int32_t), &shape.idx_token_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[2] / sizeof(float), &shape.dst_token_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_argsort_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_argsort_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_argsort_f32_i32_desc(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    ggml_backend_hrx2_argsort_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_rope_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_rope_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_rope_f32(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_rope_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_get_op_params_i32(op, 1), &shape.n_dims) ||
        !ggml_backend_hrx2_u32(ggml_get_op_params_i32(op, 2), &shape.mode) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.nheads) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ntokens) ||
        !ggml_backend_hrx2_u32(src0->ne[1] * src0->ne[2], &shape.nrows) ||
        !ggml_backend_hrx2_u32_size(src0->nb[1] / sizeof(float), &shape.src0_head_stride) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / sizeof(float), &shape.src0_token_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[1] / sizeof(float), &shape.dst_head_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[2] / sizeof(float), &shape.dst_token_stride) ||
        !ggml_backend_hrx2_u32_size(src1->nb[0] / sizeof(int32_t), &shape.pos_token_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_soft_max_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_soft_max_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_soft_max_f32(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_soft_max_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.ne01) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ne02)) {
        return false;
    }
    shape.has_mask = src1 != nullptr;
    if (src1) {
        if (!ggml_backend_hrx2_u32_size(src1->nb[1] / sizeof(float), &shape.mask_nb1) ||
            !ggml_backend_hrx2_u32_size(src1->nb[2] / sizeof(float), &shape.mask_nb2) ||
            !ggml_backend_hrx2_u32_size(src1->nb[3] / sizeof(float), &shape.mask_nb3) ||
            !ggml_backend_hrx2_u32(src1->ne[1], &shape.mask_ne1) ||
            !ggml_backend_hrx2_u32(src1->ne[2], &shape.mask_ne2) ||
            !ggml_backend_hrx2_u32(src1->ne[3], &shape.mask_ne3)) {
            return false;
        }
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_cont_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_cont_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_cont(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    ggml_backend_hrx2_cont_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.ne1) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ne2) ||
        src0->nb[1] % sizeof(float) != 0 ||
        src0->nb[2] % sizeof(float) != 0 ||
        src0->nb[3] % sizeof(float) != 0 ||
        !ggml_backend_hrx2_u32_size(src0->nb[1] / sizeof(float), &shape.src_nb1) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / sizeof(float), &shape.src_nb2) ||
        !ggml_backend_hrx2_u32_size(src0->nb[3] / sizeof(float), &shape.src_nb3)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_swiglu_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_swiglu_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_swiglu(nullptr, op)) {
        return false;
    }
    ggml_backend_hrx2_swiglu_shape shape;
    if (!ggml_backend_hrx2_u32(op->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(op), &shape.nrows)) {
        return false;
    }
    shape.glu_op = ggml_get_glu_op(op);
    shape.split_sources = op->src[1] != nullptr;
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape) {
    if (!route ||
        shape.k < route->k_min || shape.k > route->k_max ||
        shape.rows < route->rows_min || shape.rows > route->rows_max ||
        shape.cols < route->cols_min || shape.cols > route->cols_max) {
        return false;
    }
    if (route->k_pow2_guard != 0) {
        const bool k_pow2 = ggml_backend_hrx2_is_pow2_i64(shape.k);
        if ((route->k_pow2_guard > 0) != k_pow2) {
            return false;
        }
    }
    if (route->all_pot_guard != 0) {
        const bool all_pot =
            ggml_backend_hrx2_is_pow2_i64(shape.k) &&
            ggml_backend_hrx2_is_pow2_i64(shape.rows) &&
            ggml_backend_hrx2_is_pow2_i64(shape.cols);
        if ((route->all_pot_guard > 0) != all_pot) {
            return false;
        }
    }
    if (route->k_multiple_of_guard != 0 && (shape.k % route->k_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_id_shape & shape) {
    if (!route ||
        shape.k < route->k_min || shape.k > route->k_max ||
        shape.rows < route->rows_min || shape.rows > route->rows_max ||
        shape.nselected < route->cols_min || shape.nselected > route->cols_max ||
        shape.ntokens < route->nrows_min || shape.ntokens > route->nrows_max) {
        return false;
    }
    if (route->k_multiple_of_guard != 0 && (shape.k % route->k_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_f16_shape & shape) {
    if (!route ||
        shape.k < route->k_min || shape.k > route->k_max ||
        shape.rows < route->rows_min || shape.rows > route->rows_max ||
        shape.cols < route->cols_min || shape.cols > route->cols_max) {
        return false;
    }
    if (route->k_pow2_guard != 0) {
        const bool k_pow2 = ggml_backend_hrx2_is_pow2_i64(shape.k);
        if ((route->k_pow2_guard > 0) != k_pow2) {
            return false;
        }
    }
    if (route->all_pot_guard != 0) {
        const bool all_pot =
            ggml_backend_hrx2_is_pow2_i64(shape.k) &&
            ggml_backend_hrx2_is_pow2_i64(shape.rows) &&
            ggml_backend_hrx2_is_pow2_i64(shape.cols);
        if ((route->all_pot_guard > 0) != all_pot) {
            return false;
        }
    }
    if (route->k_multiple_of_guard != 0 && (shape.k % route->k_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_set_rows_shape & shape) {
    if (!route ||
        shape.nc < route->ncols_min || shape.nc > route->ncols_max ||
        shape.nr < route->nrows_min || shape.nr > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.nc % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rms_norm_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_pointwise_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_sum_rows_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_get_rows_moe_shape & shape) {
    if (!route ||
        shape.nselected < route->ncols_min || shape.nselected > route->ncols_max ||
        shape.ntokens < route->nrows_min || shape.ntokens > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.nselected % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_argsort_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rope_shape & shape) {
    if (!route ||
        (!route->supports_mode.empty() &&
            !((route->supports_mode == "NORMAL" && shape.mode == GGML_ROPE_TYPE_NORMAL) ||
              (route->supports_mode == "NEOX"   && shape.mode == GGML_ROPE_TYPE_NEOX))) ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        (route->n_dims_min != 0 && shape.n_dims < route->n_dims_min) ||
        (route->n_dims_max != 0 && shape.n_dims > route->n_dims_max) ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max ||
        shape.nheads < route->rows_min || shape.nheads > route->rows_max ||
        shape.ntokens < route->cols_min || shape.ntokens > route->cols_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_soft_max_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max ||
        route->binding_count != (shape.has_mask ? 3u : 2u)) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_cont_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_swiglu_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    if (!route->supports_glu_op.empty() && route->supports_glu_op != ggml_backend_hrx2_glu_op_key(shape.glu_op)) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_make_rms_norm_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rms_norm_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_pointwise_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_pointwise_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.pointwise.src0_row_stride") {
            binding.value = std::to_string(shape.src0_row_stride);
        } else if (spec.value_source == "shape.pointwise.src1_row_stride") {
            binding.value = std::to_string(shape.src1_row_stride);
        } else if (spec.value_source == "shape.pointwise.src1_ncols") {
            binding.value = std::to_string(shape.src1_ncols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|src0_row_stride=" + std::to_string(shape.src0_row_stride);
        plan.cache_key += "|src1_row_stride=" + std::to_string(shape.src1_row_stride);
        plan.cache_key += "|src1_ncols=" + std::to_string(shape.src1_ncols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_sum_rows_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_sum_rows_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.sum_rows.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.sum_rows.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.sum_rows.src0_row_stride") {
            binding.value = std::to_string(shape.src0_row_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|src0_row_stride=" + std::to_string(shape.src0_row_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_get_rows_moe_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_get_rows_moe_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.get_rows_moe.nexperts") {
            binding.value = std::to_string(shape.nexperts);
        } else if (spec.value_source == "shape.get_rows_moe.nselected") {
            binding.value = std::to_string(shape.nselected);
        } else if (spec.value_source == "shape.get_rows_moe.ntokens") {
            binding.value = std::to_string(shape.ntokens);
        } else if (spec.value_source == "shape.get_rows_moe.src0_token_stride") {
            binding.value = std::to_string(shape.src0_token_stride);
        } else if (spec.value_source == "shape.get_rows_moe.idx_token_stride") {
            binding.value = std::to_string(shape.idx_token_stride);
        } else if (spec.value_source == "shape.get_rows_moe.dst_token_stride") {
            binding.value = std::to_string(shape.dst_token_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|nexperts=" + std::to_string(shape.nexperts);
        plan.cache_key += "|nselected=" + std::to_string(shape.nselected);
        plan.cache_key += "|ntokens=" + std::to_string(shape.ntokens);
        plan.cache_key += "|src0_token_stride=" + std::to_string(shape.src0_token_stride);
        plan.cache_key += "|idx_token_stride=" + std::to_string(shape.idx_token_stride);
        plan.cache_key += "|dst_token_stride=" + std::to_string(shape.dst_token_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_argsort_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_argsort_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.argsort.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.argsort.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_rope_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rope_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.rope.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.rope.n_dims") {
            binding.value = std::to_string(shape.n_dims);
        } else if (spec.value_source == "shape.rope.nheads") {
            binding.value = std::to_string(shape.nheads);
        } else if (spec.value_source == "shape.rope.ntokens") {
            binding.value = std::to_string(shape.ntokens);
        } else if (spec.value_source == "shape.rope.src0_head_stride") {
            binding.value = std::to_string(shape.src0_head_stride);
        } else if (spec.value_source == "shape.rope.src0_token_stride") {
            binding.value = std::to_string(shape.src0_token_stride);
        } else if (spec.value_source == "shape.rope.dst_head_stride") {
            binding.value = std::to_string(shape.dst_head_stride);
        } else if (spec.value_source == "shape.rope.dst_token_stride") {
            binding.value = std::to_string(shape.dst_token_stride);
        } else if (spec.value_source == "shape.rope.pos_token_stride") {
            binding.value = std::to_string(shape.pos_token_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|n_dims=" + std::to_string(shape.n_dims);
        plan.cache_key += "|mode=" + std::to_string(shape.mode);
        plan.cache_key += "|nheads=" + std::to_string(shape.nheads);
        plan.cache_key += "|ntokens=" + std::to_string(shape.ntokens);
        plan.cache_key += "|src0_head_stride=" + std::to_string(shape.src0_head_stride);
        plan.cache_key += "|src0_token_stride=" + std::to_string(shape.src0_token_stride);
        plan.cache_key += "|dst_head_stride=" + std::to_string(shape.dst_head_stride);
        plan.cache_key += "|dst_token_stride=" + std::to_string(shape.dst_token_stride);
        plan.cache_key += "|pos_token_stride=" + std::to_string(shape.pos_token_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_soft_max_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_soft_max_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.soft_max.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.soft_max.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.soft_max.ne01") {
            binding.value = std::to_string(shape.ne01);
        } else if (spec.value_source == "shape.soft_max.ne02") {
            binding.value = std::to_string(shape.ne02);
        } else if (spec.value_source == "shape.soft_max.mask_nb1") {
            binding.value = std::to_string(shape.mask_nb1);
        } else if (spec.value_source == "shape.soft_max.mask_nb2") {
            binding.value = std::to_string(shape.mask_nb2);
        } else if (spec.value_source == "shape.soft_max.mask_nb3") {
            binding.value = std::to_string(shape.mask_nb3);
        } else if (spec.value_source == "shape.soft_max.mask_ne1") {
            binding.value = std::to_string(shape.mask_ne1);
        } else if (spec.value_source == "shape.soft_max.mask_ne2") {
            binding.value = std::to_string(shape.mask_ne2);
        } else if (spec.value_source == "shape.soft_max.mask_ne3") {
            binding.value = std::to_string(shape.mask_ne3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|ne01=" + std::to_string(shape.ne01);
        plan.cache_key += "|ne02=" + std::to_string(shape.ne02);
        plan.cache_key += "|mask=" + std::to_string(shape.has_mask ? 1 : 0);
        if (shape.has_mask) {
            plan.cache_key += "|mask_nb1=" + std::to_string(shape.mask_nb1);
            plan.cache_key += "|mask_nb2=" + std::to_string(shape.mask_nb2);
            plan.cache_key += "|mask_nb3=" + std::to_string(shape.mask_nb3);
            plan.cache_key += "|mask_ne1=" + std::to_string(shape.mask_ne1);
            plan.cache_key += "|mask_ne2=" + std::to_string(shape.mask_ne2);
            plan.cache_key += "|mask_ne3=" + std::to_string(shape.mask_ne3);
        }
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_cont_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_cont_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.cont.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.cont.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.cont.ne1") {
            binding.value = std::to_string(shape.ne1);
        } else if (spec.value_source == "shape.cont.ne2") {
            binding.value = std::to_string(shape.ne2);
        } else if (spec.value_source == "shape.cont.src_nb1") {
            binding.value = std::to_string(shape.src_nb1);
        } else if (spec.value_source == "shape.cont.src_nb2") {
            binding.value = std::to_string(shape.src_nb2);
        } else if (spec.value_source == "shape.cont.src_nb3") {
            binding.value = std::to_string(shape.src_nb3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|ne1=" + std::to_string(shape.ne1);
        plan.cache_key += "|ne2=" + std::to_string(shape.ne2);
        plan.cache_key += "|src_nb1=" + std::to_string(shape.src_nb1);
        plan.cache_key += "|src_nb2=" + std::to_string(shape.src_nb2);
        plan.cache_key += "|src_nb3=" + std::to_string(shape.src_nb3);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_swiglu_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_swiglu_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }
    if ((shape.split_sources && route->binding_count != 3) ||
        (!shape.split_sources && route->binding_count != 2)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.swiglu.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.swiglu.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|glu_op=";
        plan.cache_key += ggml_backend_hrx2_glu_op_key(shape.glu_op);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q8_0_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source == "shape.q8_full_unroll_factor") {
            const uint32_t block_step = route->workgroup_size[0] / 8;
            if (block_step == 0 || (shape.k / 32) % block_step != 0) {
                return false;
            }
            binding.value = std::to_string((shape.k / 32) / block_step);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q4_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_id_q4_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_id_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.mul_mat_id.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.mul_mat_id.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.mul_mat_id.nexperts") {
            binding.value = std::to_string(shape.nexperts);
        } else if (spec.value_source == "shape.mul_mat_id.nselected") {
            binding.value = std::to_string(shape.nselected);
        } else if (spec.value_source == "shape.mul_mat_id.ntokens") {
            binding.value = std::to_string(shape.ntokens);
        } else if (spec.value_source == "shape.mul_mat_id.src1_selected_stride") {
            binding.value = std::to_string(shape.src1_selected_stride);
        } else if (spec.value_source == "shape.mul_mat_id.src1_token_stride") {
            binding.value = std::to_string(shape.src1_token_stride);
        } else if (spec.value_source == "shape.mul_mat_id.idx_token_stride") {
            binding.value = std::to_string(shape.idx_token_stride);
        } else if (spec.value_source == "shape.mul_mat_id.dst_token_stride") {
            binding.value = std::to_string(shape.dst_token_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|nexperts=" + std::to_string(shape.nexperts);
        plan.cache_key += "|nselected=" + std::to_string(shape.nselected);
        plan.cache_key += "|ntokens=" + std::to_string(shape.ntokens);
        plan.cache_key += "|src1_selected_stride=" + std::to_string(shape.src1_selected_stride);
        plan.cache_key += "|src1_token_stride=" + std::to_string(shape.src1_token_stride);
        plan.cache_key += "|idx_token_stride=" + std::to_string(shape.idx_token_stride);
        plan.cache_key += "|dst_token_stride=" + std::to_string(shape.dst_token_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q6_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q5_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_set_rows_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_set_rows_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.set_rows.nc") {
            binding.value = std::to_string(shape.nc);
        } else if (spec.value_source == "shape.set_rows.nr") {
            binding.value = std::to_string(shape.nr);
        } else if (spec.value_source == "shape.set_rows.ne02") {
            binding.value = std::to_string(shape.ne02);
        } else if (spec.value_source == "shape.set_rows.ne03") {
            binding.value = std::to_string(shape.ne03);
        } else if (spec.value_source == "shape.set_rows.ne1") {
            binding.value = std::to_string(shape.ne1);
        } else if (spec.value_source == "shape.set_rows.ne11") {
            binding.value = std::to_string(shape.ne11);
        } else if (spec.value_source == "shape.set_rows.ne12") {
            binding.value = std::to_string(shape.ne12);
        } else if (spec.value_source == "shape.set_rows.src0_nb1") {
            binding.value = std::to_string(shape.src0_nb1);
        } else if (spec.value_source == "shape.set_rows.src0_nb2") {
            binding.value = std::to_string(shape.src0_nb2);
        } else if (spec.value_source == "shape.set_rows.src0_nb3") {
            binding.value = std::to_string(shape.src0_nb3);
        } else if (spec.value_source == "shape.set_rows.idx_nb0") {
            binding.value = std::to_string(shape.idx_nb0);
        } else if (spec.value_source == "shape.set_rows.idx_nb1") {
            binding.value = std::to_string(shape.idx_nb1);
        } else if (spec.value_source == "shape.set_rows.idx_nb2") {
            binding.value = std::to_string(shape.idx_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb1") {
            binding.value = std::to_string(shape.dst_nb1);
        } else if (spec.value_source == "shape.set_rows.dst_nb2") {
            binding.value = std::to_string(shape.dst_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb3") {
            binding.value = std::to_string(shape.dst_nb3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|nc=" + std::to_string(shape.nc);
        plan.cache_key += "|nr=" + std::to_string(shape.nr);
        plan.cache_key += "|ne02=" + std::to_string(shape.ne02);
        plan.cache_key += "|ne03=" + std::to_string(shape.ne03);
        plan.cache_key += "|ne1=" + std::to_string(shape.ne1);
        plan.cache_key += "|ne11=" + std::to_string(shape.ne11);
        plan.cache_key += "|ne12=" + std::to_string(shape.ne12);
        plan.cache_key += "|src0_nb1=" + std::to_string(shape.src0_nb1);
        plan.cache_key += "|src0_nb2=" + std::to_string(shape.src0_nb2);
        plan.cache_key += "|src0_nb3=" + std::to_string(shape.src0_nb3);
        plan.cache_key += "|idx_nb0=" + std::to_string(shape.idx_nb0);
        plan.cache_key += "|idx_nb1=" + std::to_string(shape.idx_nb1);
        plan.cache_key += "|idx_nb2=" + std::to_string(shape.idx_nb2);
        plan.cache_key += "|dst_nb1=" + std::to_string(shape.dst_nb1);
        plan.cache_key += "|dst_nb2=" + std::to_string(shape.dst_nb2);
        plan.cache_key += "|dst_nb3=" + std::to_string(shape.dst_nb3);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_f16_f32_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_f16_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.mul_mat_f16.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.mul_mat_f16.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.mul_mat_f16.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_ne2") {
            binding.value = std::to_string(shape.dst_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_ne3") {
            binding.value = std::to_string(shape.dst_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_ne2") {
            binding.value = std::to_string(shape.src0_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_ne3") {
            binding.value = std::to_string(shape.src0_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_stride_row") {
            binding.value = std::to_string(shape.src0_stride_row);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_stride_ne2") {
            binding.value = std::to_string(shape.src0_stride_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_stride_ne3") {
            binding.value = std::to_string(shape.src0_stride_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.src1_stride_col") {
            binding.value = std::to_string(shape.src1_stride_col);
        } else if (spec.value_source == "shape.mul_mat_f16.src1_stride_ne2") {
            binding.value = std::to_string(shape.src1_stride_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.src1_stride_ne3") {
            binding.value = std::to_string(shape.src1_stride_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_stride_col") {
            binding.value = std::to_string(shape.dst_stride_col);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_stride_ne2") {
            binding.value = std::to_string(shape.dst_stride_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_stride_ne3") {
            binding.value = std::to_string(shape.dst_stride_ne3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        plan.cache_key += "|dst_ne2=" + std::to_string(shape.dst_ne2);
        plan.cache_key += "|dst_ne3=" + std::to_string(shape.dst_ne3);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_supports_mul_mat_q8_0_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q8_0_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q8_0_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q8_0_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_q4_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q4_k_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q4_k_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q4_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_f32_f32_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_f32_f32_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_f32_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q4_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_id_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op,
        ggml_type src0_type,
        const std::vector<const ggml_backend_hrx2_kernel_route *> & routes) {
    ggml_backend_hrx2_mul_mat_id_shape shape;
    if (!ggml_backend_hrx2_mul_mat_id_k_shape(op, src0_type, &shape)) {
        return false;
    }
    for (const auto * route : routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_id_q4_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q4_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k_route(
        device_context, op, GGML_TYPE_Q4_K, device_context->mul_mat_id_q4_k_routes);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q5_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k_route(
        device_context, op, GGML_TYPE_Q5_K, device_context->mul_mat_id_q5_k_routes);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q6_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k_route(
        device_context, op, GGML_TYPE_Q6_K, device_context->mul_mat_id_q6_k_routes);
}

static bool ggml_backend_hrx2_supports_mul_mat_q6_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q6_k_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q6_k_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q6_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_q5_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q5_k_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q5_k_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q5_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_set_rows_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_set_rows_shape shape;
    if (!ggml_backend_hrx2_extract_set_rows_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->set_rows_routes) {
        if ((op->type == GGML_TYPE_F16 && route->export_name != "hrx2_set_rows_f32_f16") ||
            (op->type == GGML_TYPE_F32 && route->export_name != "hrx2_set_rows_f32_f32")) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_set_rows_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_f16_f32_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_f16_shape shape;
    if (!ggml_backend_hrx2_extract_mul_mat_f16_f32_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_f16_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_f16_f32_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_rms_norm_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_rms_norm_shape shape;
    if (!ggml_backend_hrx2_extract_rms_norm_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->rms_norm_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_rms_norm_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static const std::vector<const ggml_backend_hrx2_kernel_route *> * ggml_backend_hrx2_pointwise_routes(
        const ggml_backend_hrx2_device_context * device_context,
        enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
            return &device_context->add_routes;
        case GGML_OP_MUL:
            return &device_context->mul_routes;
        case GGML_OP_DIV:
            return &device_context->div_routes;
        case GGML_OP_SCALE:
            return &device_context->scale_routes;
        case GGML_OP_CLAMP:
            return &device_context->clamp_routes;
        default:
            return nullptr;
    }
}

static bool ggml_backend_hrx2_supports_pointwise_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_pointwise_shape shape;
    if (!ggml_backend_hrx2_extract_pointwise_shape(op, &shape)) {
        return false;
    }
    const auto * routes = ggml_backend_hrx2_pointwise_routes(device_context, op->op);
    if (!routes) {
        return false;
    }
    for (const auto * route : *routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_pointwise_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_sum_rows_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_sum_rows_shape shape;
    if (!ggml_backend_hrx2_extract_sum_rows_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->sum_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_sum_rows_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_get_rows_moe_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_get_rows_moe_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_moe_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->get_rows_moe_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_get_rows_moe_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_argsort_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_argsort_shape shape;
    if (!ggml_backend_hrx2_extract_argsort_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->argsort_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_argsort_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_rope_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    const bool has_freq_factors = op->src[2] != nullptr;
    ggml_backend_hrx2_rope_shape shape;
    if (!ggml_backend_hrx2_extract_rope_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->rope_routes) {
        if ((has_freq_factors && route->binding_count != 4) ||
            (!has_freq_factors && route->binding_count != 3)) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_rope_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_soft_max_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_soft_max_shape shape;
    if (!ggml_backend_hrx2_extract_soft_max_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->soft_max_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_soft_max_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_cont_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_cont_shape shape;
    if (!ggml_backend_hrx2_extract_cont_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->cont_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_cont_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_swiglu_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_swiglu_shape shape;
    if (!ggml_backend_hrx2_extract_swiglu_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->swiglu_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_swiglu_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static std::string ggml_backend_hrx2_tensor_summary(const ggml_tensor * tensor) {
    if (!tensor) {
        return "null";
    }
    std::string result = "'";
    result += tensor->name;
    result += "' type=";
    result += ggml_type_name(tensor->type);
    result += " op=";
    result += ggml_op_name(tensor->op);
    result += " ne=[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i) {
            result += ",";
        }
        result += std::to_string(tensor->ne[i]);
    }
    result += "] nb=[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i) {
            result += ",";
        }
        result += std::to_string(tensor->nb[i]);
    }
    result += "]";
    result += tensor->view_src ? " view=true" : " view=false";
    result += tensor->buffer ? " buffer=true" : " buffer=false";
    return result;
}

static ggml_status ggml_backend_hrx2_dispatch_rms_norm(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_rms_norm_shape shape;
    if (!ggml_backend_hrx2_extract_rms_norm_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid RMS_NORM shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

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

    for (const auto * route : context->device_context->rms_norm_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_rms_norm_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "RMS_NORM") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length == sizeof(eps)) {
            constant_data = &eps;
            constant_size = sizeof(eps);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: RMS_NORM route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.nrows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                1,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "RMS_NORM") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", constants.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", constants.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: RMS_NORM provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_cont(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_cont_shape shape;
    if (!ggml_backend_hrx2_extract_cont_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid CONT shape during dispatch: dst=%s src0=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: CONT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->cont_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_cont_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "CONT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }
        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR("HRX2: CONT route %s has unsupported constant byte length %u\n",
                    provider->route.id.c_str(), provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "CONT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: CONT provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_swiglu(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_swiglu_shape shape;
    if (!ggml_backend_hrx2_extract_swiglu_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid GLU shape during dispatch: dst=%s src0=%s src1=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(dst->src[1]).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    uint32_t binding_count = 0;
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0])) {
        GGML_LOG_ERROR("HRX2: GLU tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }
    if (shape.split_sources) {
        if (!ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
            GGML_LOG_ERROR("HRX2: GLU tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 3;
    } else {
        if (!ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
            GGML_LOG_ERROR("HRX2: GLU tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 2;
    }

    for (const auto * route : context->device_context->swiglu_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_swiglu_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GLU") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("glu_op", ggml_backend_hrx2_glu_op_key(shape.glu_op)) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }
        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR("HRX2: GLU route %s has unsupported constant byte length %u\n",
                    provider->route.id.c_str(), provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("glu_op", ggml_backend_hrx2_glu_op_key(shape.glu_op)) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                binding_count,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: GLU provider is not available for glu_op=%s ncols=%u nrows=%u\n",
            ggml_backend_hrx2_glu_op_key(shape.glu_op), shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_pointwise(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_pointwise_shape shape;
    if (!ggml_backend_hrx2_extract_pointwise_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid pointwise shape during dispatch: dst=%s src0=%s src1=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str());
        return GGML_STATUS_FAILED;
    }

    const auto * routes = ggml_backend_hrx2_pointwise_routes(context->device_context, dst->op);
    if (!routes) {
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    uint32_t binding_count = 0;
    struct scale_constants {
        float scale;
        float bias;
    } scale = {};
    const void * constant_data = nullptr;
    size_t constant_size = 0;

    if (dst->op == GGML_OP_SCALE || dst->op == GGML_OP_CLAMP) {
        if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
            GGML_LOG_ERROR("HRX2: unary pointwise tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 2;
        std::memcpy(&scale, dst->op_params, sizeof(scale));
        constant_data = &scale;
        constant_size = sizeof(scale);
    } else {
        if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
            GGML_LOG_ERROR("HRX2: pointwise tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 3;
    }

    for (const auto * route : *routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_pointwise_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", ggml_op_name(dst->op)) + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->route.constant_byte_length != constant_size) {
            GGML_LOG_ERROR(
                "HRX2: pointwise route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                constant_size);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", ggml_op_name(dst->op)) + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                binding_count,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: pointwise provider is not available for op=%s ncols=%u nrows=%u\n",
            ggml_op_name(dst->op), shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_sum_rows(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_sum_rows_shape shape;
    if (!ggml_backend_hrx2_extract_sum_rows_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid SUM_ROWS shape during dispatch: dst=%s src0=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: SUM_ROWS tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->sum_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_sum_rows_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "SUM_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: SUM_ROWS route %s has constant byte length %u but dispatch has none\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { shape.nrows, 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "SUM_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: SUM_ROWS provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_get_rows_moe(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_get_rows_moe_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_moe_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid GET_ROWS MoE shape during dispatch: dst=%s src0=%s src1=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: GET_ROWS MoE tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->get_rows_moe_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_get_rows_moe_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
                ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
                ggml_backend_hrx2_json_kv("ntokens", shape.ntokens));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: GET_ROWS MoE route %s has constant byte length %u but dispatch has none\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.nselected) * static_cast<uint64_t>(shape.ntokens);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
            ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
            ggml_backend_hrx2_json_kv("ntokens", shape.ntokens) + "," +
            ggml_backend_hrx2_json_kv("src0_token_stride", shape.src0_token_stride) + "," +
            ggml_backend_hrx2_json_kv("idx_token_stride", shape.idx_token_stride) + "," +
            ggml_backend_hrx2_json_kv("dst_token_stride", shape.dst_token_stride) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: GET_ROWS MoE provider is not available for nexperts=%u nselected=%u ntokens=%u\n",
            shape.nexperts, shape.nselected, shape.ntokens);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_argsort(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_argsort_shape shape;
    if (!ggml_backend_hrx2_extract_argsort_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid ARGSORT shape during dispatch: dst=%s src0=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: ARGSORT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->argsort_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_argsort_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "ARGSORT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: ARGSORT route %s has constant byte length %u but dispatch has none\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { shape.nrows, 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "ARGSORT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: ARGSORT provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_rope(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    ggml_backend_hrx2_rope_shape shape;
    if (!ggml_backend_hrx2_extract_rope_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid ROPE shape during dispatch: dst=%s src0=%s src1=%s src2=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str(),
                ggml_backend_hrx2_tensor_summary(src2).c_str());
        return GGML_STATUS_FAILED;
    }

    const bool has_freq_factors = src2 != nullptr;
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        (has_freq_factors && !ggml_backend_hrx2_tensor_buffer_ref(src2, &bindings[2])) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[has_freq_factors ? 3 : 2])) {
        GGML_LOG_ERROR("HRX2: ROPE tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_rope_constants constants = {};
    std::memcpy(&constants.freq_base,   reinterpret_cast<const int32_t *>(dst->op_params) + 5, sizeof(float));
    std::memcpy(&constants.freq_scale,  reinterpret_cast<const int32_t *>(dst->op_params) + 6, sizeof(float));
    std::memcpy(&constants.attn_factor, reinterpret_cast<const int32_t *>(dst->op_params) + 8, sizeof(float));

    for (const auto * route : context->device_context->rope_routes) {
        const uint32_t expected_binding_count = has_freq_factors ? 4u : 3u;
        if (route->binding_count != expected_binding_count) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_rope_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "ROPE") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("n_dims", shape.n_dims) + "," +
                ggml_backend_hrx2_json_kv("mode", shape.mode) + "," +
                ggml_backend_hrx2_json_kv("nheads", shape.nheads) + "," +
                ggml_backend_hrx2_json_kv("ntokens", shape.ntokens));
            continue;
        }

        if (provider->route.constant_byte_length != sizeof(constants)) {
            GGML_LOG_ERROR(
                "HRX2: ROPE route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                sizeof(constants));
            continue;
        }

        const uint64_t total_pairs =
            static_cast<uint64_t>(shape.ncols / 2) *
            static_cast<uint64_t>(shape.nheads) *
            static_cast<uint64_t>(shape.ntokens);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((total_pairs + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "ROPE") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("n_dims", shape.n_dims) + "," +
            ggml_backend_hrx2_json_kv("mode", shape.mode) + "," +
            ggml_backend_hrx2_json_kv("nheads", shape.nheads) + "," +
            ggml_backend_hrx2_json_kv("ntokens", shape.ntokens) + "," +
            ggml_backend_hrx2_json_kv("has_freq_factors", has_freq_factors ? 1 : 0) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &constants,
                sizeof(constants),
                bindings,
                expected_binding_count,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: ROPE provider is not available for ncols=%u nheads=%u ntokens=%u\n",
        shape.ncols,
        shape.nheads,
        shape.ntokens);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_soft_max(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_soft_max_shape shape;
    if (!ggml_backend_hrx2_extract_soft_max_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid SOFT_MAX shape during dispatch: dst=%s src0=%s src1=%s src2=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str(),
                ggml_backend_hrx2_tensor_summary(dst->src[2]).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, shape.has_mask ? &bindings[2] : &bindings[1])) {
        GGML_LOG_ERROR("HRX2: SOFT_MAX tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }
    if (shape.has_mask && !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: SOFT_MAX mask tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_soft_max_constants constants = {};
    std::memcpy(&constants.scale, dst->op_params, sizeof(float));

    for (const auto * route : context->device_context->soft_max_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_soft_max_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "SOFT_MAX") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
                ggml_backend_hrx2_json_kv("has_mask", shape.has_mask ? 1 : 0));
            continue;
        }

        if (provider->route.constant_byte_length != sizeof(constants)) {
            GGML_LOG_ERROR(
                "HRX2: SOFT_MAX route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                sizeof(constants));
            continue;
        }

        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { shape.nrows, 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "SOFT_MAX") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("has_mask", shape.has_mask ? 1 : 0) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &constants,
                sizeof(constants),
                bindings,
                shape.has_mask ? 3 : 2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: SOFT_MAX provider is not available for ncols=%u nrows=%u has_mask=%d\n",
        shape.ncols,
        shape.nrows,
        shape.has_mask ? 1 : 0);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q8_0(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q8_0_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q8_0 shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };

    for (const auto * route : context->device_context->mul_mat_q8_0_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q8_0_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q8_0 provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_f32_f32(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT F32/F32 tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_f32_f32_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT F32/F32 shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->mul_mat_f32_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q4_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT F32/F32 route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (shape.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT F32/F32 provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4_K tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q4_k_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q4_K shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };

    for (const auto * route : context->device_context->mul_mat_q4_k_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q4_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT Q4_K route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q4_K provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst,
        ggml_type src0_type,
        const char * family,
        const char * type_label,
        const std::vector<const ggml_backend_hrx2_kernel_route *> & routes) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[3])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
            ggml_backend_hrx2_json_kv("family", family) + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)) + "," +
            ggml_backend_hrx2_json_kv("src2", ggml_backend_hrx2_tensor_summary(src2)));
        GGML_LOG_ERROR("HRX2: MUL_MAT_ID %s tensor is not backed by HRX2 buffers\n", type_label);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_id_shape shape;
    if (!ggml_backend_hrx2_mul_mat_id_k_shape(dst, src0_type, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
            ggml_backend_hrx2_json_kv("family", family) + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)) + "," +
            ggml_backend_hrx2_json_kv("src2", ggml_backend_hrx2_tensor_summary(src2)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT_ID %s shape during dispatch\n", type_label);
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_id_q4_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
                ggml_backend_hrx2_json_kv("family", family) + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
                ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
                ggml_backend_hrx2_json_kv("ntokens", shape.ntokens));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT_ID %s route %s has unsupported constant byte length %u\n",
                type_label,
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (shape.nselected + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                shape.ntokens,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
            ggml_backend_hrx2_json_kv("family", family) + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
            ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
            ggml_backend_hrx2_json_kv("ntokens", shape.ntokens) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_z", config.workgroup_count[2]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                4,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
                ggml_backend_hrx2_json_kv("family", family) + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: MUL_MAT_ID %s provider is not available for k=%u rows=%u nexperts=%u nselected=%u ntokens=%u\n",
        type_label,
        shape.k,
        shape.rows,
        shape.nexperts,
        shape.nselected,
        shape.ntokens);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_q4_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    return ggml_backend_hrx2_dispatch_mul_mat_id_k(
        context, dst, GGML_TYPE_Q4_K, "mul_mat_id_q4_k_f32", "Q4_K", context->device_context->mul_mat_id_q4_k_routes);
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_q5_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    return ggml_backend_hrx2_dispatch_mul_mat_id_k(
        context, dst, GGML_TYPE_Q5_K, "mul_mat_id_q5_k_f32", "Q5_K", context->device_context->mul_mat_id_q5_k_routes);
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_q6_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    return ggml_backend_hrx2_dispatch_mul_mat_id_k(
        context, dst, GGML_TYPE_Q6_K, "mul_mat_id_q6_k_f32", "Q6_K", context->device_context->mul_mat_id_q6_k_routes);
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q6_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT Q6_K tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q6_k_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q6_K shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };

    for (const auto * route : context->device_context->mul_mat_q6_k_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q6_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT Q6_K route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q6_K provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q5_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT Q5_K tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q5_k_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q5_K shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };

    for (const auto * route : context->device_context->mul_mat_q5_k_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q5_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT Q5_K route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q5_K provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_f16_f32(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_f16_shape shape;
    if (!ggml_backend_hrx2_extract_mul_mat_f16_f32_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT F16/F32 shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->mul_mat_f16_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_f16_f32_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
                ggml_backend_hrx2_json_kv("dst_ne2", shape.dst_ne2) + "," +
                ggml_backend_hrx2_json_kv("dst_ne3", shape.dst_ne3));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (shape.cols * shape.dst_ne2 * shape.dst_ne3 + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("dst_ne2", shape.dst_ne2) + "," +
            ggml_backend_hrx2_json_kv("dst_ne3", shape.dst_ne3) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: MUL_MAT F16/F32 provider is not available for k=%u rows=%u cols=%u dst_ne2=%u dst_ne3=%u\n",
        shape.k,
        shape.rows,
        shape.cols,
        shape.dst_ne2,
        shape.dst_ne3);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_set_rows_host_fallback(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst,
        const ggml_backend_hrx2_set_rows_shape & shape) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t src0_ref = {};
    hrx_buffer_ref_t src1_ref = {};
    hrx_buffer_ref_t dst_ref = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &src0_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &src1_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &dst_ref)) {
        GGML_LOG_ERROR("HRX2: SET_ROWS host fallback tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    if (!GGML_HRX2_CHECK(hrx_stream_synchronize(context->stream))) {
        return GGML_STATUS_FAILED;
    }

    std::vector<uint8_t> src0_host(src0_ref.length);
    std::vector<uint8_t> src1_host(src1_ref.length);
    std::vector<uint8_t> dst_host(dst_ref.length);

    if (!GGML_HRX2_CHECK(hrx_synchronous_d2h(
            context->device_context->device,
            src0_ref.buffer,
            src0_ref.offset,
            src0_host.data(),
            src0_host.size())) ||
        !GGML_HRX2_CHECK(hrx_synchronous_d2h(
            context->device_context->device,
            src1_ref.buffer,
            src1_ref.offset,
            src1_host.data(),
            src1_host.size())) ||
        !GGML_HRX2_CHECK(hrx_synchronous_d2h(
            context->device_context->device,
            dst_ref.buffer,
            dst_ref.offset,
            dst_host.data(),
            dst_host.size()))) {
        return GGML_STATUS_FAILED;
    }

    const uint8_t * src0_base = src0_host.data();
    const uint8_t * src1_base = src1_host.data();
    uint8_t * dst_base = dst_host.data();

    for (uint32_t i3 = 0; i3 < shape.ne03; ++i3) {
        const uint32_t i12 = shape.ne12 == 0 ? 0 : i3 % shape.ne12;
        for (uint32_t i2 = 0; i2 < shape.ne02; ++i2) {
            const uint32_t i11 = shape.ne11 == 0 ? 0 : i2 % shape.ne11;
            for (uint32_t i = 0; i < shape.nr; ++i) {
                const size_t idx_offset =
                    static_cast<size_t>(i) * src1->nb[0] +
                    static_cast<size_t>(i11) * src1->nb[1] +
                    static_cast<size_t>(i12) * src1->nb[2];
                if (idx_offset + sizeof(int64_t) > src1_host.size()) {
                    GGML_LOG_ERROR("HRX2: SET_ROWS host fallback index offset is out of bounds\n");
                    return GGML_STATUS_FAILED;
                }
                const int64_t row = *reinterpret_cast<const int64_t *>(src1_base + idx_offset);
                if (row < 0 || row >= static_cast<int64_t>(shape.ne1)) {
                    continue;
                }

                for (uint32_t i0 = 0; i0 < shape.nc; ++i0) {
                    const size_t src0_offset =
                        static_cast<size_t>(i0) * sizeof(float) +
                        static_cast<size_t>(i) * src0->nb[1] +
                        static_cast<size_t>(i2) * src0->nb[2] +
                        static_cast<size_t>(i3) * src0->nb[3];
                    if (src0_offset + sizeof(float) > src0_host.size()) {
                        GGML_LOG_ERROR("HRX2: SET_ROWS host fallback source offset is out of bounds\n");
                        return GGML_STATUS_FAILED;
                    }
                    const float value = *reinterpret_cast<const float *>(src0_base + src0_offset);
                    const size_t dst_offset =
                        static_cast<size_t>(i0) * ggml_type_size(dst->type) +
                        static_cast<size_t>(row) * dst->nb[1] +
                        static_cast<size_t>(i2) * dst->nb[2] +
                        static_cast<size_t>(i3) * dst->nb[3];
                    if (dst->type == GGML_TYPE_F16) {
                        if (dst_offset + sizeof(ggml_fp16_t) > dst_host.size()) {
                            GGML_LOG_ERROR("HRX2: SET_ROWS host fallback destination offset is out of bounds\n");
                            return GGML_STATUS_FAILED;
                        }
                        *reinterpret_cast<ggml_fp16_t *>(dst_base + dst_offset) = GGML_FP32_TO_FP16(value);
                    } else if (dst->type == GGML_TYPE_F32) {
                        if (dst_offset + sizeof(float) > dst_host.size()) {
                            GGML_LOG_ERROR("HRX2: SET_ROWS host fallback destination offset is out of bounds\n");
                            return GGML_STATUS_FAILED;
                        }
                        *reinterpret_cast<float *>(dst_base + dst_offset) = value;
                    } else {
                        GGML_LOG_ERROR("HRX2: SET_ROWS host fallback destination type is unsupported\n");
                        return GGML_STATUS_FAILED;
                    }
                }
            }
        }
    }

    ggml_backend_hrx2_trace_event(
        "dispatch",
        ggml_backend_hrx2_json_kv("op", "SET_ROWS") + "," +
        ggml_backend_hrx2_json_kv("route_id", std::string("host_fallback_set_rows_f32_") + ggml_type_name(dst->type)) + "," +
        ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
        ggml_backend_hrx2_json_kv("nc", shape.nc) + "," +
        ggml_backend_hrx2_json_kv("nr", shape.nr) + "," +
        ggml_backend_hrx2_json_kv("ne02", shape.ne02) + "," +
        ggml_backend_hrx2_json_kv("ne03", shape.ne03));

    if (!GGML_HRX2_CHECK(hrx_synchronous_h2d(
            context->device_context->device,
            dst_host.data(),
            dst_ref.buffer,
            dst_ref.offset,
            dst_host.size()))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx2_dispatch_set_rows(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: SET_ROWS tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_set_rows_shape shape;
    if (!ggml_backend_hrx2_extract_set_rows_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid SET_ROWS shape/type/layout during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    if (!ggml_backend_hrx2_env_enabled("GGML_HRX2_ENABLE_SET_ROWS_LOOM")) {
        return ggml_backend_hrx2_dispatch_set_rows_host_fallback(context, dst, shape);
    }

    for (const auto * route : context->device_context->set_rows_routes) {
        if ((dst->type == GGML_TYPE_F16 && route->export_name != "hrx2_set_rows_f32_f16") ||
            (dst->type == GGML_TYPE_F32 && route->export_name != "hrx2_set_rows_f32_f32")) {
            continue;
        }

        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_set_rows_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "SET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("nc", shape.nc) + "," +
                ggml_backend_hrx2_json_kv("nr", shape.nr));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: SET_ROWS route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total =
            static_cast<uint64_t>(shape.nc) *
            static_cast<uint64_t>(shape.nr) *
            static_cast<uint64_t>(shape.ne02) *
            static_cast<uint64_t>(shape.ne03);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size  = */ {
                workgroup_size,
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("nc", shape.nc) + "," +
            ggml_backend_hrx2_json_kv("nr", shape.nr) + "," +
            ggml_backend_hrx2_json_kv("ne02", shape.ne02) + "," +
            ggml_backend_hrx2_json_kv("ne03", shape.ne03) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: SET_ROWS provider is not available for nc=%u nr=%u dst=%s\n",
            shape.nc, shape.nr, ggml_type_name(dst->type));
    return ggml_backend_hrx2_dispatch_set_rows_host_fallback(context, dst, shape);
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
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
            case GGML_OP_SCALE:
            case GGML_OP_CLAMP:
                if (!ggml_backend_hrx2_supports_pointwise_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported pointwise shape/type/layout: dst=%s src0=%s src1=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_pointwise(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SUM_ROWS:
                if (!ggml_backend_hrx2_supports_sum_rows_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported SUM_ROWS shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_sum_rows(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GET_ROWS:
                if (!ggml_backend_hrx2_supports_get_rows_moe_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported GET_ROWS shape/type/layout: dst=%s src0=%s src1=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_get_rows_moe(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ARGSORT:
                if (!ggml_backend_hrx2_supports_argsort_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported ARGSORT shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_argsort(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ROPE:
                if (!ggml_backend_hrx2_supports_rope_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported ROPE shape/type/layout: dst=%s src0=%s src1=%s src2=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[2]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_rope(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SOFT_MAX:
                if (!ggml_backend_hrx2_supports_soft_max_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported SOFT_MAX shape/type/layout: dst=%s src0=%s src1=%s src2=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[2]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_soft_max(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CONT:
                if (!ggml_backend_hrx2_supports_cont_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported CONT shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_cont(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GLU:
                if (!ggml_backend_hrx2_supports_swiglu_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported GLU shape/type/layout: dst=%s src0=%s src1=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_swiglu(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_RMS_NORM:
                if (!ggml_backend_hrx2_supports_rms_norm_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported RMS_NORM shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_rms_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (ggml_backend_hrx2_supports_mul_mat_q8_0_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q8_0(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_q4_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q4_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_q5_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q5_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_q6_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q6_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_f32_f32_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_f32_f32(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_f16_f32_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_f16_f32(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                {
                    GGML_LOG_ERROR("HRX2: unsupported MUL_MAT shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                if (ggml_backend_hrx2_supports_mul_mat_id_q4_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_id_q4_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_id_q5_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_id_q5_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_id_q6_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_id_q6_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                {
                    GGML_LOG_ERROR("HRX2: unsupported MUL_MAT_ID shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SET_ROWS:
                if (!ggml_backend_hrx2_supports_set_rows_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported SET_ROWS shape/type/layout: dst=%s src0=%s src1=%s src2=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[2]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_set_rows(context, node) != GGML_STATUS_SUCCESS) {
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
            return ggml_backend_hrx2_supports_rms_norm_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
            return ggml_backend_hrx2_supports_pointwise_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_SUM_ROWS:
            return ggml_backend_hrx2_supports_sum_rows_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_GET_ROWS:
            return ggml_backend_hrx2_supports_get_rows_moe_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_ARGSORT:
            return ggml_backend_hrx2_supports_argsort_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_ROPE:
            return ggml_backend_hrx2_supports_rope_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_hrx2_supports_soft_max_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_CONT:
            return ggml_backend_hrx2_supports_cont_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_GLU:
            return ggml_backend_hrx2_supports_swiglu_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_MUL_MAT:
            return ggml_backend_hrx2_supports_mul_mat_q8_0_route(ggml_backend_hrx2_get_device_context(dev), op) ||
                   ggml_backend_hrx2_supports_mul_mat_q4_k_route(ggml_backend_hrx2_get_device_context(dev), op) ||
                   ggml_backend_hrx2_supports_mul_mat_q5_k_route(ggml_backend_hrx2_get_device_context(dev), op) ||
                   ggml_backend_hrx2_supports_mul_mat_q6_k_route(ggml_backend_hrx2_get_device_context(dev), op) ||
                   ggml_backend_hrx2_supports_mul_mat_f32_f32_route(ggml_backend_hrx2_get_device_context(dev), op) ||
                   ggml_backend_hrx2_supports_mul_mat_f16_f32_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_MUL_MAT_ID:
            return ggml_backend_hrx2_supports_mul_mat_id_q4_k_route(ggml_backend_hrx2_get_device_context(dev), op) ||
                   ggml_backend_hrx2_supports_mul_mat_id_q5_k_route(ggml_backend_hrx2_get_device_context(dev), op) ||
                   ggml_backend_hrx2_supports_mul_mat_id_q6_k_route(ggml_backend_hrx2_get_device_context(dev), op);
        case GGML_OP_SET_ROWS:
            return ggml_backend_hrx2_supports_set_rows_route(ggml_backend_hrx2_get_device_context(dev), op);
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
        device_context->catalog = ggml_backend_hrx2_load_catalog();
        if (device_context->catalog) {
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "rms_norm_f32",
                "RMS_NORM",
                &device_context->rms_norm_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q8_0_f32",
                "MUL_MAT",
                &device_context->mul_mat_q8_0_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_f32_f32",
                "MUL_MAT",
                &device_context->mul_mat_f32_f32_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q4_k_f32",
                "MUL_MAT",
                &device_context->mul_mat_q4_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_id_q4_k_f32",
                "MUL_MAT_ID",
                &device_context->mul_mat_id_q4_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q5_k_f32",
                "MUL_MAT",
                &device_context->mul_mat_q5_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_id_q5_k_f32",
                "MUL_MAT_ID",
                &device_context->mul_mat_id_q5_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q6_k_f32",
                "MUL_MAT",
                &device_context->mul_mat_q6_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_id_q6_k_f32",
                "MUL_MAT_ID",
                &device_context->mul_mat_id_q6_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_f16_f32_batched",
                "MUL_MAT",
                &device_context->mul_mat_f16_f32_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "cont_f32",
                "CONT",
                &device_context->cont_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "swiglu_f32",
                "GLU",
                &device_context->swiglu_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "set_rows_f32",
                "SET_ROWS",
                &device_context->set_rows_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "add_f32",
                "ADD",
                &device_context->add_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_f32",
                "MUL",
                &device_context->mul_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "div_f32",
                "DIV",
                &device_context->div_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "scale_f32",
                "SCALE",
                &device_context->scale_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "clamp_f32",
                "CLAMP",
                &device_context->clamp_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "sum_rows_f32",
                "SUM_ROWS",
                &device_context->sum_rows_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "get_rows_moe_weights_f32",
                "GET_ROWS",
                &device_context->get_rows_moe_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "argsort_f32_i32",
                "ARGSORT",
                &device_context->argsort_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                nullptr,
                "ROPE",
                &device_context->rope_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "soft_max_f32",
                "SOFT_MAX",
                &device_context->soft_max_routes);
            const auto route_less = [](const ggml_backend_hrx2_kernel_route * lhs, const ggml_backend_hrx2_kernel_route * rhs) {
                if (lhs->priority != rhs->priority) {
                    return lhs->priority > rhs->priority;
                }
                return lhs->id < rhs->id;
            };
            std::sort(
                device_context->rms_norm_routes.begin(),
                device_context->rms_norm_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q8_0_routes.begin(),
                device_context->mul_mat_q8_0_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q4_k_routes.begin(),
                device_context->mul_mat_q4_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_id_q4_k_routes.begin(),
                device_context->mul_mat_id_q4_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q5_k_routes.begin(),
                device_context->mul_mat_q5_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_id_q5_k_routes.begin(),
                device_context->mul_mat_id_q5_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q6_k_routes.begin(),
                device_context->mul_mat_q6_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_id_q6_k_routes.begin(),
                device_context->mul_mat_id_q6_k_routes.end(),
                route_less);
            std::sort(
                device_context->cont_routes.begin(),
                device_context->cont_routes.end(),
                route_less);
            std::sort(
                device_context->swiglu_routes.begin(),
                device_context->swiglu_routes.end(),
                route_less);
            std::sort(
                device_context->set_rows_routes.begin(),
                device_context->set_rows_routes.end(),
                route_less);
            std::sort(
                device_context->add_routes.begin(),
                device_context->add_routes.end(),
                route_less);
            std::sort(
                device_context->mul_routes.begin(),
                device_context->mul_routes.end(),
                route_less);
            std::sort(
                device_context->div_routes.begin(),
                device_context->div_routes.end(),
                route_less);
            std::sort(
                device_context->scale_routes.begin(),
                device_context->scale_routes.end(),
                route_less);
            std::sort(
                device_context->clamp_routes.begin(),
                device_context->clamp_routes.end(),
                route_less);
            std::sort(
                device_context->sum_rows_routes.begin(),
                device_context->sum_rows_routes.end(),
                route_less);
            std::sort(
                device_context->get_rows_moe_routes.begin(),
                device_context->get_rows_moe_routes.end(),
                route_less);
            std::sort(
                device_context->argsort_routes.begin(),
                device_context->argsort_routes.end(),
                route_less);
            std::sort(
                device_context->rope_routes.begin(),
                device_context->rope_routes.end(),
                route_less);
            std::sort(
                device_context->soft_max_routes.begin(),
                device_context->soft_max_routes.end(),
                route_less);
        }
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
