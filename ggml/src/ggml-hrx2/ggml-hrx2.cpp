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

struct ggml_backend_hrx2_mul_mat_shape {
    uint32_t k = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct ggml_backend_hrx2_rms_norm_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
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

static bool ggml_backend_hrx2_tensors_known_non_overlapping(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return true;
    }

    const auto * a_begin = static_cast<const uint8_t *>(a->data);
    const auto * b_begin = static_cast<const uint8_t *>(b->data);
    const auto * a_end = a_begin + ggml_nbytes(a);
    const auto * b_end = b_begin + ggml_nbytes(b);
    return a_end <= b_begin || b_end <= a_begin;
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
           ggml_backend_hrx2_tensors_known_non_overlapping(src0, op) &&
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
           ggml_backend_hrx2_tensors_known_non_overlapping(src0, src1) &&
           ggml_backend_hrx2_tensors_known_non_overlapping(src0, op) &&
           ggml_backend_hrx2_tensors_known_non_overlapping(src1, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_is_pow2_i64(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
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

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q8_0(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: MUL_MAT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q8_0_shape(dst, &shape)) {
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
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q8_0 provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
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
                if (!ggml_backend_hrx2_supports_rms_norm_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported RMS_NORM shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_rms_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (!ggml_backend_hrx2_supports_mul_mat_q8_0_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported MUL_MAT shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_mul_mat_q8_0(context, node) != GGML_STATUS_SUCCESS) {
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
        case GGML_OP_MUL_MAT:
            return ggml_backend_hrx2_supports_mul_mat_q8_0_route(ggml_backend_hrx2_get_device_context(dev), op);
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
