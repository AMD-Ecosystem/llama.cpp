#include "ggml-hrx.h"

#include "ggml-hrx-catalog.h"
#include "ggml-hrx-test.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "loom-jit/ggml-hrx-loom-jit.h"

#include "hrx_runtime.h"

#include <cerrno>
#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

static constexpr size_t GGML_HRX_ALIGNMENT = 256;
static constexpr uintptr_t GGML_HRX_FAKE_PTR_BASE = 0x1000;
static constexpr size_t GGML_HRX_STAGING_ARENA_DEFAULT_SIZE = 8 * 1024 * 1024;

struct ggml_backend_hrx_reg_context;

struct ggml_backend_hrx_test_dispatch_recorder {
    std::mutex mutex;
    bool enabled = false;
    std::map<std::string, ggml_backend_hrx_test_route_record> routes;
};

static ggml_backend_hrx_test_dispatch_recorder g_ggml_backend_hrx_test_dispatch_recorder;

struct ggml_backend_hrx_compiled_route {
    const ggml_backend_hrx_catalog_route * route = nullptr;
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    ggml_hrx_loom_jit_launch_config_t launch_config = {};
};

struct ggml_backend_hrx_compiled_route_deleter {
    void operator()(ggml_backend_hrx_compiled_route * route) const;
};

using ggml_backend_hrx_compiled_route_ptr =
    std::unique_ptr<ggml_backend_hrx_compiled_route, ggml_backend_hrx_compiled_route_deleter>;

struct ggml_backend_hrx_options {
    std::string catalog_dir;
    std::string evidence_dir;
    std::string trace_jsonl_path;
    std::string loom_sanitizer;
    std::string loom_sanitizer_reporting;
    bool trace_routes = false;
    bool trace_graph = false;
    size_t staging_arena_size = GGML_HRX_STAGING_ARENA_DEFAULT_SIZE;
};

static void ggml_backend_hrx_test_record_jit_compile(const std::string & route_id) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    if (!g_ggml_backend_hrx_test_dispatch_recorder.enabled) {
        return;
    }
    g_ggml_backend_hrx_test_dispatch_recorder.routes[route_id].jit_compile_count++;
}

static void ggml_backend_hrx_test_record_jit_cache_hit(const std::string & route_id) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    if (!g_ggml_backend_hrx_test_dispatch_recorder.enabled) {
        return;
    }
    g_ggml_backend_hrx_test_dispatch_recorder.routes[route_id].jit_cache_hit_count++;
}

static void ggml_backend_hrx_test_record_dispatch(const std::string & route_id) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    if (!g_ggml_backend_hrx_test_dispatch_recorder.enabled) {
        return;
    }
    g_ggml_backend_hrx_test_dispatch_recorder.routes[route_id].dispatch_count++;
}

struct ggml_backend_hrx_staging_arena {
    hrx_stream_t stream = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * mapped = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    std::vector<hrx_buffer_t> retired_buffers;
};

struct ggml_backend_hrx_device_context {
    ggml_backend_hrx_reg_context * reg_context = nullptr;
    const ggml_backend_hrx_options * options = nullptr;
    hrx_device_t device = nullptr;
    hrx_stream_t transfer_stream = nullptr;
    ggml_hrx_loom_jit_amdgpu_t jit = nullptr;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    std::mutex streams_mutex;
    std::vector<hrx_stream_t> live_streams;
    std::vector<ggml_backend_hrx_staging_arena> staging_arenas;
    hrx_stream_t active_stream = nullptr;
    std::mutex compiled_routes_mutex;
    std::map<std::string, ggml_backend_hrx_compiled_route_ptr> compiled_routes;
};

struct ggml_backend_hrx_reg_context {
    ggml_backend_hrx_options options;
    ggml_backend_hrx_catalog_ptr catalog;
    std::ofstream trace_jsonl;
    std::mutex trace_mutex;
    bool gpu_initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx_reg_context();
};

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    std::string name;
    hrx_buffer_params_t params = {};
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * base = nullptr;
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_stream_t stream = nullptr;
    std::string name;
};

struct ggml_backend_hrx_dispatch_request {
    ggml_backend_hrx_catalog_problem problem;
    std::vector<const ggml_tensor *> tensors;
    std::vector<uint8_t> constants;
};

static bool ggml_backend_hrx_log_status(hrx_status_t status, const char * expr, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }

    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define GGML_HRX_CHECK(expr) ggml_backend_hrx_log_status((expr), #expr, __FILE__, __LINE__)

void ggml_backend_hrx_compiled_route_deleter::operator()(ggml_backend_hrx_compiled_route * route) const {
    if (route && route->executable) {
        hrx_executable_release(route->executable);
        route->executable = nullptr;
    }
    delete route;
}

static size_t ggml_backend_hrx_align_up(size_t value, size_t alignment) {
    GGML_ASSERT(alignment > 0);
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

static size_t ggml_backend_hrx_staging_arena_capacity(const ggml_backend_hrx_device_context * device_context) {
    const size_t requested =
        device_context && device_context->options ?
        device_context->options->staging_arena_size :
        GGML_HRX_STAGING_ARENA_DEFAULT_SIZE;
    return ggml_backend_hrx_align_up(std::max(requested, GGML_HRX_ALIGNMENT), GGML_HRX_ALIGNMENT);
}

static std::string ggml_backend_hrx_test_case_index_key(const std::string & target_key, const std::string & family) {
    return target_key + "\n" + family;
}

static ggml_guid_t ggml_backend_hrx_guid(void) {
    static ggml_guid guid = {
        0x1c, 0x65, 0x79, 0x0a, 0x31, 0x8b, 0x4d, 0xa6,
        0x9e, 0x16, 0x6f, 0x13, 0x39, 0xb2, 0xe7, 0x5c,
    };
    return &guid;
}

static ggml_backend_hrx_device_context * ggml_backend_hrx_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_hrx_device_context *>(dev->context);
}

static ggml_backend_hrx_buffer_type_context * ggml_backend_hrx_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
}

static ggml_backend_hrx_buffer_context * ggml_backend_hrx_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer);

static const char * ggml_backend_hrx_getenv_once(const char * name) {
    return std::getenv(name);
}

static std::string ggml_backend_hrx_env_string(const char * name) {
    const char * value = ggml_backend_hrx_getenv_once(name);
    return value ? std::string(value) : std::string();
}

static bool ggml_backend_hrx_parse_bool_value(const std::string & value) {
    return !value.empty() && value != "0" && value != "false" && value != "FALSE" && value != "off" && value != "OFF";
}

static bool ggml_backend_hrx_env_bool(const char * name) {
    return ggml_backend_hrx_parse_bool_value(ggml_backend_hrx_env_string(name));
}

static std::optional<size_t> ggml_backend_hrx_parse_size_value(const std::string & value) {
    if (value.empty()) {
        return std::nullopt;
    }

    errno = 0;
    char * end = nullptr;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
    if (errno != 0 || end == value.c_str()) {
        return std::nullopt;
    }

    size_t multiplier = 1;
    if (*end != '\0') {
        if (end[1] != '\0') {
            if ((end[1] != 'b' && end[1] != 'B') || end[2] != '\0') {
                return std::nullopt;
            }
        }
        switch (*end) {
            case 'k':
            case 'K':
                multiplier = 1024;
                break;
            case 'm':
            case 'M':
                multiplier = 1024 * 1024;
                break;
            case 'g':
            case 'G':
                multiplier = 1024 * 1024 * 1024;
                break;
            default:
                return std::nullopt;
        }
    }

    if (parsed > std::numeric_limits<size_t>::max() / multiplier) {
        return std::nullopt;
    }
    return static_cast<size_t>(parsed) * multiplier;
}

static ggml_backend_hrx_options ggml_backend_hrx_parse_options() {
    ggml_backend_hrx_options options;
    options.catalog_dir = ggml_backend_hrx_env_string("GGML_HRX_CATALOG_DIR");
    options.evidence_dir = ggml_backend_hrx_env_string("GGML_HRX_EVIDENCE_DIR");
    options.trace_jsonl_path = ggml_backend_hrx_env_string("GGML_HRX_TRACE_JSONL");
    options.loom_sanitizer = ggml_backend_hrx_env_string("GGML_HRX_LOOM_SANITIZER");
    options.loom_sanitizer_reporting = ggml_backend_hrx_env_string("GGML_HRX_LOOM_SANITIZER_REPORTING");
    options.trace_routes = ggml_backend_hrx_env_bool("GGML_HRX_TRACE_ROUTES");
    options.trace_graph = ggml_backend_hrx_env_bool("GGML_HRX_TRACE_GRAPH");

    const std::string staging_size = ggml_backend_hrx_env_string("GGML_HRX_STAGING_ARENA_SIZE");
    if (!staging_size.empty()) {
        if (auto parsed = ggml_backend_hrx_parse_size_value(staging_size)) {
            options.staging_arena_size = *parsed;
        } else {
            GGML_LOG_ERROR(
                "%s: ignoring invalid GGML_HRX_STAGING_ARENA_SIZE=%s\n",
                __func__, staging_size.c_str());
        }
    }
    return options;
}

static const char * ggml_backend_hrx_optional_c_str(const std::string & value) {
    return value.empty() ? nullptr : value.c_str();
}

static void ggml_backend_hrx_trace_event(
        ggml_backend_hrx_reg_context * reg_context,
        nlohmann::json event) {
    if (!reg_context || !reg_context->trace_jsonl.is_open()) {
        return;
    }
    event["backend"] = GGML_HRX_NAME;
    std::lock_guard<std::mutex> lock(reg_context->trace_mutex);
    reg_context->trace_jsonl << event.dump() << '\n';
    reg_context->trace_jsonl.flush();
}

static void ggml_backend_hrx_write_evidence_file(
        ggml_backend_hrx_device_context * device_context,
        const std::string & name,
        const void * data,
        size_t size) {
    if (!device_context || !device_context->options || device_context->options->evidence_dir.empty() ||
        !data || size == 0) {
        return;
    }
    std::string path = device_context->options->evidence_dir;
    if (!path.empty() && path.back() != '/') {
        path += '/';
    }
    path += name;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        GGML_LOG_WARN("%s: failed to open evidence file %s\n", __func__, path.c_str());
        return;
    }
    file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
}

static size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static void ggml_backend_hrx_register_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    if (std::find(device_context->live_streams.begin(), device_context->live_streams.end(), stream) ==
            device_context->live_streams.end()) {
        device_context->live_streams.push_back(stream);
    }
}

static void ggml_backend_hrx_reset_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena.retired_buffers.clear();
    arena.offset = 0;
}

static void ggml_backend_hrx_release_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    if (arena.buffer) {
        hrx_buffer_release(arena.buffer);
    }
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena = {};
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_find_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    for (auto & arena : device_context->staging_arenas) {
        if (arena.stream == stream) {
            return &arena;
        }
    }
    return nullptr;
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_get_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
        return arena;
    }
    device_context->staging_arenas.push_back({});
    auto & arena = device_context->staging_arenas.back();
    arena.stream = stream;
    return &arena;
}

static void ggml_backend_hrx_unregister_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    auto & streams = device_context->live_streams;
    streams.erase(std::remove(streams.begin(), streams.end(), stream), streams.end());
    auto & arenas = device_context->staging_arenas;
    auto arena_it = std::find_if(
        arenas.begin(), arenas.end(),
        [stream](const ggml_backend_hrx_staging_arena & arena) { return arena.stream == stream; });
    if (arena_it != arenas.end()) {
        ggml_backend_hrx_release_staging_arena_locked(*arena_it);
        arenas.erase(arena_it);
    }
    if (device_context->active_stream == stream) {
        device_context->active_stream = nullptr;
    }
}

static hrx_stream_t ggml_backend_hrx_retain_timeline_stream(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t stream = device_context->active_stream;
    if (!stream) {
        stream = device_context->transfer_stream;
    }
    if (stream) {
        hrx_stream_retain(stream);
    }
    return stream;
}

static bool ggml_backend_hrx_sync_streams(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    bool ok = true;
    for (hrx_stream_t stream : device_context->live_streams) {
        ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
    return ok;
}

static bool ggml_backend_hrx_sync_graph_entry_streams(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t graph_stream) {
    if (!device_context) {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t streams[] = {
        device_context->active_stream,
        device_context->transfer_stream,
    };

    bool ok = true;
    for (hrx_stream_t stream : streams) {
        if (!stream || stream == graph_stream) {
            continue;
        }
        ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
    return ok;
}

static bool ggml_backend_hrx_prepare_stream_signal(
        hrx_stream_t stream,
        hrx_semaphore_t * semaphore,
        uint64_t * signal_value,
        hrx_semaphore_list_t * wait_list,
        hrx_semaphore_list_t * signal_list,
        hrx_semaphore_t * wait_semaphores,
        uint64_t * wait_values,
        hrx_semaphore_t * signal_semaphores,
        uint64_t * signal_values) {
    hrx_timeline_point_t position = {};
    if (!GGML_HRX_CHECK(hrx_stream_flush(stream)) ||
        !GGML_HRX_CHECK(hrx_stream_get_timeline_position(stream, &position)) ||
        !GGML_HRX_CHECK(hrx_stream_get_semaphore(stream, semaphore))) {
        return false;
    }

    *signal_value = position.value + 1;
    if (position.value > 0) {
        wait_semaphores[0] = *semaphore;
        wait_values[0] = position.value;
        *wait_list = {
            /* .semaphores = */ wait_semaphores,
            /* .values     = */ wait_values,
            /* .count      = */ 1,
        };
    } else {
        *wait_list = {};
    }

    signal_semaphores[0] = *semaphore;
    signal_values[0] = *signal_value;
    *signal_list = {
        /* .semaphores = */ signal_semaphores,
        /* .values     = */ signal_values,
        /* .count      = */ 1,
    };
    return true;
}

static bool ggml_backend_hrx_finish_stream_signal(hrx_stream_t stream, uint64_t signal_value) {
    uint64_t advanced_value = 0;
    if (!GGML_HRX_CHECK(hrx_stream_advance_timeline(stream, &advanced_value))) {
        return false;
    }
    if (advanced_value != signal_value) {
        GGML_LOG_ERROR(
            "%s: stream timeline advanced to %" PRIu64 ", expected %" PRIu64 "\n",
            __func__, advanced_value, signal_value);
        return false;
    }
    return GGML_HRX_CHECK(hrx_stream_wait(stream));
}

static bool ggml_backend_hrx_queue_fill_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t buffer,
        size_t offset,
        size_t size,
        const void * pattern,
        size_t pattern_size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous fill\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_fill(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, buffer, offset, size, pattern, pattern_size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_queue_copy_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t src,
        size_t src_offset,
        hrx_buffer_t dst,
        size_t dst_offset,
        size_t size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous copy\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_copy(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, src, src_offset, dst, dst_offset, size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_ensure_staging_buffer_locked(
        ggml_backend_hrx_device_context * device_context,
        ggml_backend_hrx_staging_arena * arena,
        size_t required_capacity) {
    if (arena->buffer && arena->capacity >= required_capacity && arena->mapped) {
        return true;
    }

    if (arena->buffer) {
        arena->retired_buffers.push_back(arena->buffer);
        arena->buffer = nullptr;
        arena->mapped = nullptr;
        arena->capacity = 0;
        arena->offset = 0;
    }

    const size_t capacity = ggml_backend_hrx_align_up(
        std::max(required_capacity, ggml_backend_hrx_staging_arena_capacity(device_context)),
        GGML_HRX_ALIGNMENT);
    hrx_buffer_params_t params = {
        /* .type = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        /* .access = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage = */ HRX_BUFFER_USAGE_DEFAULT |
                       HRX_BUFFER_USAGE_MAPPING_SCOPED |
                       HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_context->device), params, capacity, &arena->buffer))) {
        return false;
    }

    void * mapped = nullptr;
    if (!GGML_HRX_CHECK(hrx_buffer_map(arena->buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, capacity, &mapped))) {
        hrx_buffer_release(arena->buffer);
        arena->buffer = nullptr;
        return false;
    }
    arena->mapped = static_cast<uint8_t *>(mapped);
    arena->capacity = capacity;
    arena->offset = 0;
    return true;
}

static bool ggml_backend_hrx_stage_and_copy_tensor(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        const void * data,
        size_t buffer_offset,
        size_t buffer_size,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: upload for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor upload\n", __func__);
        return false;
    }

    std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
    auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
    if (!arena ||
        !ggml_backend_hrx_ensure_staging_buffer_locked(
            context->device_context, arena, ggml_backend_hrx_staging_arena_capacity(context->device_context))) {
        hrx_stream_release(stream);
        return false;
    }

    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t uploaded = 0;
    bool ok = true;
    while (uploaded < size) {
        size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
        if (staging_offset >= arena->capacity) {
            ok = GGML_HRX_CHECK(hrx_stream_flush(stream)) && GGML_HRX_CHECK(hrx_stream_wait(stream));
            if (!ok) {
                break;
            }
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
            staging_offset = 0;
        }

        const size_t available = arena->capacity - staging_offset;
        const size_t chunk_size = std::min(size - uploaded, available);
        if (chunk_size == 0) {
            GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
            ok = false;
            break;
        }

        std::memcpy(arena->mapped + staging_offset, bytes + uploaded, chunk_size);
        ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
            stream,
            arena->buffer,
            staging_offset,
            context->buffer,
            buffer_offset + uploaded,
            chunk_size));
        if (!ok) {
            break;
        }

        arena->offset = ggml_backend_hrx_align_up(staging_offset + chunk_size, GGML_HRX_ALIGNMENT);
        uploaded += chunk_size;
    }

    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_copy_tensor_to_staging(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        size_t buffer_offset,
        size_t buffer_size,
        void * data,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: readback for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor readback\n", __func__);
        return false;
    }

    auto * out_bytes = static_cast<uint8_t *>(data);
    size_t copied = 0;
    bool ok = true;
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
        if (!arena ||
            !ggml_backend_hrx_ensure_staging_buffer_locked(
                context->device_context, arena, ggml_backend_hrx_staging_arena_capacity(context->device_context))) {
            hrx_stream_release(stream);
            return false;
        }

        while (copied < size) {
            size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
            if (staging_offset >= arena->capacity) {
                ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
                if (!ok) {
                    break;
                }
                ggml_backend_hrx_reset_staging_arena_locked(*arena);
                staging_offset = 0;
            }

            const size_t chunk_size = std::min(size - copied, arena->capacity - staging_offset);
            if (chunk_size == 0) {
                GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
                ok = false;
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
                stream,
                context->buffer,
                buffer_offset + copied,
                arena->buffer,
                staging_offset,
                chunk_size));
            if (!ok) {
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
            if (!ok) {
                break;
            }
            std::memcpy(out_bytes + copied, arena->mapped + staging_offset, chunk_size);
            copied += chunk_size;
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }

    hrx_stream_release(stream);
    return ok;
}

static size_t ggml_backend_hrx_total_memory(hrx_device_t device) {
    uint64_t memory_total = 0;
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
            &memory_total, sizeof(memory_total)))) {
        return 0;
    }
    return static_cast<size_t>(memory_total);
}

static std::string ggml_backend_hrx_device_architecture(hrx_device_t device) {
    std::array<char, 128> architecture = {};
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        return std::string();
    }
    return std::string(architecture.data());
}

static std::string ggml_backend_hrx_device_description(hrx_device_t device) {
    std::array<char, 128> name = {};
    std::array<char, 128> architecture = {};

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_NAME, name.data(), name.size()))) {
        std::snprintf(name.data(), name.size(), "unknown");
    }

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        std::snprintf(architecture.data(), architecture.size(), "unknown");
    }

    std::string description(name.data());
    if (!description.empty() && architecture[0] != '\0') {
        description += " (";
        description += architecture.data();
        description += ")";
    }
    return description.empty() ? std::string("HRX GPU") : description;
}

static const char * ggml_backend_hrx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_hrx_get_buft_context(buft)->name.c_str();
}

static void ggml_backend_hrx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (context->buffer) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_get_buffer_context(buffer)->base;
}

static void ggml_backend_hrx_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    (void) ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, buffer_offset, size, &value, sizeof(value));
}

static void ggml_backend_hrx_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx_stage_and_copy_tensor(context, tensor, data, buffer_offset, buffer->size, size)) {
        GGML_LOG_ERROR("%s: failed to upload tensor %s through HRX staging\n", __func__, tensor->name);
    }
}

static void ggml_backend_hrx_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx_copy_tensor_to_staging(context, tensor, buffer_offset, buffer->size, data, size)) {
        GGML_LOG_ERROR("%s: failed to read tensor %s through HRX staging\n", __func__, tensor->name);
    }
}

static bool ggml_backend_hrx_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    if (!src_buffer || src_buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return false;
    }

    auto * dst_context = ggml_backend_hrx_get_buffer_context(buffer);
    auto * src_context = ggml_backend_hrx_get_buffer_context(src_buffer);
    if (dst_context->device_context != src_context->device_context ||
        !dst_context->buffer || !src_context->buffer) {
        return false;
    }

    if (!ggml_backend_hrx_sync_streams(dst_context->device_context)) {
        return false;
    }

    const size_t src_offset = ggml_backend_hrx_tensor_offset(src_context, src);
    const size_t dst_offset = ggml_backend_hrx_tensor_offset(dst_context, dst);
    const size_t size = ggml_nbytes(src);
    return ggml_backend_hrx_queue_copy_stream_sync(
        dst_context->device_context,
        src_context->buffer, src_offset,
        dst_context->buffer, dst_offset,
        size);
}

static void ggml_backend_hrx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (buffer->size == 0 || !context->buffer) {
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    (void) ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, 0, buffer->size, &value, sizeof(value));
}

static const ggml_backend_buffer_i ggml_backend_hrx_buffer_i = {
    /* .free_buffer   = */ ggml_backend_hrx_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_hrx_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hrx_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_hrx_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);

    hrx_buffer_t hrx_buffer = nullptr;
    if (size > 0 &&
        !GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(buft_context->device_context->device),
            buft_context->params, size, &hrx_buffer))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx_buffer_context {
        /* .device_context = */ buft_context->device_context,
        /* .buffer         = */ hrx_buffer,
        /* .base           = */ reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE),
    };
    if (!context) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_hrx_buffer_i, context, size);
    if (!buffer) {
        if (context->buffer) {
            hrx_buffer_release(context->buffer);
        }
        delete context;
    }
    return buffer;
}

static size_t ggml_backend_hrx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t ggml_backend_hrx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);
    return buft_context->device_context->memory_total > 0 ?
        buft_context->device_context->memory_total :
        std::numeric_limits<size_t>::max();
}

static const ggml_backend_buffer_type_i ggml_backend_hrx_buffer_type_i = {
    /* .get_name       = */ ggml_backend_hrx_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hrx_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hrx_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hrx_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_hrx_device_buffer_type(ggml_backend_dev_t dev) {
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    static std::vector<std::unique_ptr<ggml_backend_buffer_type>> buffer_types;
    static std::vector<std::unique_ptr<ggml_backend_hrx_buffer_type_context>> contexts;

    for (const auto & buft : buffer_types) {
        auto * context = ggml_backend_hrx_get_buft_context(buft.get());
        if (context->device_context == device_context) {
            return buft.get();
        }
    }

    auto * context = new ggml_backend_hrx_buffer_type_context {
        /* .device_context = */ device_context,
        /* .name           = */ device_context->name,
        /* .params         = */ {
            /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        },
    };

    auto * buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_hrx_buffer_type_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };

    contexts.emplace_back(context);
    buffer_types.emplace_back(buft);
    return buft;
}

static const char * ggml_backend_hrx_get_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static void ggml_backend_hrx_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        ggml_backend_hrx_unregister_stream(context->device_context, context->stream);
        hrx_stream_release(context->stream);
    }
    delete context;
    delete backend;
}

static void ggml_backend_hrx_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(context->device_context, context->stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
}

static bool ggml_backend_hrx_is_metadata_op(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static const char * ggml_backend_hrx_catalog_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return "F32";
        case GGML_TYPE_F16:
            return "F16";
        case GGML_TYPE_BF16:
            return "BF16";
        case GGML_TYPE_I32:
            return "I32";
        case GGML_TYPE_Q4_K:
            return "Q4_K";
        case GGML_TYPE_Q5_K:
            return "Q5_K";
        case GGML_TYPE_Q6_K:
            return "Q6_K";
        case GGML_TYPE_Q8_0:
            return "Q8_0";
        case GGML_TYPE_Q8_1:
            return "Q8_1";
        case GGML_TYPE_I64:
            return "I64";
        default:
            return ggml_type_name(type);
    }
}

static void ggml_backend_hrx_set_shape_alias(
        ggml_backend_hrx_catalog_problem * problem,
        const char * prefix,
        const char * key,
        int64_t value) {
    problem->shape[key] = value;
    std::string namespaced = prefix;
    namespaced += ".";
    namespaced += key;
    problem->shape[std::move(namespaced)] = value;
}

template <typename T>
static void ggml_backend_hrx_append_constant(std::vector<uint8_t> * constants, const T & value) {
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&value);
    constants->insert(constants->end(), bytes, bytes + sizeof(T));
}

static int64_t ggml_backend_hrx_tensor_row_count(const ggml_tensor * tensor) {
    return tensor ? tensor->ne[1] * tensor->ne[2] * tensor->ne[3] : 0;
}

static int64_t ggml_backend_hrx_tensor_row_stride_elements(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 ? static_cast<int64_t>(tensor->nb[1] / sizeof(float)) : 0;
}

static bool ggml_backend_hrx_is_f32_row_contiguous(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 && tensor->nb[0] == sizeof(float);
}

static bool ggml_backend_hrx_is_row_contiguous(const ggml_tensor * tensor) {
    return tensor && tensor->nb[0] == ggml_type_size(tensor->type);
}

static void ggml_backend_hrx_add_tensor_facts(
        ggml_backend_hrx_catalog_problem * problem,
        const char * prefix,
        const ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        problem->facts[std::string(prefix) + ".ne" + std::to_string(i)] = tensor->ne[i];
        problem->facts[std::string(prefix) + ".nb" + std::to_string(i)] = tensor->nb[i];
    }
}

static bool ggml_backend_hrx_request_matches_loaded_route(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_dispatch_request & request,
        const char * family) {
    if (!device_context || !device_context->reg_context || !device_context->reg_context->catalog) {
        return false;
    }
    const auto * route = ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, request.problem);
    return route && (!family || route->family == family);
}

static const ggml_tensor * ggml_backend_hrx_zero_offset_source_chain_target(
        const ggml_tensor * tensor,
        enum ggml_op op) {
    for (const ggml_tensor * cur = tensor; cur != nullptr;) {
        if (cur->op == op) {
            return cur;
        }
        if (cur->view_offs != 0) {
            return nullptr;
        }
        if (cur->view_src) {
            cur = cur->view_src;
            continue;
        }
        if ((cur->op == GGML_OP_VIEW ||
             cur->op == GGML_OP_RESHAPE ||
             cur->op == GGML_OP_PERMUTE ||
             cur->op == GGML_OP_TRANSPOSE) &&
            cur->src[0]) {
            cur = cur->src[0];
            continue;
        }
        return nullptr;
    }
    return nullptr;
}

static bool ggml_backend_hrx_make_mul_mat_problem(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_catalog_problem * out_problem) {
    if (!device_context || !node || node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1] || !out_problem) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(node)) {
        return false;
    }
    out_problem->op = "MUL_MAT";
    out_problem->target_key = device_context->architecture;
    out_problem->supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src0->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(src1->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32) ?
            "batched_attention_broadcast" : "contiguous"},
    };
    out_problem->shape = {
        {"k", src0->ne[0]},
        {"rows", src0->ne[1]},
        {"cols", src1->ne[1]},
    };
    out_problem->shape["mul_mat_f16.k"] = src0->ne[0];
    out_problem->shape["mul_mat_f16.rows"] = src0->ne[1];
    out_problem->shape["mul_mat_f16.cols"] = src1->ne[1];
    out_problem->shape["mul_mat_f16.dst_ne2"] = node->ne[2];
    out_problem->shape["mul_mat_f16.dst_ne3"] = node->ne[3];
    out_problem->shape["mul_mat_f16.src0_ne2"] = src0->ne[2];
    out_problem->shape["mul_mat_f16.src0_ne3"] = src0->ne[3];
    out_problem->shape["mul_mat_f16.src0_stride_row"] = src0->nb[1];
    out_problem->shape["mul_mat_f16.src0_stride_ne2"] = src0->nb[2];
    out_problem->shape["mul_mat_f16.src0_stride_ne3"] = src0->nb[3];
    out_problem->shape["mul_mat_f16.src1_stride_col"] = src1->nb[1];
    out_problem->shape["mul_mat_f16.src1_stride_ne2"] = src1->nb[2];
    out_problem->shape["mul_mat_f16.src1_stride_ne3"] = src1->nb[3];
    out_problem->shape["mul_mat_f16.dst_stride_col"] = node->nb[1];
    out_problem->shape["mul_mat_f16.dst_stride_ne2"] = node->nb[2];
    out_problem->shape["mul_mat_f16.dst_stride_ne3"] = node->nb[3];
    out_problem->facts = {
        {"src0.ne0", src0->ne[0]},
        {"src0.ne1", src0->ne[1]},
        {"src0.ne2", src0->ne[2]},
        {"src0.ne3", src0->ne[3]},
        {"src1.ne0", src1->ne[0]},
        {"src1.ne1", src1->ne[1]},
        {"src1.ne2", src1->ne[2]},
        {"src1.ne3", src1->ne[3]},
        {"dst.ne0", node->ne[0]},
        {"dst.ne1", node->ne[1]},
        {"dst.ne2", node->ne[2]},
        {"dst.ne3", node->ne[3]},
    };
    return true;
}

struct ggml_backend_hrx_binary_f32_layout {
    std::string key;
    bool rhs_row_broadcast = false;
};

static bool ggml_backend_hrx_describe_binary_f32_layout(
        const ggml_tensor * node,
        ggml_backend_hrx_binary_f32_layout * out_layout) {
    if (!node || !node->src[0] || !node->src[1] || !out_layout) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!ggml_backend_hrx_is_f32_row_contiguous(src0) ||
        !ggml_backend_hrx_is_f32_row_contiguous(src1) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }

    const int64_t ncols = node->ne[0];
    const int64_t nrows = ggml_backend_hrx_tensor_row_count(node);
    if (src0->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src0) != nrows) {
        return false;
    }
    if (src1->ne[0] == ncols && ggml_backend_hrx_tensor_row_count(src1) == nrows) {
        out_layout->key = "contiguous";
        out_layout->rhs_row_broadcast = false;
        return true;
    }
    if (src1->ne[0] == 1 && ggml_backend_hrx_tensor_row_count(src1) == nrows) {
        out_layout->key = "contiguous_src0_rhs_column_broadcast";
        out_layout->rhs_row_broadcast = false;
        return true;
    }
    if (src1->ne[0] == ncols && ggml_backend_hrx_tensor_row_count(src1) == 1) {
        out_layout->key = "contiguous_src0_rhs_row_broadcast";
        out_layout->rhs_row_broadcast = true;
        return true;
    }
    return false;
}

static void ggml_backend_hrx_set_pointwise_shape(
        ggml_backend_hrx_catalog_problem * problem,
        const ggml_tensor * node,
        bool rhs_row_broadcast) {
    ggml_backend_hrx_set_shape_alias(problem, "pointwise", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(problem, "pointwise", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_set_shape_alias(
        problem, "pointwise", "src0_row_stride",
        ggml_backend_hrx_tensor_row_stride_elements(node->src[0]));
    ggml_backend_hrx_set_shape_alias(
        problem, "pointwise", "src1_row_stride",
        rhs_row_broadcast ? 0 : ggml_backend_hrx_tensor_row_stride_elements(node->src[1]));
    ggml_backend_hrx_set_shape_alias(problem, "pointwise", "src1_ncols", node->src[1]->ne[0]);
}

static bool ggml_backend_hrx_make_add_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_ADD) {
        return false;
    }
    ggml_backend_hrx_binary_f32_layout layout;
    if (!ggml_backend_hrx_describe_binary_f32_layout(node, &layout)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "ADD";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout.key},
    };
    ggml_backend_hrx_set_pointwise_shape(&out_request->problem, node, layout.rhs_row_broadcast);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_rms_norm_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL ||
        !node->src[0] || !node->src[1] || node->src[0]->op != GGML_OP_RMS_NORM ||
        !node->src[0]->src[0] || node->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[1]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    const ggml_tensor * src = node->src[0]->src[0];
    const ggml_tensor * weight = node->src[1];
    const int64_t ncols = node->ne[0];
    const int64_t nrows = ggml_backend_hrx_tensor_row_count(node);
    if (src->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src) != nrows ||
        weight->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(weight) != 1) {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "MUL";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src->type)},
        {"weight_type", ggml_backend_hrx_catalog_type_name(weight->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_src_row_broadcast_weight"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm_mul", "ncols", ncols);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm_mul", "nrows", nrows);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", weight);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src, weight, node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->src[0]->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

static bool ggml_backend_hrx_make_add_rms_norm_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL ||
        !node->src[0] || !node->src[1] || node->src[0]->op != GGML_OP_RMS_NORM ||
        !node->src[0]->src[0] || node->src[0]->src[0]->op != GGML_OP_ADD ||
        !node->src[0]->src[0]->src[0] || !node->src[0]->src[0]->src[1] ||
        node->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->src[0]->type != GGML_TYPE_F32 ||
        node->src[0]->src[0]->src[1]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]->src[0]->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]->src[0]->src[1]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[1]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    const ggml_tensor * add = node->src[0]->src[0];
    const ggml_tensor * src0 = add->src[0];
    const ggml_tensor * src1 = add->src[1];
    const ggml_tensor * weight = node->src[1];
    const int64_t ncols = node->ne[0];
    const int64_t nrows = ggml_backend_hrx_tensor_row_count(node);
    if (src0->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src0) != nrows ||
        src1->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(src1) != nrows ||
        weight->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(weight) != 1 ||
        add->ne[0] != ncols || ggml_backend_hrx_tensor_row_count(add) != nrows) {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "MUL";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src_type", ggml_backend_hrx_catalog_type_name(src0->type)},
        {"weight_type", ggml_backend_hrx_catalog_type_name(weight->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_same_shape_add"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "add_rms_norm_mul", "ncols", ncols);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "add_rms_norm_mul", "nrows", nrows);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src0);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", src1);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "add_dst", add);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "weight", weight);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src0, src1, add, weight, node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->src[0]->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

static bool ggml_backend_hrx_make_mul_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL) {
        return false;
    }
    ggml_backend_hrx_dispatch_request fused_request;
    if (ggml_backend_hrx_make_add_rms_norm_mul_request(device_context, node, &fused_request) &&
        device_context->reg_context &&
        device_context->reg_context->catalog &&
        ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, fused_request.problem)) {
        *out_request = std::move(fused_request);
        return true;
    }
    if (ggml_backend_hrx_make_rms_norm_mul_request(device_context, node, &fused_request) &&
        device_context->reg_context &&
        device_context->reg_context->catalog &&
        ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, fused_request.problem)) {
        *out_request = std::move(fused_request);
        return true;
    }
    ggml_backend_hrx_binary_f32_layout layout;
    if (!ggml_backend_hrx_describe_binary_f32_layout(node, &layout)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "MUL";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout.key},
    };
    ggml_backend_hrx_set_pointwise_shape(&out_request->problem, node, layout.rhs_row_broadcast);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_div_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_DIV) {
        return false;
    }
    ggml_backend_hrx_binary_f32_layout layout;
    if (!ggml_backend_hrx_describe_binary_f32_layout(node, &layout)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "DIV";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout.key},
    };
    ggml_backend_hrx_set_pointwise_shape(&out_request->problem, node, layout.rhs_row_broadcast);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!out_request || !ggml_backend_hrx_make_mul_mat_problem(device_context, node, &out_request->problem)) {
        return false;
    }
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_id_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_MUL_MAT_ID ||
        !node->src[0] || !node->src[1] || !node->src[2] ||
        node->src[1]->type != GGML_TYPE_F32 || node->src[2]->type != GGML_TYPE_I32 ||
        node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node->src[0]) ||
        !ggml_is_contiguous(node->src[1]) || !ggml_is_contiguous(node->src[2]) ||
        !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * ids = node->src[2];
    const char * layout = nullptr;
    if (src0->type == GGML_TYPE_Q4_K) {
        layout = "q4_k_expert_planes_rhs_token_or_selected_rows";
    } else if (src0->type == GGML_TYPE_Q5_K) {
        layout = "q5_k_expert_planes_rhs_token_or_selected_rows";
    } else if (src0->type == GGML_TYPE_Q6_K) {
        layout = "q6_k_expert_planes_rhs_token_or_selected_rows";
    } else {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "MUL_MAT_ID";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src0->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(src1->type)},
        {"src2_type", ggml_backend_hrx_catalog_type_name(ids->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", layout},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "k", src0->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "rows", src0->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "nexperts", src0->ne[2]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "nselected", ids->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "mul_mat_id", "ntokens", ids->ne[1]);
    out_request->problem.shape["k"] = src0->ne[0];
    out_request->problem.shape["rows"] = src0->ne[1];
    out_request->problem.shape["cols"] = ids->ne[0];
    out_request->problem.shape["nrows"] = ids->ne[1];
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "src1_selected_stride",
        src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "src1_token_stride",
        static_cast<int64_t>(src1->nb[2] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "idx_token_stride",
        static_cast<int64_t>(ids->nb[1] / sizeof(int32_t)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "mul_mat_id", "dst_token_stride",
        static_cast<int64_t>(node->nb[2] / sizeof(float)));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src0);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", src1);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src2", ids);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src0, src1, ids, node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_scale_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SCALE || !node->src[0] ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "SCALE";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    float scale = 0.0f;
    float bias = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));
    std::memcpy(&bias, reinterpret_cast<const uint8_t *>(node->op_params) + sizeof(float), sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, scale);
    ggml_backend_hrx_append_constant(&out_request->constants, bias);
    return true;
}

static bool ggml_backend_hrx_make_clamp_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CLAMP || !node->src[0] ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "CLAMP";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "pointwise", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    float min_value = 0.0f;
    float max_value = 0.0f;
    std::memcpy(&min_value, node->op_params, sizeof(float));
    std::memcpy(&max_value, reinterpret_cast<const uint8_t *>(node->op_params) + sizeof(float), sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, min_value);
    ggml_backend_hrx_append_constant(&out_request->constants, max_value);
    return true;
}

static bool ggml_backend_hrx_make_rms_norm_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_RMS_NORM || !node->src[0] ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "RMS_NORM";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rms_norm", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    float eps = 0.0f;
    std::memcpy(&eps, node->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, eps);
    return true;
}

static bool ggml_backend_hrx_make_sum_rows_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SUM_ROWS || !node->src[0] ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "SUM_ROWS";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_row_reduction"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "sum_rows", "ncols", node->src[0]->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "sum_rows", "nrows", ggml_backend_hrx_tensor_row_count(node->src[0]));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "sum_rows", "src0_row_stride",
        ggml_backend_hrx_tensor_row_stride_elements(node->src[0]));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_soft_max_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SOFT_MAX || !node->src[0] ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    const ggml_tensor * mask = node->src[1];
    if (mask && !ggml_backend_hrx_is_f32_row_contiguous(mask)) {
        return false;
    }

    out_request->problem = {};
    out_request->problem.op = "SOFT_MAX";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", mask ? ggml_backend_hrx_catalog_type_name(mask->type) : "none"},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"mask", mask ? "required" : "none"},
        {"max_bias", "0"},
        {"sinks", "none"},
        {"layout", "contiguous"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "nrows", ggml_backend_hrx_tensor_row_count(node));
    out_request->problem.shape["rows"] = 0;
    out_request->problem.shape["cols"] = 0;
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "ne01", node->src[0]->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "ne02", node->src[0]->ne[2]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_nb1", mask ? mask->nb[1] : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_nb2", mask ? mask->nb[2] : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_nb3", mask ? mask->nb[3] : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_ne1", mask ? mask->ne[1] : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_ne2", mask ? mask->ne[2] : 0);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "soft_max", "mask_ne3", mask ? mask->ne[3] : 0);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    if (mask) {
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", mask);
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = mask ? std::vector<const ggml_tensor *>{node->src[0], mask, node} :
        std::vector<const ggml_tensor *>{node->src[0], node};
    out_request->constants.clear();
    float scale = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));
    ggml_backend_hrx_append_constant(&out_request->constants, scale);
    return true;
}

static bool ggml_backend_hrx_make_argsort_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_ARGSORT || !node->src[0] ||
        node->type != GGML_TYPE_I32 ||
        node->op_params[0] != static_cast<int32_t>(GGML_SORT_ORDER_DESC) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_is_contiguous(node)) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "ARGSORT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"order", "DESC"},
        {"layout", "contiguous_rows"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "argsort", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "argsort", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_get_rows_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_GET_ROWS ||
        !node->src[0] || !node->src[1] || node->src[1]->type != GGML_TYPE_I32 ||
        !ggml_backend_hrx_is_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    const bool quantized_src = ggml_is_quantized(node->src[0]->type);
    if (node->src[0]->type != GGML_TYPE_F32 && !quantized_src) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "GET_ROWS";
    out_request->problem.target_key = device_context->architecture;
    const bool moe_weights_view =
        node->src[0]->type == GGML_TYPE_F32 &&
        node->src[0]->ne[0] == 1 &&
        node->src[0]->ne[2] == node->ne[2] &&
        node->src[0]->ne[3] == node->ne[3] &&
        node->src[1]->ne[0] == node->ne[1] &&
        node->src[1]->ne[1] == node->ne[2] &&
        node->src[1]->ne[2] == node->ne[3] &&
        node->ne[0] == 1 &&
        node->ne[1] > 0 &&
        node->ne[2] > 0;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", moe_weights_view ? "moe_weights_topk_view" :
            (quantized_src ? "quantized_embedding_rows_1d" : "embedding_rows_1d")},
    };
    if (moe_weights_view) {
        // Route domains for the MoE gather are keyed by the logical output
        // matrix [topk, tokens], while the underlying ggml_get_rows tensor is
        // shaped [1, topk, tokens]. Keep both fact families available without
        // letting the standard get_rows aliases overwrite the route keys.
        out_request->problem.shape["ncols"] = node->ne[1];
        out_request->problem.shape["nrows"] = node->ne[2];
        out_request->problem.shape["get_rows.ncols"] = node->ne[0];
        out_request->problem.shape["get_rows.nrows"] = node->ne[1];
    } else {
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "ncols", node->ne[0]);
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "nrows", node->ne[1]);
    }
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows", "src0_nrows", node->src[0]->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows_moe", "nexperts", node->src[0]->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows_moe", "nselected", node->src[1]->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "get_rows_moe", "ntokens", node->src[1]->ne[1]);
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows_moe", "src0_token_stride",
        static_cast<int64_t>(node->src[0]->nb[2] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows_moe", "idx_token_stride",
        static_cast<int64_t>(node->src[1]->nb[1] / sizeof(int32_t)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows_moe", "dst_token_stride",
        static_cast<int64_t>(node->nb[2] / sizeof(float)));
    ggml_backend_hrx_set_shape_alias(
        &out_request->problem, "get_rows", "idx_row_stride",
        node->src[1]->ne[1] == 1 ? 1 : node->src[1]->nb[1] / sizeof(int32_t));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node->src[1], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_set_rows_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_SET_ROWS ||
        !node->src[0] || !node->src[1] || !node->src[2] ||
        node->src[0]->type != GGML_TYPE_F32 ||
        (node->src[1]->type != GGML_TYPE_I64 && node->src[1]->type != GGML_TYPE_I32) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_is_contiguous(node->src[1]) ||
        !ggml_backend_hrx_is_row_contiguous(node)) {
        return false;
    }

    auto make_cont_set_rows = [&]() -> bool {
        const ggml_tensor * cont = ggml_backend_hrx_zero_offset_source_chain_target(node->src[0], GGML_OP_CONT);
        if (!cont || !cont->src[0] || cont->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F16 ||
            node->src[1]->type != GGML_TYPE_I64 ||
            !ggml_backend_hrx_is_f32_row_contiguous(cont->src[0]) ||
            !ggml_backend_hrx_is_f32_row_contiguous(cont) ||
            cont->view_src != nullptr ||
            ggml_nelements(node->src[0]) != ggml_nelements(cont)) {
            return false;
        }

        ggml_backend_hrx_dispatch_request request = {};
        request.problem.op = "SET_ROWS";
        request.problem.target_key = device_context->architecture;
        request.problem.supports = {
            {"src0_type", ggml_backend_hrx_catalog_type_name(cont->type)},
            {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
            {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
            {"layout", "cont_src0_to_reshape_set_rows"},
            {"cont_ncols", std::to_string(cont->ne[0])},
        };
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "ncols", cont->ne[0]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "ne1", cont->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "ne2", cont->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "src_nb1", cont->src[0]->nb[1] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "src_nb2", cont->src[0]->nb[2] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "cont", "src_nb3", cont->src[0]->nb[3] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "nc", node->ne[0]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "nr", node->src[0]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne02", node->src[0]->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne03", node->src[0]->ne[3]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne1", node->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne11", node->src[1]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne12", node->src[1]->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "src0_nb1", node->src[0]->nb[1] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "src0_nb2", node->src[0]->nb[2] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "src0_nb3", node->src[0]->nb[3] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb0", node->src[1]->nb[0] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb1", node->src[1]->nb[1] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb2", node->src[1]->nb[2] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb1", node->nb[1] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb2", node->nb[2] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb3", node->nb[3] / sizeof(ggml_fp16_t));
        request.problem.shape["ncols"] = node->ne[0];
        request.problem.shape["nrows"] = node->src[0]->ne[1];
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", cont->src[0]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", node->src[1]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
        request.tensors = {cont->src[0], node->src[1], node};
        request.constants.clear();
        if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "cont_set_rows_f32")) {
            return false;
        }
        *out_request = std::move(request);
        return true;
    };

    auto make_rope_set_rows = [&]() -> bool {
        const ggml_tensor * rope = ggml_backend_hrx_zero_offset_source_chain_target(node->src[0], GGML_OP_ROPE);
        if (!rope || !rope->src[0] || !rope->src[1] || !rope->src[2] ||
            rope->src[0]->type != GGML_TYPE_F32 || rope->src[1]->type != GGML_TYPE_I32 ||
            rope->src[2]->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32 ||
            node->src[1]->type != GGML_TYPE_I64 || node->type != GGML_TYPE_F16 ||
            !ggml_backend_hrx_is_f32_row_contiguous(rope->src[0]) ||
            !ggml_is_contiguous(rope->src[1]) ||
            !ggml_is_contiguous(rope->src[2]) ||
            !ggml_backend_hrx_is_f32_row_contiguous(rope) ||
            rope->view_src != nullptr ||
            node->src[0]->ne[0] != rope->ne[0] * rope->ne[1] ||
            node->src[0]->ne[1] != rope->ne[2] ||
            node->src[0]->ne[2] != 1 ||
            node->src[0]->ne[3] != 1 ||
            node->src[0]->nb[0] != sizeof(float) ||
            node->src[0]->nb[1] != rope->nb[2] ||
            node->ne[0] != rope->ne[0] * rope->ne[1]) {
            return false;
        }

        int32_t n_dims = 0;
        int32_t mode = 0;
        float freq_base = 10000.0f;
        float freq_scale = 1.0f;
        float ext_factor = 0.0f;
        float attn_factor = 1.0f;
        std::memcpy(&n_dims, reinterpret_cast<const uint8_t *>(rope->op_params) + sizeof(int32_t), sizeof(int32_t));
        std::memcpy(&mode, reinterpret_cast<const uint8_t *>(rope->op_params) + 2 * sizeof(int32_t), sizeof(int32_t));
        std::memcpy(&freq_base, reinterpret_cast<const uint8_t *>(rope->op_params) + 5 * sizeof(float), sizeof(float));
        std::memcpy(&freq_scale, reinterpret_cast<const uint8_t *>(rope->op_params) + 6 * sizeof(float), sizeof(float));
        std::memcpy(&ext_factor, reinterpret_cast<const uint8_t *>(rope->op_params) + 7 * sizeof(float), sizeof(float));
        std::memcpy(&attn_factor, reinterpret_cast<const uint8_t *>(rope->op_params) + 8 * sizeof(float), sizeof(float));
        if (ext_factor != 0.0f || (mode & ~(GGML_ROPE_TYPE_NEOX)) != 0) {
            return false;
        }

        ggml_backend_hrx_dispatch_request request = {};
        request.problem.op = "SET_ROWS";
        request.problem.target_key = device_context->architecture;
        request.problem.supports = {
            {"src0_type", ggml_backend_hrx_catalog_type_name(rope->src[0]->type)},
            {"src1_type", ggml_backend_hrx_catalog_type_name(rope->src[1]->type)},
            {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
            {"mode", (mode & GGML_ROPE_TYPE_NEOX) ? "NEOX" : "NORMAL"},
            {"freq_factor", "present"},
            {"ext_factor", "0"},
            {"layout", "rope-view-set-rows-f16"},
        };
        const int64_t nheads = rope->src[0]->ne[1];
        const int64_t ntokens = rope->src[0]->ne[2];
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "ncols", rope->src[0]->ne[0]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "n_dims", n_dims);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "nheads", nheads);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "ntokens", ntokens);
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "src0_head_stride", rope->src[0]->nb[1] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "src0_token_stride", rope->src[0]->nb[2] / sizeof(float));
        ggml_backend_hrx_set_shape_alias(&request.problem, "rope", "pos_token_stride", rope->src[1]->nb[0] / sizeof(int32_t));
        request.problem.shape["ncols"] = rope->src[0]->ne[0];
        request.problem.shape["nrows"] = nheads * ntokens;
        request.problem.shape["n_dims"] = n_dims;
        request.problem.shape["rows"] = nheads;
        request.problem.shape["cols"] = ntokens;
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne1", node->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne11", node->src[1]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "ne12", node->src[1]->ne[2]);
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb0", node->src[1]->nb[0] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb1", node->src[1]->nb[1] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "idx_nb2", node->src[1]->nb[2] / sizeof(int64_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb1", node->nb[1] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb2", node->nb[2] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_set_shape_alias(&request.problem, "set_rows", "dst_nb3", node->nb[3] / sizeof(ggml_fp16_t));
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", rope->src[0]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", rope->src[1]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "src2", rope->src[2]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "idx", node->src[1]);
        ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
        request.tensors = {rope->src[0], rope->src[1], rope->src[2], node->src[1], node};
        request.constants.clear();
        ggml_backend_hrx_append_constant<float>(&request.constants, std::pow(freq_base, -2.0f / static_cast<float>(n_dims)));
        ggml_backend_hrx_append_constant<float>(&request.constants, freq_scale);
        ggml_backend_hrx_append_constant<float>(&request.constants, attn_factor);
        if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "rope_set_rows_f32")) {
            return false;
        }
        *out_request = std::move(request);
        return true;
    };

    if (make_rope_set_rows() || make_cont_set_rows()) {
        return true;
    }

    const ggml_tensor * src = node->src[0];
    const ggml_tensor * idx = node->src[1];
    out_request->problem = {};
    out_request->problem.op = "SET_ROWS";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(src->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(idx->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_rows"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "nc", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "nr", src->ne[1]);
    out_request->problem.shape["ncols"] = node->ne[0];
    out_request->problem.shape["nrows"] = src->ne[1];
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne02", src->ne[2]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne03", src->ne[3]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne1", node->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne11", idx->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "ne12", idx->ne[2]);
    const size_t dst_element_size = node->type == GGML_TYPE_F16 ? sizeof(ggml_fp16_t) : sizeof(float);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "src0_nb1", src->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "src0_nb2", src->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "src0_nb3", src->nb[3] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "idx_nb0", idx->nb[0] / sizeof(int64_t));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "idx_nb1", idx->nb[1] / sizeof(int64_t));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "idx_nb2", idx->nb[2] / sizeof(int64_t));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "dst_nb1", node->nb[1] / dst_element_size);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "dst_nb2", node->nb[2] / dst_element_size);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "set_rows", "dst_nb3", node->nb[3] / dst_element_size);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", src);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", idx);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src2", node->src[2]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {src, idx, node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_f16_cont_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CONT ||
        !node->src[0] || node->src[0]->op != GGML_OP_PERMUTE ||
        !node->src[0]->src[0] || node->src[0]->src[0]->op != GGML_OP_MUL_MAT ||
        !node->src[0]->src[0]->src[0] || !node->src[0]->src[0]->src[1] ||
        node->type != GGML_TYPE_F32 || node->view_src != nullptr ||
        !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * permute = node->src[0];
    const ggml_tensor * mul_mat = permute->src[0];
    if (ggml_get_op_params_i32(permute, 0) != 0 ||
        ggml_get_op_params_i32(permute, 1) != 2 ||
        ggml_get_op_params_i32(permute, 2) != 1 ||
        ggml_get_op_params_i32(permute, 3) != 3 ||
        !ggml_backend_hrx_make_mul_mat_problem(device_context, mul_mat, &out_request->problem)) {
        return false;
    }
    const int64_t rows = mul_mat->src[0]->ne[1];
    const int64_t cols = mul_mat->src[1]->ne[1];
    if (mul_mat->src[0]->type != GGML_TYPE_F16 ||
        mul_mat->src[1]->type != GGML_TYPE_F32 ||
        mul_mat->type != GGML_TYPE_F32 ||
        permute->ne[0] != mul_mat->ne[0] ||
        permute->ne[1] != mul_mat->ne[2] ||
        permute->ne[2] != mul_mat->ne[1] ||
        permute->ne[3] != mul_mat->ne[3] ||
        node->ne[0] != rows * mul_mat->ne[2] ||
        node->ne[1] != cols ||
        node->ne[2] != mul_mat->ne[3] ||
        node->ne[3] != 1) {
        return false;
    }

    out_request->problem.op = "CONT";
    out_request->problem.supports["layout"] = "decode_kqv_permute_contiguous_noalias";
    out_request->tensors = {mul_mat->src[0], mul_mat->src[1], node};
    out_request->constants.clear();
    if (!ggml_backend_hrx_request_matches_loaded_route(device_context, *out_request, "mul_mat_f16_f32_batched_cont")) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx_make_softmax_kqv_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CONT ||
        !node->src[0] || node->src[0]->op != GGML_OP_PERMUTE ||
        !node->src[0]->src[0] || node->src[0]->src[0]->op != GGML_OP_MUL_MAT ||
        !node->src[0]->src[0]->src[0] || !node->src[0]->src[0]->src[1] ||
        node->src[0]->src[0]->src[1]->op != GGML_OP_SOFT_MAX ||
        !node->src[0]->src[0]->src[1]->src[0] ||
        !node->src[0]->src[0]->src[1]->src[1] ||
        node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * permute = node->src[0];
    const ggml_tensor * kqv = permute->src[0];
    const ggml_tensor * softmax = kqv->src[1];
    const ggml_tensor * kq = softmax->src[0];
    const ggml_tensor * mask = softmax->src[1];
    const ggml_tensor * v = kqv->src[0];
    if (ggml_get_op_params_i32(permute, 0) != 0 ||
        ggml_get_op_params_i32(permute, 1) != 2 ||
        ggml_get_op_params_i32(permute, 2) != 1 ||
        ggml_get_op_params_i32(permute, 3) != 3 ||
        kq->type != GGML_TYPE_F32 ||
        mask->type != GGML_TYPE_F32 ||
        softmax->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F16 ||
        kqv->type != GGML_TYPE_F32 ||
        kqv->src[1] != softmax ||
        !ggml_is_contiguous(kq) ||
        !ggml_is_contiguous(mask) ||
        !ggml_is_contiguous(v)) {
        return false;
    }
    const int64_t kv = kq->ne[0];
    const int64_t n = kq->ne[1];
    const int64_t nheads = kq->ne[2];
    const int64_t d = v->ne[1];
    if (kq->ne[3] != 1 ||
        mask->ne[0] != kv || mask->ne[1] != n || mask->ne[2] != nheads || mask->ne[3] != 1 ||
        v->ne[0] != kv ||
        kqv->ne[0] != d || kqv->ne[1] != n || kqv->ne[2] != nheads || kqv->ne[3] != 1 ||
        node->ne[0] != d * nheads || node->ne[1] != n || node->ne[2] != 1 || node->ne[3] != 1) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    request.problem.op = "CONT";
    request.problem.target_key = device_context->architecture;
    request.problem.supports = {
        {"kq_type", ggml_backend_hrx_catalog_type_name(kq->type)},
        {"mask_type", ggml_backend_hrx_catalog_type_name(mask->type)},
        {"v_type", ggml_backend_hrx_catalog_type_name(v->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "phi4_decode_kq_softmax_kqv_permute_cont"},
        {"fusion", "SOFT_MAX + MUL_MAT(F16xF32) + PERMUTE + CONT"},
    };
    request.problem.shape = {
        {"k", kv},
        {"rows", n},
        {"cols", nheads},
    };
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", kq);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", mask);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src2", v);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
    request.tensors = {kq, mask, v, node};
    request.constants.clear();
    float scale = 1.0f;
    std::memcpy(&scale, softmax->op_params, sizeof(scale));
    ggml_backend_hrx_append_constant<float>(&request.constants, scale);
    if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "softmax_kqv_f32_f16")) {
        return false;
    }
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_cont_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (ggml_backend_hrx_make_softmax_kqv_request(device_context, node, out_request)) {
        return true;
    }
    if (ggml_backend_hrx_make_mul_mat_f16_cont_request(device_context, node, out_request)) {
        return true;
    }
    if (!device_context || !out_request || !node || node->op != GGML_OP_CONT || !node->src[0] ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node) ||
        // The generic Loom CONT kernel indexes i0 directly across node->ne[0].
        // Flattening layouts such as attention PERMUTE->CONT need a dedicated route.
        node->src[0]->ne[0] != node->ne[0]) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "CONT";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "row_contiguous_src_to_contiguous_dst"},
    };
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "ne1", node->ne[1]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "ne2", node->ne[2]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "src_nb1", node->src[0]->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "src_nb2", node->src[0]->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "cont", "src_nb3", node->src[0]->nb[3] / sizeof(float));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_cpy_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_CPY || !node->src[0] ||
        !ggml_is_contiguous(node->src[0]) || !ggml_is_contiguous(node)) {
        return false;
    }
    if (node->src[0]->type == GGML_TYPE_F32 && node->type == GGML_TYPE_Q8_1) {
        out_request->problem = {};
        out_request->problem.op = "QUANTIZE";
        out_request->problem.target_key = device_context->architecture;
        out_request->problem.supports = {
            {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
            {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
            {"layout", "contiguous"},
        };
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "q8_1", "blocks", node->src[0]->ne[0] / ggml_blck_size(GGML_TYPE_Q8_1));
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "q8_1", "ne1", node->src[0]->ne[1]);
        ggml_backend_hrx_set_shape_alias(&out_request->problem, "q8_1", "z_count", node->src[0]->ne[2] * node->src[0]->ne[3]);
        out_request->tensors = {node->src[0], node};
        out_request->constants.clear();
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->ne[0]));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->nb[1] / sizeof(float)));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->nb[2] / sizeof(float)));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->src[0]->nb[3] / sizeof(float)));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->ne[0]));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->ne[1]));
        ggml_backend_hrx_append_constant<int32_t>(&out_request->constants, static_cast<int32_t>(node->ne[2]));
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
        return true;
    }
    out_request->problem = {};
    out_request->problem.op = "CPY";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous_src_to_contiguous_dst"},
    };
    out_request->problem.shape["ncols"] = ggml_nelements(node);
    out_request->problem.shape["nrows"] = 1;
    out_request->problem.shape["copy.n"] = ggml_nelements(node);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = {node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_rope_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_ROPE ||
        !node->src[0] || !node->src[1] || node->src[0]->type != GGML_TYPE_F32 ||
        node->src[1]->type != GGML_TYPE_I32 || node->type != GGML_TYPE_F32 ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_is_contiguous(node->src[1]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    const ggml_tensor * freq = node->src[2];
    if (freq && (freq->type != GGML_TYPE_F32 || !ggml_is_contiguous(freq))) {
        return false;
    }
    int32_t n_dims = 0;
    int32_t mode = 0;
    float freq_base = 10000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    std::memcpy(&n_dims, reinterpret_cast<const uint8_t *>(node->op_params) + sizeof(int32_t), sizeof(int32_t));
    std::memcpy(&mode, reinterpret_cast<const uint8_t *>(node->op_params) + 2 * sizeof(int32_t), sizeof(int32_t));
    std::memcpy(&freq_base, reinterpret_cast<const uint8_t *>(node->op_params) + 5 * sizeof(float), sizeof(float));
    std::memcpy(&freq_scale, reinterpret_cast<const uint8_t *>(node->op_params) + 6 * sizeof(float), sizeof(float));
    std::memcpy(&ext_factor, reinterpret_cast<const uint8_t *>(node->op_params) + 7 * sizeof(float), sizeof(float));
    std::memcpy(&attn_factor, reinterpret_cast<const uint8_t *>(node->op_params) + 8 * sizeof(float), sizeof(float));
    if (ext_factor != 0.0f || (mode & ~(GGML_ROPE_TYPE_NEOX)) != 0) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = freq_scale == 1.0f ? "ROPE" : "ROPE_SCALE";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(node->src[1]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"mode", (mode & GGML_ROPE_TYPE_NEOX) ? "NEOX" : "NORMAL"},
        {"freq_factor", freq ? "src2" : "none"},
        {"ext_factor", "0"},
        {"layout", "element-strided-head-token-ne3-1"},
    };
    if (freq) {
        out_request->problem.supports["src2_type"] = ggml_backend_hrx_catalog_type_name(freq->type);
    }
    const int64_t ntokens = node->src[0]->ne[2];
    const int64_t nheads = node->src[0]->ne[1];
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "ncols", node->src[0]->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "n_dims", n_dims);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "nheads", nheads);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "ntokens", ntokens);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "src0_head_stride", node->src[0]->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "src0_token_stride", node->src[0]->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "dst_head_stride", node->nb[1] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "dst_token_stride", node->nb[2] / sizeof(float));
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "rope", "pos_token_stride", node->src[1]->nb[0] / sizeof(int32_t));
    out_request->problem.shape["ncols"] = node->src[0]->ne[0];
    out_request->problem.shape["nrows"] = nheads * ntokens;
    out_request->problem.shape["n_dims"] = n_dims;
    out_request->problem.shape["rows"] = nheads;
    out_request->problem.shape["cols"] = ntokens;
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    if (freq) {
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src2", freq);
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = freq ?
        std::vector<const ggml_tensor *>{node->src[0], node->src[1], freq, node} :
        std::vector<const ggml_tensor *>{node->src[0], node->src[1], node};
    out_request->constants.clear();
    // Loom ROPE kernels compute pow(theta_scale, pair), so pass the geometric base.
    ggml_backend_hrx_append_constant<float>(&out_request->constants, std::pow(freq_base, -2.0f / static_cast<float>(n_dims)));
    ggml_backend_hrx_append_constant<float>(&out_request->constants, freq_scale);
    ggml_backend_hrx_append_constant<float>(&out_request->constants, attn_factor);
    if (out_request->problem.op == "ROPE_SCALE") {
        ggml_backend_hrx_append_constant<float>(&out_request->constants, 1.0f);
    }
    return true;
}

static bool ggml_backend_hrx_make_mul_mat_q4_k_swiglu_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !out_request || !node || node->op != GGML_OP_GLU ||
        ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU ||
        !node->src[0] || !node->src[1] ||
        node->src[0]->op != GGML_OP_MUL_MAT ||
        node->src[1]->op != GGML_OP_MUL_MAT ||
        !node->src[0]->src[0] || !node->src[0]->src[1] ||
        !node->src[1]->src[0] || !node->src[1]->src[1] ||
        node->src[0]->src[1] != node->src[1]->src[1] ||
        node->src[0]->src[0]->type != GGML_TYPE_Q4_K ||
        node->src[1]->src[0]->type != GGML_TYPE_Q4_K ||
        node->src[0]->src[1]->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(node->src[0], node->src[1]) ||
        !ggml_are_same_shape(node->src[0], node) ||
        !ggml_is_contiguous(node->src[0]->src[0]) ||
        !ggml_is_contiguous(node->src[1]->src[0]) ||
        !ggml_is_contiguous(node->src[0]->src[1]) ||
        !ggml_is_contiguous(node)) {
        return false;
    }

    const ggml_tensor * x = node->src[0];
    const ggml_tensor * gate = node->src[1];
    const ggml_tensor * rhs = x->src[1];
    const int64_t k = x->src[0]->ne[0];
    const int64_t rows = x->src[0]->ne[1];
    const int64_t cols = rhs->ne[1];
    if (gate->src[0]->ne[0] != k || gate->src[0]->ne[1] != rows ||
        node->ne[0] != rows || ggml_backend_hrx_tensor_row_count(node) != cols) {
        return false;
    }

    ggml_backend_hrx_dispatch_request request = {};
    request.problem.op = "GLU";
    request.problem.target_key = device_context->architecture;
    request.problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(x->src[0]->type)},
        {"src1_type", ggml_backend_hrx_catalog_type_name(gate->src[0]->type)},
        {"rhs_type", ggml_backend_hrx_catalog_type_name(rhs->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", "contiguous"},
        {"fusion", "MUL_MAT_Q4_K + MUL_MAT_Q4_K + SWIGLU"},
    };
    request.problem.shape = {
        {"k", k},
        {"rows", rows},
        {"cols", cols},
    };
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src0", x->src[0]);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "src1", rhs);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "gate", gate->src[0]);
    ggml_backend_hrx_add_tensor_facts(&request.problem, "dst", node);
    request.tensors = {x->src[0], gate->src[0], rhs, node};
    request.constants.clear();
    if (!ggml_backend_hrx_request_matches_loaded_route(device_context, request, "mul_mat_q4_k_swiglu_f32")) {
        return false;
    }
    *out_request = std::move(request);
    return true;
}

static bool ggml_backend_hrx_make_glu_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (ggml_backend_hrx_make_mul_mat_q4_k_swiglu_request(device_context, node, out_request)) {
        return true;
    }
    if (!device_context || !out_request || !node || node->op != GGML_OP_GLU || !node->src[0] ||
        ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU ||
        !ggml_backend_hrx_is_f32_row_contiguous(node->src[0]) ||
        !ggml_backend_hrx_is_f32_row_contiguous(node)) {
        return false;
    }
    const bool split = node->src[1] != nullptr;
    if (split && !ggml_backend_hrx_is_f32_row_contiguous(node->src[1])) {
        return false;
    }
    out_request->problem = {};
    out_request->problem.op = "GLU";
    out_request->problem.target_key = device_context->architecture;
    out_request->problem.supports = {
        {"src0_type", ggml_backend_hrx_catalog_type_name(node->src[0]->type)},
        {"dst_type", ggml_backend_hrx_catalog_type_name(node->type)},
        {"layout", split ? "contiguous_split_swiglu" : "packed_contiguous"},
        {"glu_op", "SWIGLU"},
    };
    if (split) {
        out_request->problem.supports["src1_type"] = ggml_backend_hrx_catalog_type_name(node->src[1]->type);
    } else {
        out_request->problem.supports["swapped"] = "false";
    }
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "swiglu", "ncols", node->ne[0]);
    ggml_backend_hrx_set_shape_alias(&out_request->problem, "swiglu", "nrows", ggml_backend_hrx_tensor_row_count(node));
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src0", node->src[0]);
    if (split) {
        ggml_backend_hrx_add_tensor_facts(&out_request->problem, "src1", node->src[1]);
    }
    ggml_backend_hrx_add_tensor_facts(&out_request->problem, "dst", node);
    out_request->tensors = split ? std::vector<const ggml_tensor *>{node->src[0], node->src[1], node} :
        std::vector<const ggml_tensor *>{node->src[0], node};
    out_request->constants.clear();
    return true;
}

static bool ggml_backend_hrx_make_dispatch_request(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node,
        ggml_backend_hrx_dispatch_request * out_request) {
    if (!device_context || !node || !out_request) {
        return false;
    }
    *out_request = {};
    switch (node->op) {
        case GGML_OP_ADD:
            return ggml_backend_hrx_make_add_request(device_context, node, out_request);
        case GGML_OP_MUL:
            return ggml_backend_hrx_make_mul_request(device_context, node, out_request);
        case GGML_OP_DIV:
            return ggml_backend_hrx_make_div_request(device_context, node, out_request);
        case GGML_OP_MUL_MAT:
            return ggml_backend_hrx_make_mul_mat_request(device_context, node, out_request);
        case GGML_OP_MUL_MAT_ID:
            return ggml_backend_hrx_make_mul_mat_id_request(device_context, node, out_request);
        case GGML_OP_SCALE:
            return ggml_backend_hrx_make_scale_request(device_context, node, out_request);
        case GGML_OP_CLAMP:
            return ggml_backend_hrx_make_clamp_request(device_context, node, out_request);
        case GGML_OP_RMS_NORM:
            return ggml_backend_hrx_make_rms_norm_request(device_context, node, out_request);
        case GGML_OP_SUM_ROWS:
            return ggml_backend_hrx_make_sum_rows_request(device_context, node, out_request);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_hrx_make_soft_max_request(device_context, node, out_request);
        case GGML_OP_ARGSORT:
            return ggml_backend_hrx_make_argsort_request(device_context, node, out_request);
        case GGML_OP_GET_ROWS:
            return ggml_backend_hrx_make_get_rows_request(device_context, node, out_request);
        case GGML_OP_SET_ROWS:
            return ggml_backend_hrx_make_set_rows_request(device_context, node, out_request);
        case GGML_OP_CONT:
            return ggml_backend_hrx_make_cont_request(device_context, node, out_request);
        case GGML_OP_CPY:
            return ggml_backend_hrx_make_cpy_request(device_context, node, out_request);
        case GGML_OP_ROPE:
            return ggml_backend_hrx_make_rope_request(device_context, node, out_request);
        case GGML_OP_GLU:
            return ggml_backend_hrx_make_glu_request(device_context, node, out_request);
        default:
            return false;
    }
}

static std::string ggml_backend_hrx_compiled_route_key(
        const ggml_backend_hrx_catalog_route & route,
        const std::vector<ggml_backend_hrx_catalog_binding> & bindings,
        const std::vector<int64_t> & workload_arguments,
        const std::string & target_key) {
    std::string key = target_key;
    key += "|";
    key += route.id;
    key += "|";
    key += route.artifact_id;
    key += "|";
    key += route.root_symbol;
    for (const auto & binding : bindings) {
        key += "|";
        key += binding.key;
        key += "=";
        key += binding.value;
    }
    for (int64_t value : workload_arguments) {
        key += "|arg=";
        key += std::to_string(value);
    }
    return key;
}

static bool ggml_backend_hrx_resolve_workload_arguments(
        const ggml_backend_hrx_catalog_route & route,
        const ggml_backend_hrx_catalog_problem & problem,
        std::vector<int64_t> * out_arguments,
        std::string * out_error) {
    out_arguments->clear();
    for (const std::string & source : route.workload_argument_sources) {
        static constexpr const char * k_shape_prefix = "shape.";
        std::string shape_key = source;
        if (shape_key.compare(0, 6, k_shape_prefix) == 0) {
            shape_key = shape_key.substr(6);
        }
        const auto it = problem.shape.find(shape_key);
        if (it == problem.shape.end()) {
            if (out_error) {
                *out_error = "route " + route.id + " workload argument references missing shape value " + source;
            }
            return false;
        }
        out_arguments->push_back(it->second);
    }
    return true;
}

static ggml_backend_hrx_buffer_context * ggml_backend_hrx_tensor_buffer_context(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return nullptr;
    }
    return ggml_backend_hrx_get_buffer_context(buffer);
}

static bool ggml_backend_hrx_make_tensor_binding(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * tensor,
        hrx_buffer_ref_t * out_ref) {
    auto * buffer_context = ggml_backend_hrx_tensor_buffer_context(tensor);
    if (!buffer_context || buffer_context->device_context != device_context || !buffer_context->buffer || !out_ref) {
        return false;
    }
    *out_ref = {
        /* .buffer = */ buffer_context->buffer,
        /* .offset = */ ggml_backend_hrx_tensor_offset(buffer_context, tensor),
        /* .length = */ ggml_nbytes(tensor),
    };
    return true;
}

static ggml_backend_hrx_compiled_route * ggml_backend_hrx_get_compiled_route(
        ggml_backend_hrx_device_context * device_context,
        const ggml_backend_hrx_catalog_route & route,
        const ggml_backend_hrx_catalog_problem & problem,
        const std::vector<ggml_backend_hrx_catalog_binding> & resolved_bindings,
        const std::vector<int64_t> & workload_arguments) {
    if (!device_context || !device_context->jit || !device_context->reg_context || !device_context->reg_context->catalog) {
        return nullptr;
    }
    const std::string cache_key = ggml_backend_hrx_compiled_route_key(route, resolved_bindings, workload_arguments, problem.target_key);
    {
        std::lock_guard<std::mutex> lock(device_context->compiled_routes_mutex);
        const auto it = device_context->compiled_routes.find(cache_key);
        if (it != device_context->compiled_routes.end()) {
            ggml_backend_hrx_test_record_jit_cache_hit(route.id);
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "route_jit_cache_hit"},
                {"device", device_context->name},
                {"route_id", route.id},
                {"artifact_id", route.artifact_id},
                {"launch_workload_argument_count", it->second->launch_config.workload_argument_count},
                {"workgroup_count", {
                    it->second->launch_config.workgroup_count[0],
                    it->second->launch_config.workgroup_count[1],
                    it->second->launch_config.workgroup_count[2],
                }},
                {"workgroup_size", {
                    it->second->launch_config.workgroup_size[0],
                    it->second->launch_config.workgroup_size[1],
                    it->second->launch_config.workgroup_size[2],
                }},
            });
            return it->second.get();
        }
    }

    const auto * artifact = ggml_backend_hrx_catalog_find_artifact(*device_context->reg_context->catalog, route.artifact_id);
    if (!artifact || artifact->data.empty()) {
        GGML_LOG_ERROR("%s: route %s references missing artifact %s\n", __func__, route.id.c_str(), route.artifact_id.c_str());
        return nullptr;
    }

    std::vector<ggml_hrx_loom_jit_config_binding_t> jit_bindings;
    jit_bindings.reserve(resolved_bindings.size());
    for (const auto & binding : resolved_bindings) {
        jit_bindings.push_back({
            /* .key = */ binding.key.c_str(),
            /* .value = */ binding.value.c_str(),
        });
    }
    const std::string module_name = "ggml_hrx_" + route.id;
    ggml_hrx_loom_jit_compile_options_t compile_options = {
        /* .structure_size = */ sizeof(ggml_hrx_loom_jit_compile_options_t),
        /* .source_data = */ artifact->data.data(),
        /* .source_size = */ artifact->data.size(),
        /* .source_format = */ GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE,
        /* .source_identifier = */ artifact->path.c_str(),
        /* .root_symbol = */ route.root_symbol.c_str(),
        /* .module_name = */ module_name.c_str(),
        /* .artifact_identifier = */ route.id.c_str(),
        /* .config_bindings = */ jit_bindings.data(),
        /* .config_binding_count = */ jit_bindings.size(),
        /* .workload_arguments = */ workload_arguments.empty() ? nullptr : workload_arguments.data(),
        /* .workload_argument_count = */ workload_arguments.size(),
    };
    ggml_hrx_loom_jit_compile_result_t compile_result = {};
    if (!GGML_HRX_CHECK(ggml_hrx_loom_jit_amdgpu_compile(device_context->jit, &compile_options, &compile_result))) {
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    ggml_backend_hrx_write_evidence_file(
        device_context, route.id + ".hsaco", compile_result.hsaco_data, compile_result.hsaco_size);
    ggml_backend_hrx_write_evidence_file(
        device_context, route.id + ".compile-report.json",
        compile_result.compile_report_json, compile_result.compile_report_json_size);

    hrx_executable_t executable = nullptr;
    if (!GGML_HRX_CHECK(hrx_executable_load_data(
            device_context->device,
            compile_result.hsaco_data,
            compile_result.hsaco_size,
            device_context->architecture.c_str(),
            &executable))) {
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    uint32_t export_ordinal = 0;
    const char * export_name = route.export_name.empty() ? nullptr : route.export_name.c_str();
    if (!export_name || !GGML_HRX_CHECK(hrx_executable_lookup_export_by_name(executable, export_name, &export_ordinal))) {
        GGML_LOG_ERROR("%s: failed to resolve export %s for route %s\n", __func__, export_name ? export_name : "<empty>", route.id.c_str());
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    hrx_executable_export_info_t export_info = {};
    if (!GGML_HRX_CHECK(hrx_executable_export_info(executable, export_ordinal, &export_info))) {
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    if (export_info.binding_count != route.binding_count ||
        export_info.parameter_count != route.parameter_count ||
        export_info.constant_byte_length != route.constant_byte_length) {
        GGML_LOG_ERROR(
            "%s: route %s JIT export ABI mismatch "
            "(bindings=%u expected=%u constants_size=%u expected_constants_size=%u "
            "parameters=%u expected_parameters=%u)\n",
            __func__,
            route.id.c_str(),
            export_info.binding_count,
            route.binding_count,
            export_info.constant_byte_length,
            route.constant_byte_length,
            export_info.parameter_count,
            route.parameter_count);
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }

    ggml_backend_hrx_compiled_route_ptr compiled(new (std::nothrow) ggml_backend_hrx_compiled_route());
    if (!compiled) {
        hrx_executable_release(executable);
        ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);
        return nullptr;
    }
    compiled->route = &route;
    compiled->executable = executable;
    compiled->export_ordinal = export_ordinal;
    compiled->export_info = export_info;
    compiled->launch_config = compile_result.launch_config;

    ggml_backend_hrx_test_record_jit_compile(route.id);
    ggml_backend_hrx_trace_event(device_context->reg_context, {
        {"event", "route_jit_compiled"},
        {"device", device_context->name},
        {"route_id", route.id},
        {"artifact_id", route.artifact_id},
        {"root_symbol", route.root_symbol},
        {"launch_workload_argument_count", compile_result.launch_config.workload_argument_count},
        {"binding_count", export_info.binding_count},
        {"parameter_count", export_info.parameter_count},
        {"constant_byte_length", export_info.constant_byte_length},
        {"workgroup_count", {
            compile_result.launch_config.workgroup_count[0],
            compile_result.launch_config.workgroup_count[1],
            compile_result.launch_config.workgroup_count[2],
        }},
        {"workgroup_size", {
            compile_result.launch_config.workgroup_size[0],
            compile_result.launch_config.workgroup_size[1],
            compile_result.launch_config.workgroup_size[2],
        }},
    });
    ggml_hrx_loom_jit_compile_result_deinitialize(&compile_result);

    std::lock_guard<std::mutex> lock(device_context->compiled_routes_mutex);
    auto inserted = device_context->compiled_routes.emplace(cache_key, std::move(compiled));
    if (!inserted.second) {
        return inserted.first->second.get();
    }
    return inserted.first->second.get();
}

static bool ggml_backend_hrx_dispatch_node(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * node) {
    ggml_backend_hrx_dispatch_request request;
    if (!ggml_backend_hrx_make_dispatch_request(device_context, node, &request) ||
        !device_context->reg_context || !device_context->reg_context->catalog) {
        return false;
    }
    const auto * route = ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, request.problem);
    if (!route) {
        return false;
    }
    if (route->binding_count != request.tensors.size() || route->constant_byte_length != request.constants.size()) {
        ggml_backend_hrx_trace_event(device_context->reg_context, {
            {"event", "route_rejected"},
            {"reason", "request_abi_mismatch"},
            {"route_id", route->id},
            {"binding_count", route->binding_count},
            {"request_binding_count", request.tensors.size()},
            {"constant_byte_length", route->constant_byte_length},
            {"request_constant_byte_length", request.constants.size()},
        });
        return false;
    }

    std::vector<ggml_backend_hrx_catalog_binding> resolved_bindings;
    std::string binding_error;
    if (!ggml_backend_hrx_catalog_make_config_bindings(*route, request.problem, &resolved_bindings, &binding_error)) {
        GGML_LOG_ERROR("%s: %s\n", __func__, binding_error.c_str());
        return false;
    }
    std::vector<int64_t> workload_arguments;
    std::string workload_error;
    if (!ggml_backend_hrx_resolve_workload_arguments(*route, request.problem, &workload_arguments, &workload_error)) {
        GGML_LOG_ERROR("%s: %s\n", __func__, workload_error.c_str());
        return false;
    }
    auto * compiled = ggml_backend_hrx_get_compiled_route(device_context, *route, request.problem, resolved_bindings, workload_arguments);
    if (!compiled || !compiled->executable) {
        return false;
    }

    std::vector<hrx_buffer_ref_t> bindings(request.tensors.size());
    for (size_t i = 0; i < request.tensors.size(); ++i) {
        if (!ggml_backend_hrx_make_tensor_binding(device_context, request.tensors[i], &bindings[i])) {
            return false;
        }
    }

    hrx_dispatch_config_t dispatch_config = {
        /* .workgroup_count = */ {
            compiled->launch_config.workgroup_count[0],
            compiled->launch_config.workgroup_count[1],
            compiled->launch_config.workgroup_count[2],
        },
        /* .workgroup_size = */ {
            compiled->launch_config.workgroup_size[0],
            compiled->launch_config.workgroup_size[1],
            compiled->launch_config.workgroup_size[2],
        },
        /* .subgroup_size = */ compiled->launch_config.subgroup_size,
    };
    ggml_backend_hrx_trace_event(device_context->reg_context, {
        {"event", "route_dispatch"},
        {"device", device_context->name},
        {"route_id", route->id},
        {"shape", request.problem.shape},
        {"launch_workload_argument_count", compiled->launch_config.workload_argument_count},
        {"workgroup_count", {
            dispatch_config.workgroup_count[0],
            dispatch_config.workgroup_count[1],
            dispatch_config.workgroup_count[2],
        }},
        {"workgroup_size", {
            dispatch_config.workgroup_size[0],
            dispatch_config.workgroup_size[1],
            dispatch_config.workgroup_size[2],
        }},
    });
    ggml_backend_hrx_test_record_dispatch(route->id);
    return GGML_HRX_CHECK(hrx_stream_dispatch(
        device_context->active_stream,
        compiled->executable,
        compiled->export_ordinal,
        &dispatch_config,
        request.constants.empty() ? nullptr : request.constants.data(),
        request.constants.size(),
        bindings.data(),
        bindings.size(),
        HRX_DISPATCH_FLAG_NONE));
}

static bool ggml_backend_hrx_is_fused_producer_node(
        ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        int node_index) {
    if (!device_context || !cgraph || node_index < 0 || node_index >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * producer = cgraph->nodes[node_index];
    if (!producer ||
        (producer->op != GGML_OP_RMS_NORM &&
         producer->op != GGML_OP_ADD &&
         producer->op != GGML_OP_CONT &&
         producer->op != GGML_OP_ROPE &&
         producer->op != GGML_OP_MUL_MAT &&
         producer->op != GGML_OP_SOFT_MAX)) {
        return false;
    }
    for (int i = node_index + 1; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * consumer = cgraph->nodes[i];
        if (!consumer || !consumer->src[0]) {
            continue;
        }
        bool producer_consumed = false;
        const char * expected_family = nullptr;
        if (consumer->op == GGML_OP_MUL) {
            const bool rms_norm_mul_producer =
                producer->op == GGML_OP_RMS_NORM && consumer->src[0] == producer;
            const bool add_rms_norm_mul_producer =
                consumer->src[0]->op == GGML_OP_RMS_NORM &&
                consumer->src[0]->src[0] &&
                ((producer->op == GGML_OP_RMS_NORM && consumer->src[0] == producer) ||
                 (producer->op == GGML_OP_ADD && consumer->src[0]->src[0] == producer));
            producer_consumed = rms_norm_mul_producer || add_rms_norm_mul_producer;
            expected_family = add_rms_norm_mul_producer ? "add_rms_norm_mul_f32" :
                (rms_norm_mul_producer ? "rms_norm_mul_f32" : nullptr);
        } else if (consumer->op == GGML_OP_SET_ROWS) {
            if (producer->op == GGML_OP_CONT &&
                ggml_backend_hrx_zero_offset_source_chain_target(consumer->src[0], GGML_OP_CONT) == producer) {
                producer_consumed = true;
                expected_family = "cont_set_rows_f32";
            } else if (producer->op == GGML_OP_ROPE &&
                       ggml_backend_hrx_zero_offset_source_chain_target(consumer->src[0], GGML_OP_ROPE) == producer) {
                producer_consumed = true;
                expected_family = "rope_set_rows_f32";
            }
        } else if (consumer->op == GGML_OP_GLU) {
            if (producer->op == GGML_OP_MUL_MAT && (consumer->src[0] == producer || consumer->src[1] == producer)) {
                producer_consumed = true;
                expected_family = "mul_mat_q4_k_swiglu_f32";
            }
        } else if (consumer->op == GGML_OP_CONT) {
            if (producer->op == GGML_OP_MUL_MAT &&
                consumer->src[0] &&
                consumer->src[0]->op == GGML_OP_PERMUTE &&
                consumer->src[0]->src[0] == producer) {
                producer_consumed = true;
                expected_family = (producer->src[1] && producer->src[1]->op == GGML_OP_SOFT_MAX) ?
                    "softmax_kqv_f32_f16" : "mul_mat_f16_f32_batched_cont";
            } else if (producer->op == GGML_OP_SOFT_MAX &&
                       consumer->src[0] &&
                       consumer->src[0]->op == GGML_OP_PERMUTE &&
                       consumer->src[0]->src[0] &&
                       consumer->src[0]->src[0]->op == GGML_OP_MUL_MAT &&
                       consumer->src[0]->src[0]->src[1] == producer) {
                producer_consumed = true;
                expected_family = "softmax_kqv_f32_f16";
            }
        }
        if (!producer_consumed || !expected_family) {
            continue;
        }
        ggml_backend_hrx_dispatch_request request = {};
        if (!ggml_backend_hrx_make_dispatch_request(device_context, consumer, &request) ||
            !device_context->reg_context || !device_context->reg_context->catalog) {
            continue;
        }
        const auto * route = ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, request.problem);
        if (route && route->family == expected_family) {
            ggml_backend_hrx_trace_event(device_context->reg_context, {
                {"event", "fused_producer_skipped"},
                {"producer_op", ggml_op_desc(producer)},
                {"consumer_op", ggml_op_desc(consumer)},
                {"route_id", route->id},
            });
            return true;
        }
    }
    return false;
}

static enum ggml_status ggml_backend_hrx_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->device_context->options && context->device_context->options->trace_graph) {
        ggml_backend_hrx_trace_event(context->device_context->reg_context, {
            {"event", "graph_compute_begin"},
            {"device", context->device_context->name},
            {"node_count", cgraph ? cgraph->n_nodes : 0},
        });
    }
    if (!ggml_backend_hrx_sync_graph_entry_streams(context->device_context, context->stream)) {
        return GGML_STATUS_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        context->device_context->active_stream = context->stream;
    }

    for (int i = 0; cgraph && i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!ggml_backend_hrx_is_metadata_op(node)) {
            if (ggml_backend_hrx_is_fused_producer_node(context->device_context, cgraph, i)) {
                continue;
            }
            if (ggml_backend_hrx_dispatch_node(context->device_context, node)) {
                continue;
            }
            if (context->device_context->options && context->device_context->options->trace_graph) {
                ggml_backend_hrx_trace_event(context->device_context->reg_context, {
                    {"event", "unsupported_compute_node"},
                    {"device", context->device_context->name},
                    {"op", ggml_op_desc(node)},
                    {"node", ggml_get_name(node)},
                });
            }
            GGML_LOG_ERROR(
                "%s: HRX3 backend has no matching compute route; unsupported op %s node=%s\n",
                __func__, ggml_op_desc(node), ggml_get_name(node));
            return GGML_STATUS_FAILED;
        }
    }

    ggml_backend_hrx_synchronize(backend);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_hrx_i = {
    /* .get_name           = */ ggml_backend_hrx_get_name,
    /* .free               = */ ggml_backend_hrx_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ ggml_backend_hrx_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_hrx_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static const char * ggml_backend_hrx_device_get_name(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->name.c_str();
}

static const char * ggml_backend_hrx_device_get_description(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->description.c_str();
}

static void ggml_backend_hrx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = ggml_backend_hrx_get_device_context(dev);
    *free = context->memory_total;
    *total = context->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_hrx_device_get_name(dev);
    props->description = ggml_backend_hrx_device_get_description(dev);
    props->type = ggml_backend_hrx_device_get_type(dev);
    ggml_backend_hrx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async = */ true,
        /* .host_buffer = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events = */ false,
    };
}

static ggml_backend_t ggml_backend_hrx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    hrx_stream_t stream = nullptr;
    if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &stream))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx_context {
        /* .device_context = */ device_context,
        /* .stream         = */ stream,
        /* .name           = */ device_context->name,
    };
    if (!context) {
        hrx_stream_release(stream);
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_hrx_guid(),
        /* .iface   = */ ggml_backend_hrx_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    if (!backend) {
        hrx_stream_release(stream);
        delete context;
        return nullptr;
    }

    ggml_backend_hrx_register_stream(device_context, stream);
    return backend;
}

static bool ggml_backend_hrx_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (ggml_backend_hrx_is_metadata_op(op)) {
        return true;
    }
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    ggml_backend_hrx_dispatch_request request;
    if (!ggml_backend_hrx_make_dispatch_request(device_context, op, &request)) {
        ggml_backend_hrx_trace_event(device_context->reg_context, {
            {"event", "supports_op_rejected"},
            {"reason", "request_builder"},
            {"op", ggml_op_desc(op)},
            {"node", ggml_get_name(op)},
        });
        return false;
    }
    const bool supported =
        device_context->reg_context &&
        device_context->reg_context->catalog &&
        ggml_backend_hrx_catalog_find_route(*device_context->reg_context->catalog, request.problem) != nullptr;
    if (!supported) {
        ggml_backend_hrx_trace_event(device_context->reg_context, {
            {"event", "supports_op_rejected"},
            {"reason", "no_catalog_route"},
            {"op", request.problem.op},
            {"supports", request.problem.supports},
            {"shape", request.problem.shape},
        });
    }
    return supported;
}

static bool ggml_backend_hrx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_hrx_buffer_type_get_name) {
        return false;
    }
    return buft->device == dev;
}

static const ggml_backend_device_i ggml_backend_hrx_device_i = {
    /* .get_name             = */ ggml_backend_hrx_device_get_name,
    /* .get_description      = */ ggml_backend_hrx_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx_device_get_type,
    /* .get_props            = */ ggml_backend_hrx_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_backend_hrx_reg_context * ggml_backend_hrx_get_reg_context(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_hrx_reg_context *>(reg->context);
}

static const char * ggml_backend_hrx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX_NAME;
}

static size_t ggml_backend_hrx_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_hrx_get_reg_context(reg)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * context = ggml_backend_hrx_get_reg_context(reg);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * ggml_backend_hrx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hrx_reg_i = {
    /* .get_name         = */ ggml_backend_hrx_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hrx_reg_get_proc_address,
};

ggml_backend_hrx_reg_context::~ggml_backend_hrx_reg_context() {
    for (auto & device_context : device_contexts) {
        if (device_context) {
            ggml_backend_hrx_sync_streams(device_context.get());
            std::lock_guard<std::mutex> lock(device_context->compiled_routes_mutex);
            device_context->compiled_routes.clear();
        }
        if (device_context && device_context->transfer_stream) {
            hrx_status_t status = hrx_stream_synchronize(device_context->transfer_stream);
            if (!hrx_status_is_ok(status)) {
                hrx_status_ignore(status);
            }
            ggml_backend_hrx_unregister_stream(device_context.get(), device_context->transfer_stream);
            hrx_stream_release(device_context->transfer_stream);
            device_context->transfer_stream = nullptr;
        }
        if (device_context && device_context->jit) {
            ggml_hrx_loom_jit_amdgpu_release(device_context->jit);
            device_context->jit = nullptr;
        }
        if (device_context && device_context->device) {
            hrx_device_release(device_context->device);
            device_context->device = nullptr;
        }
    }
    if (gpu_initialized) {
        hrx_status_t status = hrx_gpu_shutdown();
        if (!hrx_status_is_ok(status)) {
            hrx_status_ignore(status);
        }
    }
}

static std::unique_ptr<ggml_backend_hrx_reg_context> ggml_backend_hrx_create_reg_context() {
    auto context = std::make_unique<ggml_backend_hrx_reg_context>();
    context->options = ggml_backend_hrx_parse_options();

    if (!context->options.trace_jsonl_path.empty()) {
        context->trace_jsonl.open(context->options.trace_jsonl_path, std::ios::out | std::ios::app);
        if (!context->trace_jsonl) {
            GGML_LOG_ERROR(
                "%s: failed to open GGML_HRX_TRACE_JSONL path %s\n",
                __func__, context->options.trace_jsonl_path.c_str());
        }
    }

    ggml_backend_hrx_trace_event(context.get(), {
        {"event", "backend_init"},
        {"catalog_dir", context->options.catalog_dir},
        {"evidence_dir", context->options.evidence_dir},
        {"trace_routes", context->options.trace_routes},
        {"trace_graph", context->options.trace_graph},
        {"staging_arena_size", context->options.staging_arena_size},
    });

    std::string catalog_error;
    context->catalog = ggml_backend_hrx_load_catalog(
        ggml_backend_hrx_optional_c_str(context->options.catalog_dir),
        &catalog_error);
    if (!context->catalog) {
        GGML_LOG_ERROR("%s: %s\n", __func__, catalog_error.c_str());
        ggml_backend_hrx_trace_event(context.get(), {
            {"event", "catalog_error"},
            {"error", catalog_error},
        });
        return context;
    }
    ggml_backend_hrx_trace_event(context.get(), {
        {"event", "catalog_loaded"},
        {"catalog_id", context->catalog->catalog_id},
        {"source", context->catalog->source},
        {"sources", context->catalog->source_count},
        {"artifacts", context->catalog->artifact_count},
        {"families", context->catalog->family_count},
        {"routes", context->catalog->route_count},
        {"fusions", context->catalog->fusion_count},
    });

    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->gpu_initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }

    int device_count = 0;
    if (!GGML_HRX_CHECK(hrx_gpu_device_count(&device_count)) || device_count <= 0) {
        return context;
    }

    context->device_contexts.reserve(device_count);
    context->devices.reserve(device_count);

    for (int i = 0; i < device_count; ++i) {
        hrx_device_t device = nullptr;
        if (!GGML_HRX_CHECK(hrx_gpu_device_get(i, &device)) || !device) {
            continue;
        }
        hrx_device_retain(device);

        auto device_context = std::make_unique<ggml_backend_hrx_device_context>();
        device_context->reg_context = context.get();
        device_context->options = &context->options;
        device_context->device = device;
        device_context->name = std::string(GGML_HRX_NAME) + std::to_string(i);
        device_context->description = ggml_backend_hrx_device_description(device);
        device_context->architecture = ggml_backend_hrx_device_architecture(device);
        device_context->memory_total = ggml_backend_hrx_total_memory(device);
        ggml_hrx_loom_jit_amdgpu_options_t jit_options = {
            /* .structure_size      = */ sizeof(ggml_hrx_loom_jit_amdgpu_options_t),
            /* .processor           = */ device_context->architecture.c_str(),
            /* .identifier          = */ device_context->name.c_str(),
            /* .sanitizer           = */ ggml_backend_hrx_optional_c_str(context->options.loom_sanitizer),
            /* .sanitizer_reporting = */ ggml_backend_hrx_optional_c_str(context->options.loom_sanitizer_reporting),
        };
        if (!GGML_HRX_CHECK(ggml_hrx_loom_jit_amdgpu_create(&jit_options, &device_context->jit))) {
            device_context->jit = nullptr;
        }
        if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &device_context->transfer_stream))) {
            if (device_context->jit) {
                ggml_hrx_loom_jit_amdgpu_release(device_context->jit);
                device_context->jit = nullptr;
            }
            hrx_device_release(device);
            continue;
        }
        ggml_backend_hrx_register_stream(device_context.get(), device_context->transfer_stream);

        ggml_backend_hrx_trace_event(context.get(), {
            {"event", "device_initialized"},
            {"device", device_context->name},
            {"description", device_context->description},
            {"architecture", device_context->architecture},
            {"memory_total", device_context->memory_total},
            {"jit_available", device_context->jit != nullptr},
        });

        context->device_contexts.emplace_back(std::move(device_context));
        context->devices.push_back({
            /* .iface   = */ ggml_backend_hrx_device_i,
            /* .reg     = */ nullptr,
            /* .context = */ context->device_contexts.back().get(),
        });
    }

    return context;
}

} // namespace

ggml_backend_t ggml_backend_hrx_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid HRX device index %zu\n", __func__, dev_num);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, dev_num), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    return reg ? static_cast<int>(ggml_backend_reg_dev_count(reg)) : 0;
}

void ggml_backend_hrx_get_device_description(int device, char * description, size_t description_size) {
    if (!description || description_size == 0) {
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        description[0] = '\0';
        return;
    }

    const char * value = ggml_backend_dev_description(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)));
    std::snprintf(description, description_size, "%s", value ? value : "");
}

void ggml_backend_hrx_get_device_memory(int device, size_t * free, size_t * total) {
    if (free) {
        *free = 0;
    }
    if (total) {
        *total = 0;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        return;
    }

    ggml_backend_dev_memory(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)), free, total);
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_reg_t ggml_backend_hrx_reg(void) {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context =
        ggml_backend_hrx_create_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_hrx_reg_i,
        /* .context     = */ context.get(),
    };

    if (context) {
        for (auto & device : context->devices) {
            device.reg = &reg;
        }
    }

    return &reg;
}

std::vector<ggml_backend_hrx_test_case> ggml_backend_hrx_test_cases(
        ggml_backend_dev_t dev,
        const std::string & family) {
    std::vector<ggml_backend_hrx_test_case> out;
    if (!dev) {
        return out;
    }
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    if (!device_context || !device_context->reg_context || !device_context->reg_context->catalog) {
        return out;
    }
    const auto & catalog = *device_context->reg_context->catalog;
    std::vector<size_t> case_indices;
    if (family.empty()) {
        for (size_t i = 0; i < catalog.test_cases.size(); ++i) {
            if (catalog.test_cases[i].target_key == device_context->architecture) {
                case_indices.push_back(i);
            }
        }
    } else {
        const auto it = catalog.test_cases_by_target_family.find(
            ggml_backend_hrx_test_case_index_key(device_context->architecture, family));
        if (it == catalog.test_cases_by_target_family.end()) {
            return out;
        }
        case_indices = it->second;
    }
    out.reserve(case_indices.size());
    for (const size_t test_case_index : case_indices) {
        if (test_case_index >= catalog.test_cases.size()) {
            continue;
        }
        const auto & catalog_case = catalog.test_cases[test_case_index];
        ggml_backend_hrx_test_case test_case;
        test_case.id = catalog_case.id;
        test_case.op = catalog_case.op;
        test_case.family = catalog_case.family;
        test_case.scenario = catalog_case.scenario;
        test_case.expected_route_id = catalog_case.expected_route_id;
        test_case.supports = catalog_case.supports;
        test_case.shape = catalog_case.shape;
        test_case.tolerance = catalog_case.tolerance;
        test_case.repeat = catalog_case.repeat;
        out.push_back(std::move(test_case));
    }
    return out;
}

void ggml_backend_hrx_test_reset_dispatch_record(void) {
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    g_ggml_backend_hrx_test_dispatch_recorder.enabled = true;
    g_ggml_backend_hrx_test_dispatch_recorder.routes.clear();
}

ggml_backend_hrx_test_route_record ggml_backend_hrx_test_get_route_record(
        const std::string & route_id) {
    ggml_backend_hrx_test_route_record record;
    if (route_id.empty()) {
        return record;
    }
    std::lock_guard<std::mutex> lock(g_ggml_backend_hrx_test_dispatch_recorder.mutex);
    const auto it = g_ggml_backend_hrx_test_dispatch_recorder.routes.find(route_id);
    if (it == g_ggml_backend_hrx_test_dispatch_recorder.routes.end()) {
        return record;
    }
    return it->second;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
