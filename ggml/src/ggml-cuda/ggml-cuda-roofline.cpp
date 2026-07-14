// Per-op GPU roofline profiling for the HIP/ROCm backend via rocprofiler-sdk.
//
// Attributes each GPU kernel's device-measured duration to the ggml op that launched
// it, using a unique external correlation id per op invocation, and writes a JSON
// report on exit with one row per invocation listing the kernels it dispatched.
// Aggregation (grouping identical ops, averaging, counting) is left to the consumer.
// Activated at runtime by the environment variable GGML_ROOFLINE_OUT=<path.json>;
// every entry point is a no-op unless it is set.
//
// rocprofiler-sdk is loaded with dlopen and configured with rocprofiler_force_configure
// at runtime, and its entry points are resolved with dlsym. It is intentionally not
// linked: its global constructors abort when the library is loaded early as part of
// another shared object.

#include "ggml-cuda-roofline.h"
#include "ggml-impl.h"

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>

#include <dlfcn.h>
#include <cxxabi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int n_op_params = GGML_MAX_OP_PARAMS / (int) sizeof(int32_t);  // 16

// Geometry and memory traffic captured for one distinct ggml op shape.
struct op_record {
    const char * op    = "";                        // op / unary / glu name (ggml_op_desc)
    const char * dtype = "";                        // destination type
    const char * quant = "";                        // src0 type (weight quantization for matmul)
    int64_t      ne[4]                   = {0, 0, 0, 0};  // destination shape
    int64_t      src_ne[GGML_MAX_SRC][4] = {};           // source shapes
    const char * src_types[GGML_MAX_SRC] = {};           // source types (ggml_type_name)
    int          n_src                   = 0;
    int32_t      op_params[n_op_params]  = {};            // op parameters (conv / pool / rope / ... geometry)
    int64_t      bytes                   = 0;             // total HBM traffic: destination + all sources
    int64_t      dst_bytes               = 0;             // ggml_nbytes(destination)
    int64_t      src_bytes[GGML_MAX_SRC] = {};            // ggml_nbytes(source)
    // Two identities per tensor, so the consumer can tell which tensors actually move memory
    // without seeing the graph. A *tensor id* is one ggml_tensor's address (identifies a single
    // tensor; used to dedup a tensor read by several ops). A *storage id* is the address of the
    // tensor that owns the underlying buffer -- ggml's view_src root -- so tensors that alias the
    // same memory (a view and its source; an in-place op's output and the dst it was handed in as
    // a source) share one storage id; that is how the consumer spots internal and in-place tensors.
    uint64_t     dst_sid                 = 0;             // destination's storage id (its buffer)
    uint64_t     src_ids[GGML_MAX_SRC]   = {};            // each source's tensor id (dedup key)
    uint64_t     src_sids[GGML_MAX_SRC]  = {};            // each source's storage id (its buffer)
    int64_t      M = 0, N = 0, K = 0, n_experts = 0, top_k = 0;  // matmul dimensions
    std::vector<op_record> fused_nodes;                  // per-node geometry when this row is a fused group (else empty)
};

// One kernel dispatch: which kernel ran, when it started, and for how long. The start
// time is kept only to order kernels within an op (buffer records arrive unordered).
struct dispatch {
    uint64_t start_ns    = 0;
    uint64_t kernel_id   = 0;
    uint64_t duration_ns = 0;
};

std::mutex                                           g_mutex;
std::unordered_map<uint64_t, op_record>              g_records;      // geometry id -> geometry
std::unordered_map<uint64_t, uint64_t>               g_invocations;  // invocation id -> geometry id
std::unordered_map<uint64_t, std::vector<dispatch>>  g_dispatches;   // invocation id -> dispatches
std::unordered_map<uint64_t, std::string>            g_kernel_names; // kernel id -> demangled symbol
std::atomic<uint64_t>                                g_next_invocation{1};
thread_local uint64_t                                g_current_invocation = 0;  // id pushed by the last begin_op on this thread

bool                     g_active = false;
std::string              g_out_path;
std::string              g_device;                          // GPU architecture, e.g. "gfx1151"
rocprofiler_context_id_t g_context{0};
rocprofiler_buffer_id_t  g_buffer{0};

// rocprofiler-sdk entry points, resolved at runtime via dlsym.
using fn_create_context = rocprofiler_status_t (*)(rocprofiler_context_id_t *);
using fn_create_buffer  = rocprofiler_status_t (*)(rocprofiler_context_id_t, size_t, size_t,
                                                   rocprofiler_buffer_policy_t,
                                                   rocprofiler_buffer_tracing_cb_t, void *,
                                                   rocprofiler_buffer_id_t *);
using fn_config_buffer  = rocprofiler_status_t (*)(rocprofiler_context_id_t,
                                                   rocprofiler_buffer_tracing_kind_t,
                                                   const rocprofiler_tracing_operation_t *,
                                                   size_t, rocprofiler_buffer_id_t);
using fn_config_callback = rocprofiler_status_t (*)(rocprofiler_context_id_t,
                                                    rocprofiler_callback_tracing_kind_t,
                                                    const rocprofiler_tracing_operation_t *,
                                                    size_t, rocprofiler_callback_tracing_cb_t, void *);
using fn_start          = rocprofiler_status_t (*)(rocprofiler_context_id_t);
using fn_flush          = rocprofiler_status_t (*)(rocprofiler_buffer_id_t);
using fn_get_thread_id  = rocprofiler_status_t (*)(rocprofiler_thread_id_t *);
using fn_push_id        = rocprofiler_status_t (*)(rocprofiler_context_id_t, rocprofiler_thread_id_t,
                                                   rocprofiler_user_data_t);
using fn_pop_id         = rocprofiler_status_t (*)(rocprofiler_context_id_t, rocprofiler_thread_id_t,
                                                   rocprofiler_user_data_t *);

fn_create_context  p_create_context  = nullptr;
fn_create_buffer   p_create_buffer   = nullptr;
fn_config_buffer   p_config_buffer   = nullptr;
fn_config_callback p_config_callback = nullptr;
fn_start           p_start           = nullptr;
fn_flush           p_flush           = nullptr;
fn_get_thread_id   p_get_thread_id   = nullptr;
fn_push_id         p_push_id         = nullptr;
fn_pop_id          p_pop_id          = nullptr;

// Resolve the required entry points; returns false if any is missing. The code-object
// callback service is optional (kernel-symbol names) and resolved separately.
bool resolve_symbols() {
    p_create_context  = (fn_create_context)  dlsym(RTLD_DEFAULT, "rocprofiler_create_context");
    p_create_buffer   = (fn_create_buffer)   dlsym(RTLD_DEFAULT, "rocprofiler_create_buffer");
    p_config_buffer   = (fn_config_buffer)   dlsym(RTLD_DEFAULT, "rocprofiler_configure_buffer_tracing_service");
    p_start           = (fn_start)           dlsym(RTLD_DEFAULT, "rocprofiler_start_context");
    p_flush           = (fn_flush)           dlsym(RTLD_DEFAULT, "rocprofiler_flush_buffer");
    p_get_thread_id   = (fn_get_thread_id)   dlsym(RTLD_DEFAULT, "rocprofiler_get_thread_id");
    p_push_id         = (fn_push_id)         dlsym(RTLD_DEFAULT, "rocprofiler_push_external_correlation_id");
    p_pop_id          = (fn_pop_id)          dlsym(RTLD_DEFAULT, "rocprofiler_pop_external_correlation_id");
    p_config_callback = (fn_config_callback) dlsym(RTLD_DEFAULT, "rocprofiler_configure_callback_tracing_service");
    return p_create_context && p_create_buffer && p_config_buffer && p_start && p_flush &&
           p_get_thread_id && p_push_id && p_pop_id;
}

std::string demangle(const char * name) {
    if (!name) return "";
    int    status     = 0;
    char * demangled  = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string result = (status == 0 && demangled) ? demangled : name;
    free(demangled);
    return result;
}

void json_escape(std::ostringstream & out, const std::string & str) {
    for (char c : str) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else                       out << c;
    }
}

uint64_t hash_mix(uint64_t hash, uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

// Storage identity: ggml points a view/in-place tensor's view_src at the buffer it aliases, so
// following the chain to the end yields the tensor that actually owns the HBM. Two tensors touch
// the same memory iff they share this root. VIEW/RESHAPE outputs carry view_src, and so do
// in-place ops (e.g. CPY does ggml_view_tensor(dst)), which lets the consumer tell a read that
// actually targets the op's own output (an in-place destination) from a genuine input read.
const ggml_tensor * roofline_storage(const ggml_tensor * t) {
    while (t && t->view_src) t = t->view_src;
    return t;
}

// Fill a record's geometry and single-node HBM byte fields from one ggml node.
void fill_head_record(op_record & rec, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    rec.op    = ggml_op_desc(node);
    rec.dtype = ggml_type_name(node->type);
    rec.quant = src0 ? ggml_type_name(src0->type) : "";
    for (int d = 0; d < 4; d++) rec.ne[d] = node->ne[d];
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        if (node->src[j]) {
            for (int d = 0; d < 4; d++) rec.src_ne[j][d] = node->src[j]->ne[d];
            rec.src_types[j] = ggml_type_name(node->src[j]->type);
            rec.n_src = j + 1;
        }
    }
    memcpy(rec.op_params, node->op_params, sizeof(rec.op_params));

    // total HBM traffic for this op: write the destination and read every source.
    rec.dst_bytes = (int64_t) ggml_nbytes(node);
    rec.bytes     = rec.dst_bytes;
    rec.dst_sid   = (uint64_t) (uintptr_t) roofline_storage(node);
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        if (node->src[j]) {
            rec.src_bytes[j] = (int64_t) ggml_nbytes(node->src[j]);
            rec.bytes       += rec.src_bytes[j];
            rec.src_ids[j]   = (uint64_t) (uintptr_t) node->src[j];
            rec.src_sids[j]  = (uint64_t) (uintptr_t) roofline_storage(node->src[j]);
        }
    }

    if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && src0 && src1) {
        rec.K = src0->ne[0];
        rec.N = src0->ne[1];
        if (node->op == GGML_OP_MUL_MAT) {
            rec.M = src1->ne[1];
        } else {
            // MUL_MAT_ID (MoE): src0 = experts [K,N,E], src1 = input (often broadcast
            // [K,1,n_tokens]), src2 = ids [n_expert_used, n_tokens]. top_k is the number of
            // experts routed per token = ids->ne[0]; src1->ne[1] is 1 when the input is
            // broadcast, so it must not be used. Fall back to src1->ne[1] only if ids absent.
            rec.M         = src1->ne[2];
            rec.n_experts = src0->ne[2];
            rec.top_k     = node->src[2] ? node->src[2]->ne[0] : src1->ne[1];
        }
    }
}

// Dedup hash of one node's geometry (destination, all sources, op params, types); distinct
// shapes get distinct ids so the report can be deduplicated.
uint64_t head_geometry_id(const op_record & rec, const ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    uint64_t geometry_id = hash_mix(0, (uint64_t) node->op);
    for (int d = 0; d < 4; d++) geometry_id = hash_mix(geometry_id, (uint64_t) rec.ne[d]);
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        for (int d = 0; d < 4; d++) geometry_id = hash_mix(geometry_id, (uint64_t) rec.src_ne[j][d]);
    }
    for (int p = 0; p < n_op_params; p++) geometry_id = hash_mix(geometry_id, (uint64_t) (uint32_t) rec.op_params[p]);
    geometry_id = hash_mix(geometry_id, (uint64_t) (src0 ? src0->type : 0));
    geometry_id = hash_mix(geometry_id, (uint64_t) node->type);
    return geometry_id;
}

// Buffer callback: records each kernel dispatch (kernel id + device time) under the
// invocation that launched it. Invoked asynchronously as the tracing buffer fills.
void buffer_callback(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                     rocprofiler_record_header_t ** headers, size_t n_headers, void *, uint64_t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t i = 0; i < n_headers; i++) {
        rocprofiler_record_header_t * header = headers[i];
        if (header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
            header->kind     == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
            auto * record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(header->payload);
            const uint64_t invocation = record->correlation_id.external.value;
            g_dispatches[invocation].push_back({record->start_timestamp,
                                                record->dispatch_info.kernel_id,
                                                record->end_timestamp - record->start_timestamp});
        }
    }
}

// Code-object callback: records kernel id -> demangled symbol as kernels load. The SDK
// frees the name string on unload, so the demangled copy is kept.
void code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t *, void *) {
    if (record.kind      != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
        record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
        record.phase     != ROCPROFILER_CALLBACK_PHASE_LOAD) {
        return;
    }
    auto * symbol = static_cast<
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *>(record.payload);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_kernel_names[symbol->kernel_id] = demangle(symbol->kernel_name);
}

void write_shape(std::ostringstream & out, const char * key, const int64_t ne[4]) {
    out << "\"" << key << "\": [" << ne[0] << ", " << ne[1] << ", " << ne[2] << ", " << ne[3] << "]";
}

// One fused node: its geometry plus the raw per-tensor data the consumer needs to compute the
// fused group's HBM traffic itself -- destination/source byte counts, a storage id per tensor
// (which buffer it uses, so aliasing tensors are recognised) and a source tensor id (dedup key).
// The consumer applies the exclusion policy (skip views, drop in-place destinations and internal
// reads, dedup); the producer only records each tensor's size and identity.
void write_fused_node(std::ostringstream & out, const op_record & rec) {
    out << "{\"ggml_op\": \"" << rec.op << "\", "
        << "\"dtype\": \"" << rec.dtype << "\", \"quant\": \"" << rec.quant << "\", ";
    write_shape(out, "ne", rec.ne);  out << ", ";
    out << "\"src_ne\": [";
    for (int j = 0; j < rec.n_src; j++) {
        if (j) out << ", ";
        out << "[" << rec.src_ne[j][0] << ", " << rec.src_ne[j][1] << ", "
            << rec.src_ne[j][2] << ", " << rec.src_ne[j][3] << "]";
    }
    out << "], \"src_types\": [";
    for (int j = 0; j < rec.n_src; j++) {
        if (j) out << ", ";
        out << "\"" << (rec.src_types[j] ? rec.src_types[j] : "") << "\"";
    }
    out << "], \"op_params\": [";
    for (int p = 0; p < n_op_params; p++) { if (p) out << ", "; out << rec.op_params[p]; }
    out << "], \"dst_bytes\": " << rec.dst_bytes << ", \"dst_sid\": " << rec.dst_sid << ", ";
    out << "\"src_bytes\": [";
    for (int j = 0; j < rec.n_src; j++) { if (j) out << ", "; out << rec.src_bytes[j]; }
    out << "], \"src_ids\": [";
    for (int j = 0; j < rec.n_src; j++) { if (j) out << ", "; out << rec.src_ids[j]; }
    out << "], \"src_sids\": [";
    for (int j = 0; j < rec.n_src; j++) { if (j) out << ", "; out << rec.src_sids[j]; }
    out << "], \"M\": " << rec.M << ", \"N\": " << rec.N << ", \"K\": " << rec.K
        << ", \"n_experts\": " << rec.n_experts << ", \"top_k\": " << rec.top_k << "}";
}

// Write the per-invocation JSON report. Registered with atexit while profiling is active.
void write_report() {
    if (!g_active || !p_flush) return;
    p_flush(g_buffer);

    std::lock_guard<std::mutex> lock(g_mutex);
    double total_us = 0.0;
    for (auto & [invocation, dispatches] : g_dispatches) {
        for (const auto & d : dispatches) total_us += d.duration_ns / 1e3;
        // order kernels within an op causally (buffer records arrive unordered)
        std::sort(dispatches.begin(), dispatches.end(),
                  [](const dispatch & a, const dispatch & b) { return a.start_ns < b.start_ns; });
    }

    std::ostringstream out;
    out << "{\n  \"device\": \"" << g_device << "\",\n";
    out << "  \"total_gpu_time_us\": " << total_us << ",\n  \"rows\": [\n";
    bool first = true;
    for (const auto & [invocation, geometry_id] : g_invocations) {
        auto dispatch_it = g_dispatches.find(invocation);
        if (dispatch_it == g_dispatches.end()) continue;  // no kernels recorded for this op
        auto record_it = g_records.find(geometry_id);
        if (record_it == g_records.end()) continue;
        const op_record & rec = record_it->second;

        if (!first) out << ",\n";
        first = false;

        out << "    {\"ggml_op\": \"" << rec.op << "\", ";
        if (!rec.fused_nodes.empty()) {
            out << "\"fused_ops\": [";
            for (size_t k = 0; k < rec.fused_nodes.size(); k++) {
                if (k) out << ", ";
                write_fused_node(out, rec.fused_nodes[k]);
            }
            out << "], ";
        }
        out << "\"dtype\": \"" << rec.dtype << "\", \"quant\": \"" << rec.quant << "\", ";
        write_shape(out, "ne", rec.ne);  out << ", ";
        out << "\"src_ne\": [";
        for (int j = 0; j < rec.n_src; j++) {
            if (j) out << ", ";
            out << "[" << rec.src_ne[j][0] << ", " << rec.src_ne[j][1] << ", "
                << rec.src_ne[j][2] << ", " << rec.src_ne[j][3] << "]";
        }
        out << "], ";
        out << "\"src_types\": [";
        for (int j = 0; j < rec.n_src; j++) {
            if (j) out << ", ";
            out << "\"" << (rec.src_types[j] ? rec.src_types[j] : "") << "\"";
        }
        out << "], ";
        out << "\"op_params\": [";
        for (int p = 0; p < n_op_params; p++) { if (p) out << ", "; out << rec.op_params[p]; }
        out << "], ";
        out << "\"bytes\": " << rec.bytes << ", ";
        out << "\"dst_bytes\": " << rec.dst_bytes << ", \"dst_sid\": " << rec.dst_sid << ", ";
        out << "\"src_bytes\": [";
        for (int j = 0; j < rec.n_src; j++) { if (j) out << ", "; out << rec.src_bytes[j]; }
        out << "], \"src_ids\": [";
        for (int j = 0; j < rec.n_src; j++) { if (j) out << ", "; out << rec.src_ids[j]; }
        out << "], \"src_sids\": [";
        for (int j = 0; j < rec.n_src; j++) { if (j) out << ", "; out << rec.src_sids[j]; }
        out << "], ";
        out << "\"M\": " << rec.M << ", \"N\": " << rec.N << ", \"K\": " << rec.K
            << ", \"n_experts\": " << rec.n_experts << ", \"top_k\": " << rec.top_k << ", ";
        out << "\"kernels\": [";
        bool kernel_first = true;
        for (const auto & d : dispatch_it->second) {
            if (!kernel_first) out << ", ";
            kernel_first = false;
            auto name_it = g_kernel_names.find(d.kernel_id);
            out << "{\"name\": \"";
            json_escape(out, name_it != g_kernel_names.end() ? name_it->second : "");
            out << "\", \"gpu_time_us\": " << d.duration_ns / 1e3 << "}";
        }
        out << "]}";
    }
    out << "\n  ]\n}\n";

    std::ofstream file(g_out_path);
    file << out.str();
    // runs from atexit, where the ggml logger is already torn down, so write directly
    fprintf(stderr, "ggml-roofline: wrote %s (%zu invocations, %zu unique ops, total GPU %.1f us)\n",
            g_out_path.c_str(), g_invocations.size(), g_records.size(), total_us);
}

int tool_init(rocprofiler_client_finalize_t, void *) {
    if (!resolve_symbols()) {
        GGML_LOG_ERROR("ggml-roofline: could not resolve rocprofiler-sdk symbols; disabled\n");
        g_active = false;
        return -1;
    }
    if (p_create_context(&g_context) != ROCPROFILER_STATUS_SUCCESS) return -1;
    // 256 MiB buffer, 4 MiB watermark: batch dispatch records so the callback fires in
    // large chunks instead of per record.
    if (p_create_buffer(g_context, 256 * 1024 * 1024, 4 * 1024 * 1024,
                        ROCPROFILER_BUFFER_POLICY_LOSSLESS, buffer_callback, nullptr, &g_buffer)
            != ROCPROFILER_STATUS_SUCCESS) return -1;
    if (p_config_buffer(g_context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, g_buffer)
            != ROCPROFILER_STATUS_SUCCESS) return -1;
    // Kernel-symbol names are optional: timing works without them, kernels just omit "name".
    if (p_config_callback) {
        p_config_callback(g_context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                          code_object_callback, nullptr);
    }
    if (p_start(g_context) != ROCPROFILER_STATUS_SUCCESS) return -1;
    return 0;
}

void tool_fini(void *) {}

rocprofiler_tool_configure_result_t g_configure_result{
    sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

rocprofiler_tool_configure_result_t *
configure(uint32_t, const char *, uint32_t, rocprofiler_client_id_t * client_id) {
    client_id->name = "ggml-roofline";
    g_active = true;
    return &g_configure_result;
}

using fn_force_configure = rocprofiler_status_t (*)(
    rocprofiler_tool_configure_result_t * (*)(uint32_t, const char *, uint32_t, rocprofiler_client_id_t *));

} // namespace

void ggml_cuda_roofline_init(void) {
    static std::once_flag once;
    std::call_once(once, [] {
        const char * out_path = getenv("GGML_ROOFLINE_OUT");
        if (!out_path || !out_path[0]) return;
        g_out_path = out_path;

        // Per-op attribution requires the eager node loop, so graphs must be disabled.
        // The backend reads this env var lazily and only checks for its presence; set it
        // here before that happens, without overriding an explicit user setting.
        if (setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0) == 0) {
            GGML_LOG_INFO("ggml-roofline: graphs disabled (required for per-op attribution)\n");
        }

        void * handle = dlopen("librocprofiler-sdk.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!handle) handle = dlopen("librocprofiler-sdk.so", RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            GGML_LOG_ERROR("ggml-roofline: cannot dlopen librocprofiler-sdk: %s\n", dlerror());
            return;
        }
        auto force_configure = (fn_force_configure) dlsym(handle, "rocprofiler_force_configure");
        if (!force_configure || force_configure(&configure) != ROCPROFILER_STATUS_SUCCESS) {
            GGML_LOG_ERROR("ggml-roofline: rocprofiler_force_configure failed\n");
            return;
        }
        std::atexit(write_report);
        GGML_LOG_INFO("ggml-roofline: enabled -> %s\n", g_out_path.c_str());
    });
}

void ggml_cuda_roofline_set_device(const char * arch) {
    if (g_active && arch) g_device = arch;
}

void ggml_cuda_roofline_reset(void) {
    if (!g_active) return;
    // Drain any pending records first so warmup dispatches are accounted, then dropped.
    if (p_flush) p_flush(g_buffer);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_records.clear();
    g_invocations.clear();
    g_dispatches.clear();
    // g_next_invocation stays monotonic so a late warmup record cannot collide with a
    // post-reset invocation id; g_kernel_names is kept (code objects do not reload).
}

void ggml_cuda_roofline_begin_op(const struct ggml_tensor * node) {
    if (!g_active || node == nullptr || !p_push_id) return;

    op_record rec;
    fill_head_record(rec, node);
    const uint64_t geometry_id = head_geometry_id(rec, node);

    // Each op invocation gets a unique correlation id so its kernels stay separate; the
    // shared geometry is stored once per shape.
    const uint64_t invocation = g_next_invocation.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_invocations.emplace(invocation, geometry_id);
        if (g_records.find(geometry_id) == g_records.end()) g_records.emplace(geometry_id, rec);
    }

    // Tag the kernels launched until the next op with this invocation id. rocprofiler
    // keeps a per-thread stack, so pop the previous id before pushing the new one.
    static thread_local rocprofiler_thread_id_t thread_id = [] {
        rocprofiler_thread_id_t t = 0; if (p_get_thread_id) p_get_thread_id(&t); return t;
    }();
    static thread_local bool pushed = false;
    if (pushed) {
        rocprofiler_user_data_t previous;
        p_pop_id(g_context, thread_id, &previous);
    }
    rocprofiler_user_data_t current;
    current.value = invocation;
    p_push_id(g_context, thread_id, current);
    pushed = true;

    // Remember which invocation this thread just tagged so a following fusion can override
    // its record to cover the whole fused span (see ggml_cuda_roofline_fuse_ops).
    g_current_invocation = invocation;
}

void ggml_cuda_roofline_fuse_ops(const struct ggml_cgraph * cgraph, int node_idx, int node_count) {
    if (!g_active || cgraph == nullptr || node_count < 2 || g_current_invocation == 0) return;

    const ggml_tensor * head = cgraph->nodes[node_idx];

    // Keep the head op's geometry (op name, shapes, M/N/K) so the row is still labelled by the
    // head op. The row-level byte fields stay head-only and are ignored for fused rows; the
    // consumer derives the group's traffic from the per-node data recorded below.
    op_record rec;
    fill_head_record(rec, head);

    // Record every fused node's geometry plus its raw per-tensor byte counts and storage/tensor
    // ids (captured by fill_head_record while the graph is live). The consumer sums exact FLOPs
    // across the group and computes its external HBM traffic from them -- see write_fused_node.
    rec.fused_nodes.reserve(node_count);
    for (int j = node_idx; j < node_idx + node_count; ++j) {
        op_record sub;
        fill_head_record(sub, cgraph->nodes[j]);
        rec.fused_nodes.push_back(std::move(sub));
    }

    // Fused geometry id: head geometry plus each fused node's op and destination shape, so
    // identical fusions deduplicate and distinct ones stay separate.
    uint64_t geometry_id = head_geometry_id(rec, head);
    for (int j = node_idx; j < node_idx + node_count; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        geometry_id = hash_mix(geometry_id, (uint64_t) n->op);
        for (int d = 0; d < 4; d++) geometry_id = hash_mix(geometry_id, (uint64_t) n->ne[d]);
    }

    // Re-point the current invocation at the fused record. The provisional head-only record
    // from begin_op stays in g_records; if no non-fused invocation references it, it is
    // simply never emitted (the report iterates g_invocations).
    std::lock_guard<std::mutex> lock(g_mutex);
    g_invocations[g_current_invocation] = geometry_id;
    if (g_records.find(geometry_id) == g_records.end()) g_records.emplace(geometry_id, std::move(rec));
}
