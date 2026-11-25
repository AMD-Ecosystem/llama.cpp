// HIPEP: a device per ROCm device (HIPEP0 over ROCm0) whose splits are compiled
// and run by libhipep_backend (hipep_backend.h, hip-ep):
//
//   llama-cli -m M.gguf --device HIPEP0 -ngl 99
//
// the layers given to it are its: llama.cpp puts their weights and cache in
// its buffers - ROCm0's memory, allocated by ggml-hip's buffer code under a
// buffer type of its own - and it takes every operation of them ggml-hip
// runs. each split the scheduler hands it is cut into runs: what the library
// compiles runs by compiled code, the rest - and a run whose code is not ready
// or cannot be had - by a ggml-hip backend of its own on the same device.
// listed with ROCm0 it shares the layers as -ts says (--device ROCm0,HIPEP0
// -ts 3,1: ggml-hip the first three quarters, it the last).
//
// environment: HIPEP_BACKEND_WAIT=1 compiles on the calling thread (a split
// waits for its code); HIPEP_BACKEND_LOG=1 says what each run ran by;
// HIPEP_BACKEND_PROFILE=1 says where the time went when the backend goes.
// the library's own: HIPEP_BACKEND_CACHE, HIPEP_BACKEND_OPS,
// HIPEP_BACKEND_COMPILE=0 (code from the cache only).

#include "ggml-hipep.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "split-desc.h"

#ifndef GGML_BACKEND_DL
#    include "ggml-cuda.h"
#endif

#include <hip/hip_runtime_api.h>
#include <hipep_backend.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static bool hipep_env(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && *value != 0 && std::strcmp(value, "0") != 0;
}

// the environment, read once
struct hipep_options {
    bool wait       = hipep_env("HIPEP_BACKEND_WAIT");
    bool log        = hipep_env("HIPEP_BACKEND_LOG");
    bool profile    = hipep_env("HIPEP_BACKEND_PROFILE");
    bool precompile = [] {
        const char * value = std::getenv("HIPEP_BACKEND_PRECOMPILE");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
};

static const hipep_options & hipep_opts() {
    static const hipep_options options;
    return options;
}

struct hipep_device_context {
    ggml_backend_dev_t rocm  = nullptr;
    int                index = 0;
    std::string        name;
    std::string        description;
    // ROCm0's buffer type under another address, name and device: ggml-hip
    // allocates and copies its buffers, but they are this device's - the
    // layers llama.cpp gives it, its compute memory apart from ROCm0's
    ggml_backend_buffer_type buft = {};

    // the library's context, opened when first needed and closed with the
    // last backend: nothing of it runs when the process tears HIP down
    std::mutex          mutex;
    hipep_backend_ctx * sdk      = nullptr;
    int                 backends = 0;

    hipep_backend_ctx * open() {
        std::lock_guard<std::mutex> lock(mutex);
        if (sdk == nullptr) {
            sdk = hipep_backend_open(index, nullptr);
            if (sdk == nullptr) {
                GGML_LOG_ERROR("%s: %s\n", name.c_str(), hipep_backend_last_error(nullptr));
            }
        }
        return sdk;
    }
};

// a run of a split's nodes, the graph's nodes [begin, end): what the library
// compiles, or what it does not, which its ggml-hip runs
struct hipep_segment {
    int                  begin    = 0;
    int                  end      = 0;
    bool                 compiled = false;
    // compiled: its nodes and what it hands on, the tensors it reads, its code
    hipep_split_nodes    split;
    hipep_split_tensors  tensors;
    std::vector<void *>  data;
    hipep_backend_code * code   = nullptr;
    // compiling asked for, or the code cannot be had or failed: the library is
    // not asked again, and ggml-hip runs the run
    bool                 asked  = false;
    bool                 failed = false;
    std::string          ops;
};

// what a split llama.cpp hands again is: the same tensors where they were, cut
// as before
struct hipep_memo {
    std::vector<uintptr_t>     fingerprint;
    std::vector<hipep_segment> segments;
};

struct hipep_profile {
    int64_t runs    = 0;
    double  seconds = 0;
};

struct hipep_backend_context {
    hipep_device_context * device = nullptr;
    ggml_backend_t         rocm   = nullptr;

    std::unordered_map<const ggml_cgraph *, hipep_memo> memos;

    int64_t splits = 0, compiled = 0, fallback = 0;
    double  wait_seconds = 0, run_seconds = 0, fallback_seconds = 0, collect_seconds = 0;
    std::map<std::string, hipep_profile> profile;
};

static std::vector<ggml_backend_device> & hipep_devices() {
    static std::vector<ggml_backend_device> list;
    return list;
}

static hipep_device_context & hipep_device_of(ggml_backend_dev_t dev) {
    return *static_cast<hipep_device_context *>(dev->context);
}

// ---------------------------------------------------------------- backend

static ggml_guid_t ggml_backend_hipep_guid() {
    static ggml_guid guid = { 0x68, 0x69, 0x70, 0x65, 0x70, 0x2d, 0x62, 0x61,
                              0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x30, 0x01 };
    return &guid;
}

static const char * ggml_backend_hipep_get_name(ggml_backend_t backend) {
    return static_cast<hipep_backend_context *>(backend->context)->device->name.c_str();
}

static void ggml_backend_hipep_free(ggml_backend_t backend) {
    auto * ctx = static_cast<hipep_backend_context *>(backend->context);
    if (hipep_opts().log || hipep_opts().profile) {
        fprintf(stderr, "%s: %lld splits, %lld by compiled code, %lld by ggml-hip; collecting %.3f s, waiting %.3f s, "
                      "compiled %.3f s, ggml-hip %.3f s\n",
                      ctx->device->name.c_str(), (long long) ctx->splits, (long long) ctx->compiled,
                      (long long) ctx->fallback, ctx->collect_seconds, ctx->wait_seconds, ctx->run_seconds,
                      ctx->fallback_seconds);
    }
    if (hipep_opts().profile) {
        std::vector<std::pair<std::string, hipep_profile>> rows(ctx->profile.begin(), ctx->profile.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto & a, const auto & b) { return a.second.seconds > b.second.seconds; });
        for (const auto & [what, p] : rows) {
            fprintf(stderr, "%s profile: %7lld runs %8.3f s %8.1f us/run  %s\n", ctx->device->name.c_str(),
                          (long long) p.runs, p.seconds, 1e6 * p.seconds / std::max<int64_t>(p.runs, 1),
                          what.c_str());
        }
    }
    ggml_backend_free(ctx->rocm);
    {
        hipep_device_context &      device = *ctx->device;
        std::lock_guard<std::mutex> lock(device.mutex);
        if (--device.backends == 0 && device.sdk != nullptr) {
            hipep_backend_close(device.sdk);
            device.sdk = nullptr;
        }
    }
    delete ctx;
    delete backend;
}

static void ggml_backend_hipep_synchronize(ggml_backend_t backend) {
    ggml_backend_synchronize(static_cast<hipep_backend_context *>(backend->context)->rocm);
    (void) hipDeviceSynchronize();
}

// the split llama.cpp hands, as it is now: what tells it from another
static std::vector<uintptr_t> hipep_fingerprint(const ggml_cgraph & split) {
    std::vector<uintptr_t> fingerprint;
    fingerprint.reserve(static_cast<size_t>(split.n_nodes) * 16);
    for (int i = 0; i < split.n_nodes; ++i) {
        const ggml_tensor * node = split.nodes[i];
        for (uintptr_t word : { reinterpret_cast<uintptr_t>(node), reinterpret_cast<uintptr_t>(node->data),
                                static_cast<uintptr_t>(node->op), static_cast<uintptr_t>(node->flags) }) {
            fingerprint.push_back(word);
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            fingerprint.push_back(static_cast<uintptr_t>(node->ne[d]));
            fingerprint.push_back(static_cast<uintptr_t>(node->nb[d]));
        }
        for (const ggml_tensor * src : node->src) {
            if (src != nullptr) {
                fingerprint.push_back(reinterpret_cast<uintptr_t>(src));
                fingerprint.push_back(reinterpret_cast<uintptr_t>(src->data));
            }
        }
    }
    return fingerprint;
}

static std::string hipep_ops_of(const hipep_split_nodes & split) {
    std::map<std::string, int> ops;
    for (const ggml_tensor * node : split.nodes) {
        if (node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE && node->op != GGML_OP_PERMUTE &&
            node->op != GGML_OP_TRANSPOSE) {
            ++ops[ggml_op_desc(node)];
        }
    }
    std::string text;
    for (const auto & [op, n] : ops) {
        text += (text.empty() ? "" : " ") + op + (n > 1 ? "x" + std::to_string(n) : "");
    }
    return text;
}

static hipep_segment hipep_segment_of(int begin, int end, bool compiled) {
    hipep_segment segment;
    segment.begin    = begin;
    segment.end      = end;
    segment.compiled = compiled;
    return segment;
}

// nodes [begin, end) of `graph`, as a graph of their own (ggml_graph_view, which
// ggml-base does not export)
static ggml_cgraph hipep_graph_view(ggml_cgraph * graph, int begin, int end) {
    ggml_cgraph view = *graph;
    view.size        = 0;
    view.n_nodes     = end - begin;
    view.n_leafs     = 0;
    view.nodes       = graph->nodes + begin;
    view.grads       = nullptr;
    view.grad_accs   = nullptr;
    view.leafs       = nullptr;
    view.uid         = 0;
    view.from_sched  = false;
    return view;
}

static bool hipep_is_view(const ggml_tensor * node) {
    return node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
           node->op == GGML_OP_TRANSPOSE;
}

// the split cut into runs of what the library compiles and what it does not
// (it takes every operation of its layers ggml-hip runs). a view goes with the
// run it is in. a compiled run hands on what the split does and what a later
// run reads
static std::vector<hipep_segment> hipep_segments_of(hipep_backend_ctx * sdk, const ggml_cgraph & graph) {
    std::vector<hipep_segment> segments;
    const hipep_split_nodes    whole = hipep_split_nodes_of(graph);
    if (sdk == nullptr || !whole.computable || whole.nodes.empty()) {
        segments.push_back(hipep_segment_of(0, graph.n_nodes, false));
        segments.back().failed = !whole.nodes.empty();
        return segments;
    }
    std::unordered_map<const ggml_tensor *, bool> exported;
    for (size_t i = 0; i < whole.nodes.size(); ++i) {
        exported.emplace(whole.nodes[i], whole.exported[i]);
    }
    // per node: compiled (1) or not (0); a view, or a node nothing computes
    // (-1), goes with the run it is in, the first ones with the first run
    std::vector<int> kinds(graph.n_nodes, -1);
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && !hipep_is_view(node)) {
            hipep_split_tensors op;
            hipep_collect_op(node, op);
            const hipep_backend_split_desc desc = op.desc();
            kinds[i]                            = hipep_backend_supports(sdk, &desc) != 0 ? 1 : 0;
        }
    }
    auto first = std::find_if(kinds.begin(), kinds.end(), [](int kind) { return kind != -1; });
    int  kind  = first == kinds.end() ? 0 : *first;
    for (int i = 0; i < graph.n_nodes; ++i) {
        kind = kinds[i] == -1 ? kind : kinds[i];
        if (segments.empty() || segments.back().compiled != (kind == 1)) {
            segments.push_back(hipep_segment_of(i, i + 1, kind == 1));
        }
        segments.back().end = i + 1;
    }
    // what the runs after each one read
    std::unordered_set<const ggml_tensor *> read_after;
    for (size_t s = segments.size(); s-- > 0;) {
        hipep_segment & segment = segments[s];
        if (segment.compiled) {
            for (int i = segment.begin; i < segment.end; ++i) {
                const ggml_tensor * node  = graph.nodes[i];
                auto                found = exported.find(node);
                if (found == exported.end()) {
                    continue;
                }
                segment.split.nodes.push_back(node);
                segment.split.exported.push_back(found->second ||
                                                 (!hipep_is_view(node) && read_after.count(node) != 0));
            }
            segment.failed = segment.split.nodes.empty() || !hipep_collect_split(segment.split, segment.tensors);
            segment.data.resize(segment.tensors.tensors.size());
            if (hipep_opts().profile || hipep_opts().log) {
                segment.ops = hipep_ops_of(segment.split);
            }
        }
        for (int i = segment.begin; i < segment.end; ++i) {
            for (const ggml_tensor * src : graph.nodes[i]->src) {
                if (src != nullptr) {
                    read_after.insert(src);
                    if (src->view_src != nullptr) {
                        read_after.insert(src->view_src);
                    }
                }
            }
        }
    }
    return segments;
}

// before the first step: the scheduler's splits of the graphs llama.cpp reserves (a prompt's worst-case ubatch, a
// decode step) compiled now, while the context is made, and not at the first step that runs them
// (HIPEP_BACKEND_PRECOMPILE=0: not). a graph only split - a probe of whether a fused op stays on the layer's
// device - never runs: its splits are not compiled
static void ggml_backend_hipep_graph_optimize(ggml_backend_t backend, ggml_cgraph * graph,
                                              ggml_backend_graph_optimize_params * params) {
    auto * ctx = static_cast<hipep_backend_context *>(backend->context);
    if (ctx->splits > 0 || (params != nullptr && params->split_only) || !hipep_opts().precompile) {
        return;
    }
    hipep_backend_ctx * sdk = ctx->device->open();
    if (sdk == nullptr) {
        return;
    }
    for (const hipep_segment & segment : hipep_segments_of(sdk, *graph)) {
        if (!segment.compiled || segment.failed) {
            continue;
        }
        const hipep_backend_split_desc desc = segment.tensors.desc();
        if (hipep_backend_find(sdk, &desc) == nullptr) {
            hipep_backend_compile(sdk, &desc, /*wait=*/true);
        }
    }
}

// one compiled run, by its code: false when there is none (yet) or it failed
static bool hipep_run_compiled(hipep_backend_context * ctx,
                               hipep_backend_ctx *     sdk,
                               hipep_segment &         segment,
                               std::string &           why) {
    if (segment.code == nullptr && !segment.failed) {
        const hipep_backend_split_desc desc = segment.tensors.desc();
        segment.code                        = hipep_backend_find(sdk, &desc);
        if (segment.code == nullptr && !segment.asked) {
            segment.code  = hipep_backend_compile(sdk, &desc, hipep_opts().wait);
            segment.asked = true;
        }
        if (segment.code == nullptr) {
            why = hipep_backend_last_error(sdk);
        }
    }
    if (segment.code == nullptr) {
        return false;
    }
    for (size_t i = 0; i < segment.data.size(); ++i) {
        segment.data[i] = segment.tensors.originals[i]->data;
    }
    if (hipep_backend_run(segment.code, segment.data.data(), nullptr) != 0) {
        // the code ran and failed: never again
        why            = hipep_backend_last_error(sdk);
        segment.code   = nullptr;
        segment.failed = true;
        GGML_LOG_ERROR("%s: split of %s: %s\n", ctx->device->name.c_str(), segment.ops.c_str(), why.c_str());
        return false;
    }
    return true;
}

static ggml_status ggml_backend_hipep_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    using clock      = std::chrono::steady_clock;
    const auto since = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };
    auto *     ctx   = static_cast<hipep_backend_context *>(backend->context);

    // the compiled code runs on the library's own stream, which the scheduler
    // does not know of: both sides wait here - for the device's work before
    // the split, and for the split's before anything else goes on
    clock::time_point start = clock::now();
    (void) hipDeviceSynchronize();
    ctx->wait_seconds += since(start);

    start                              = clock::now();
    hipep_backend_ctx *    sdk         = ctx->device->open();
    std::vector<uintptr_t> fingerprint = hipep_fingerprint(*graph);
    hipep_memo &           memo        = ctx->memos[graph];
    if (memo.fingerprint != fingerprint) {
        memo.fingerprint = std::move(fingerprint);
        memo.segments    = hipep_segments_of(sdk, *graph);
    }
    ctx->collect_seconds += since(start);
    if (graph->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }
    ++ctx->splits;

    ggml_status status = GGML_STATUS_SUCCESS;
    for (hipep_segment & segment : memo.segments) {
        start = clock::now();
        std::string why;
        const bool  ran = segment.compiled && sdk != nullptr && hipep_run_compiled(ctx, sdk, segment, why);
        if (!ran) {
            ggml_cgraph run = hipep_graph_view(graph, segment.begin, segment.end);
            status          = ggml_backend_graph_compute_async(ctx->rocm, &run);
            ggml_backend_synchronize(ctx->rocm);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        (void) hipDeviceSynchronize();
        const double seconds = since(start);
        ++(ran ? ctx->compiled : ctx->fallback);
        (ran ? ctx->run_seconds : ctx->fallback_seconds) += seconds;
        if (hipep_opts().profile) {
            hipep_profile & p = ctx->profile[std::string(ran ? "compiled " : "ggml-hip ") +
                                             (segment.compiled ? segment.ops : std::string("(not compiled)"))];
            ++p.runs;
            p.seconds += seconds;
        }
        if (hipep_opts().log) {
            fprintf(stderr, "%s: split (%d nodes): %s%s%s\n", ctx->device->name.c_str(), segment.end - segment.begin,
                    ran ? "compiled" : "ggml-hip", why.empty() ? "" : ": ", why.c_str());
        }
    }
    return status;
}

static const ggml_backend_i ggml_backend_hipep_interface = {
    /* .get_name                = */ ggml_backend_hipep_get_name,
    /* .free                    = */ ggml_backend_hipep_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_hipep_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_hipep_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ ggml_backend_hipep_graph_optimize,
};

// ---------------------------------------------------------------- device

static const char * ggml_backend_hipep_device_get_name(ggml_backend_dev_t dev) {
    return hipep_device_of(dev).name.c_str();
}

static const char * ggml_backend_hipep_device_get_description(ggml_backend_dev_t dev) {
    return hipep_device_of(dev).description.c_str();
}

static void ggml_backend_hipep_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // ROCm0's: the layers are given to this device (-ts 1,0), in its memory
    ggml_backend_dev_memory(hipep_device_of(dev).rocm, free, total);
}

// ROCm0's type: the same GPU, integrated or not
static enum ggml_backend_dev_type ggml_backend_hipep_device_get_type(ggml_backend_dev_t dev) {
    return ggml_backend_dev_type(hipep_device_of(dev).rocm);
}

// ROCm0's properties under its own name. The device id is ROCm0's: llama.cpp's
// default devices take one device a GPU, ggml-hip's, so HIPEP runs a model only
// when --device names it
static void ggml_backend_hipep_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_dev_get_props(hipep_device_of(dev).rocm, props);
    props->name                      = ggml_backend_hipep_device_get_name(dev);
    props->description               = ggml_backend_hipep_device_get_description(dev);
    props->caps.async                = false;
    props->caps.host_buffer          = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events               = false;
}

static ggml_backend_t ggml_backend_hipep_device_init(ggml_backend_dev_t dev, const char * params) {
    hipep_device_context & device = hipep_device_of(dev);
    ggml_backend_t         rocm   = ggml_backend_dev_init(device.rocm, params);
    if (rocm == nullptr) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(device.mutex);
        ++device.backends;
    }
    auto * ctx   = new hipep_backend_context;
    ctx->device  = &device;
    ctx->rocm    = rocm;
    return new ggml_backend{ ggml_backend_hipep_guid(), ggml_backend_hipep_interface, dev, ctx };
}

static ggml_backend_buffer_type_t ggml_backend_hipep_device_get_buffer_type(ggml_backend_dev_t dev) {
    return &hipep_device_of(dev).buft;
}

// every operation ggml-hip runs: what the library does not compile of a split,
// its own ggml-hip runs (hipep_segments_of), so its layers stay on the GPU
static bool ggml_backend_hipep_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_dev_supports_op(hipep_device_of(dev).rocm, op);
}

static const char * ggml_backend_hipep_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return hipep_device_of(buft->device).name.c_str();
}

static bool ggml_backend_hipep_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // its own buffers alone: the layers given to it are what it runs, whatever
    // the order the devices are listed in (it reads ROCm0's through a copy)
    return buft == &hipep_device_of(dev).buft;
}

static const ggml_backend_device_i ggml_backend_hipep_device_interface = {
    /* .get_name             = */ ggml_backend_hipep_device_get_name,
    /* .get_description      = */ ggml_backend_hipep_device_get_description,
    /* .get_memory           = */ ggml_backend_hipep_device_get_memory,
    /* .get_type             = */ ggml_backend_hipep_device_get_type,
    /* .get_props            = */ ggml_backend_hipep_device_get_props,
    /* .init_backend         = */ ggml_backend_hipep_device_init,
    /* .get_buffer_type      = */ ggml_backend_hipep_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hipep_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hipep_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---------------------------------------------------------------- registry

static const char * ggml_backend_hipep_reg_get_name(ggml_backend_reg_t) {
    return GGML_HIPEP_NAME;
}

static size_t ggml_backend_hipep_reg_get_device_count(ggml_backend_reg_t) {
    return hipep_devices().size();
}

static ggml_backend_dev_t ggml_backend_hipep_reg_get_device(ggml_backend_reg_t, size_t index) {
    return index < hipep_devices().size() ? &hipep_devices()[index] : nullptr;
}

// the scheduler marks what each split hands on (GGML_TENSOR_FLAG_SPLIT_OUTPUT): what compiled code writes
static bool ggml_backend_hipep_reads_split_outputs() {
    return true;
}

static void * ggml_backend_hipep_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_reads_split_outputs") == 0) {
        return reinterpret_cast<void *>(static_cast<ggml_backend_reads_split_outputs_t>(
            ggml_backend_hipep_reads_split_outputs));
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hipep_reg_interface = {
    /* .get_name         = */ ggml_backend_hipep_reg_get_name,
    /* .get_device_count = */ ggml_backend_hipep_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hipep_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hipep_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hipep_reg(void) {
    static ggml_backend_reg reg = { GGML_BACKEND_API_VERSION, ggml_backend_hipep_reg_interface, nullptr };
    static std::once_flag   made;
    std::call_once(made, [] {
        if (hipep_backend_abi_version() != HIPEP_BACKEND_ABI_VERSION) {
            GGML_LOG_ERROR("%s: libhipep_backend is of ABI %d, not %d: no devices\n", __func__,
                           hipep_backend_abi_version(), HIPEP_BACKEND_ABI_VERSION);
            return;
        }
        // a device over each ggml-hip device
#ifdef GGML_BACKEND_DL
        ggml_backend_reg_t rocm = ggml_backend_reg_by_name("ROCm");
#else
        ggml_backend_reg_t rocm = ggml_backend_cuda_reg();
#endif
        if (rocm == nullptr) {
            GGML_LOG_ERROR("%s: no ggml-hip devices to run over\n", __func__);
            return;
        }
        static std::vector<hipep_device_context> contexts(ggml_backend_reg_dev_count(rocm));
        for (size_t i = 0; i < contexts.size(); ++i) {
            hipep_device_context & c = contexts[i];
            c.rocm                   = ggml_backend_reg_dev_get(rocm, i);
            c.index                  = static_cast<int>(i); // ggml-hip's device i is HIP device i
            c.name                   = GGML_HIPEP_NAME + std::to_string(c.index);
            c.description = std::string("hip-ep compiled splits over ") + ggml_backend_dev_name(c.rocm);
            c.buft        = *ggml_backend_dev_buffer_type(c.rocm);
        }
        for (hipep_device_context & c : contexts) {
            hipep_devices().push_back(ggml_backend_device{ ggml_backend_hipep_device_interface, &reg, &c });
        }
        // the buffer type is this device's, by its name too: llama.cpp
        // allocates a layer's weights and cache on the device their buffer
        // type names, and finds buffer types by name (tensor overrides)
        for (size_t i = 0; i < contexts.size(); ++i) {
            contexts[i].buft.device        = &hipep_devices()[i];
            contexts[i].buft.iface.get_name = ggml_backend_hipep_buffer_type_get_name;
        }
    });
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hipep_reg)
