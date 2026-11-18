// HIPEP: a device per ROCm device (HIPEP0 over ROCm0) whose splits are compiled
// and run by libhipep_backend (hipep_backend.h, hip-ep):
//
//   llama-cli -m M.gguf --device HIPEP0,ROCm0 -ts 1,0 -ngl 99
//
// the layers given to it (-ts) are its: llama.cpp puts their weights and cache
// in its buffers - ROCm0's memory, allocated by ggml-hip's buffer code under a
// buffer type of its own - and their norms on it, so they are in its splits.
// listed after ROCm0 (--device ROCm0,HIPEP0 -ts 3,1) it runs the last layers
// and ggml-hip the others; listed first with no layers (-ts 0,1) it takes,
// from ROCm0's layers, what it compiles. it takes what the library compiles
// and ggml-hip runs too; each split the scheduler hands it runs by compiled
// code - or, until that is ready or when it cannot be, by a ggml-hip backend
// of its own on the same device.
//
// environment: HIPEP_BACKEND_WAIT=1 compiles on the calling thread (a split
// waits for its code); HIPEP_BACKEND_LOG=1 says what each split ran by;
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
#include <vector>

static bool hipep_env(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && *value != 0 && std::strcmp(value, "0") != 0;
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

// what a split llama.cpp hands again is: the same tensors where they were,
// collected as before
struct hipep_memo {
    std::vector<uintptr_t> fingerprint;
    hipep_split_nodes      split;
    hipep_split_tensors    tensors;
    std::vector<void *>    data;
    hipep_backend_code *   code       = nullptr;
    bool                   computable = true;
    // compiling asked for, or the code failed: the library is not asked again
    bool                   asked  = false;
    bool                   failed = false;
    std::string            ops;
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
    if (hipep_env("HIPEP_BACKEND_LOG") || hipep_env("HIPEP_BACKEND_PROFILE")) {
        fprintf(stderr, "%s: %lld splits, %lld by compiled code, %lld by ggml-hip; collecting %.3f s, waiting %.3f s, "
                      "compiled %.3f s, ggml-hip %.3f s\n",
                      ctx->device->name.c_str(), (long long) ctx->splits, (long long) ctx->compiled,
                      (long long) ctx->fallback, ctx->collect_seconds, ctx->wait_seconds, ctx->run_seconds,
                      ctx->fallback_seconds);
    }
    if (hipep_env("HIPEP_BACKEND_PROFILE")) {
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

static ggml_status ggml_backend_hipep_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    using clock      = std::chrono::steady_clock;
    const auto since = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };
    auto *     ctx   = static_cast<hipep_backend_context *>(backend->context);

    // its buffers are ROCm0's, so the scheduler copies nothing between it and
    // ROCm0 - and so does not always wait for one before the other runs. both
    // sides wait here: for ROCm0's work before the split, and for the split's
    // before ROCm0 goes on
    clock::time_point start = clock::now();
    (void) hipDeviceSynchronize();
    ctx->wait_seconds += since(start);

    start                              = clock::now();
    std::vector<uintptr_t> fingerprint = hipep_fingerprint(*graph);
    hipep_memo &           memo        = ctx->memos[graph];
    if (memo.fingerprint != fingerprint) {
        memo             = {};
        memo.fingerprint = std::move(fingerprint);
        memo.split       = hipep_split_nodes_of(*graph);
        memo.computable  = memo.split.computable && hipep_collect_split(memo.split, memo.tensors);
        memo.data.resize(memo.tensors.tensors.size());
        if (hipep_env("HIPEP_BACKEND_PROFILE") || hipep_env("HIPEP_BACKEND_LOG")) {
            memo.ops = hipep_ops_of(memo.split);
        }
    }

    hipep_backend_ctx * sdk = ctx->device->open();
    bool                ran = false;
    std::string         why;
    if (memo.computable && !memo.split.nodes.empty() && sdk != nullptr) {
        ++ctx->splits;
        if (memo.code == nullptr && !memo.failed) {
            const hipep_backend_split_desc desc = memo.tensors.desc();
            memo.code                           = hipep_backend_find(sdk, &desc);
            if (memo.code == nullptr && !memo.asked) {
                memo.code  = hipep_backend_compile(sdk, &desc, hipep_env("HIPEP_BACKEND_WAIT"));
                memo.asked = true;
            }
            if (memo.code == nullptr) {
                why = hipep_backend_last_error(sdk);
            }
        }
        ctx->collect_seconds += since(start);
        start = clock::now();
        if (memo.code != nullptr) {
            for (size_t i = 0; i < memo.data.size(); ++i) {
                memo.data[i] = memo.tensors.originals[i]->data;
            }
            ran = hipep_backend_run(memo.code, memo.data.data(), nullptr) == 0;
            if (!ran) {
                // the code ran and failed: never again
                why         = hipep_backend_last_error(sdk);
                memo.code   = nullptr;
                memo.failed = true;
                GGML_LOG_ERROR("%s: split of %s: %s\n", ctx->device->name.c_str(), memo.ops.c_str(), why.c_str());
            }
        }
    } else if (!memo.split.nodes.empty()) {
        ++ctx->splits;
        ctx->collect_seconds += since(start);
        start = clock::now();
    } else {
        // nothing to compute
        return GGML_STATUS_SUCCESS;
    }

    ggml_status status = GGML_STATUS_SUCCESS;
    if (!ran) {
        status = ggml_backend_graph_compute_async(ctx->rocm, graph);
        ggml_backend_synchronize(ctx->rocm);
    }
    (void) hipDeviceSynchronize();
    const double seconds = since(start);
    ++(ran ? ctx->compiled : ctx->fallback);
    (ran ? ctx->run_seconds : ctx->fallback_seconds) += seconds;
    if (hipep_env("HIPEP_BACKEND_PROFILE")) {
        hipep_profile & p = ctx->profile[std::string(ran ? "compiled " : "ggml-hip ") + memo.ops];
        ++p.runs;
        p.seconds += seconds;
    }
    if (hipep_env("HIPEP_BACKEND_LOG")) {
        fprintf(stderr, "%s: split (%d nodes): %s%s%s\n", ctx->device->name.c_str(), graph->n_nodes,
                ran ? "compiled" : "ggml-hip", why.empty() ? "" : ": ", why.c_str());
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
    /* .graph_optimize          = */ nullptr,
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

static enum ggml_backend_dev_type ggml_backend_hipep_device_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hipep_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hipep_device_get_name(dev);
    props->description = ggml_backend_hipep_device_get_description(dev);
    ggml_backend_hipep_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type      = ggml_backend_hipep_device_get_type(dev);
    props->device_id = nullptr;
    props->caps      = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
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

static bool ggml_backend_hipep_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    hipep_device_context & device = hipep_device_of(dev);
    if (!ggml_backend_dev_supports_op(device.rocm, op)) {
        return false;
    }
    // an operation writing into its own buffers (a recurrent state's update in
    // place) is its, compiled or not: no other device takes them. what it does
    // not compile, its ggml-hip runs
    const ggml_tensor * storage = op->view_src != nullptr ? op->view_src : op;
    if (storage->buffer != nullptr && ggml_backend_buffer_get_type(storage->buffer) == &device.buft &&
        ggml_backend_buffer_get_usage(storage->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return true;
    }
    hipep_backend_ctx * sdk = device.open();
    if (sdk == nullptr) {
        return false;
    }
    hipep_split_tensors tensors;
    hipep_collect_op(op, tensors);
    const hipep_backend_split_desc desc = tensors.desc();
    return hipep_backend_supports(sdk, &desc) != 0;
}

static const char * ggml_backend_hipep_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return hipep_device_of(buft->device).name.c_str();
}

static bool ggml_backend_hipep_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // its own buffers, and ROCm0's: it takes what it compiles of ROCm0's
    // layers too, unless ROCm0 is listed first (it then keeps them)
    hipep_device_context & device = hipep_device_of(dev);
    return buft == &device.buft || buft == ggml_backend_dev_buffer_type(device.rocm);
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

static const ggml_backend_reg_i ggml_backend_hipep_reg_interface = {
    /* .get_name         = */ ggml_backend_hipep_reg_get_name,
    /* .get_device_count = */ ggml_backend_hipep_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hipep_reg_get_device,
    /* .get_proc_address = */ nullptr,
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
            c.index                  = std::atoi(ggml_backend_dev_name(c.rocm) + 4);
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
