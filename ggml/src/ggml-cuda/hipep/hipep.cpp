// Each split ggml-hip computes is cut into runs: what libhipep_backend compiles runs by its code, the rest - and a
// run whose code is not ready or fails - by ggml-hip.
//
// environment: GGML_HIP_HIPEP=0 turns it off; HIPEP_BACKEND_WAIT=1 compiles on the calling thread;
// HIPEP_BACKEND_LOG=1 says what each run ran by; HIPEP_BACKEND_PROFILE=1 says where the time went;
// HIPEP_BACKEND_PRECOMPILE=0 compiles at the first step, not when the context is made.
// the library's own: HIPEP_BACKEND_CACHE, HIPEP_BACKEND_OPS, HIPEP_BACKEND_COMPILE=0.

#include "hipep.h"

#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "split-desc.h"

#include <hip/hip_runtime_api.h>
#include <hipep_backend.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static bool hipep_env(const char * name, bool otherwise) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == 0) {
        return otherwise;
    }
    return std::strcmp(value, "0") != 0;
}

// the environment, read once
struct hipep_options {
    bool enabled    = hipep_env("GGML_HIP_HIPEP", true);
    bool wait       = hipep_env("HIPEP_BACKEND_WAIT", false);
    bool log        = hipep_env("HIPEP_BACKEND_LOG", false);
    bool profile    = hipep_env("HIPEP_BACKEND_PROFILE", false);
    bool precompile = hipep_env("HIPEP_BACKEND_PRECOMPILE", true);
};

static const hipep_options & hipep_opts() {
    static const hipep_options options;
    return options;
}

// the library's context of a device, opened when first needed and closed with its last backend: nothing of it
// runs when the process tears HIP down
struct hipep_device {
    std::mutex          mutex;
    hipep_backend_ctx * sdk      = nullptr;
    bool                opened   = false;
    int                 backends = 0;
};

static hipep_device & hipep_device_of(int device) {
    static hipep_device devices[GGML_CUDA_MAX_DEVICES];
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    return devices[device];
}

static hipep_backend_ctx * hipep_open(int device) {
    hipep_device &              d = hipep_device_of(device);
    std::lock_guard<std::mutex> lock(d.mutex);
    if (!d.opened) {
        d.opened = true;
        if (hipep_backend_abi_version() != HIPEP_BACKEND_ABI_VERSION) {
            GGML_LOG_ERROR("%s: libhipep_backend is of ABI %d, not %d: ggml-hip runs every split\n", __func__,
                           hipep_backend_abi_version(), HIPEP_BACKEND_ABI_VERSION);
        } else {
            d.sdk = hipep_backend_open(device, nullptr);
            if (d.sdk == nullptr) {
                GGML_LOG_ERROR("%s: %s: ggml-hip runs every split\n", __func__, hipep_backend_last_error(nullptr));
            }
        }
    }
    return d.sdk;
}

// a run of a split's nodes, the graph's nodes [begin, end): what the library compiles, or what it does not
struct hipep_segment {
    int                  begin    = 0;
    int                  end      = 0;
    bool                 compiled = false;
    // compiled: its nodes and what it hands on, the tensors it reads, its code
    hipep_split_nodes    split;
    hipep_split_tensors  tensors;
    std::vector<void *>  data;
    hipep_backend_code * code   = nullptr;
    // compiling asked for, or the code cannot be had or failed: the library is not asked again
    bool                 asked  = false;
    bool                 failed = false;
    std::string          ops;
};

// a split as cut before, and what told it apart
struct hipep_memo {
    std::vector<uintptr_t>     fingerprint;
    std::vector<hipep_segment> segments;
};

struct hipep_profile {
    int64_t runs    = 0;
    double  seconds = 0;
};

// a ggml-hip backend's
struct hipep_state {
    int                 device = 0;
    hipep_backend_ctx * sdk    = nullptr;

    std::unordered_map<const ggml_cgraph *, hipep_memo> memos;

    int64_t splits = 0, compiled = 0, fallback = 0;
    double  wait_seconds = 0, run_seconds = 0, fallback_seconds = 0, collect_seconds = 0;
    std::map<std::string, hipep_profile> profile;
};

static hipep_state * hipep_state_of(int device, void ** state) {
    if (*state == nullptr) {
        auto * s   = new hipep_state;
        s->device  = device;
        s->sdk     = hipep_open(device);
        hipep_device &              d = hipep_device_of(device);
        std::lock_guard<std::mutex> lock(d.mutex);
        ++d.backends;
        *state = s;
    }
    return static_cast<hipep_state *>(*state);
}

bool ggml_cuda_hipep_enabled() {
    return hipep_opts().enabled;
}

bool ggml_cuda_hipep_reads_split_outputs() {
    return true;
}

void ggml_cuda_hipep_free(void * state) {
    auto * s = static_cast<hipep_state *>(state);
    if (s == nullptr) {
        return;
    }
    if (hipep_opts().log || hipep_opts().profile) {
        fprintf(stderr, "ggml-hip/hip-ep: %lld splits, %lld runs by compiled code, %lld by ggml-hip; collecting %.3f s, waiting %.3f s, compiled %.3f s, ggml-hip %.3f s\n",
                (long long) s->splits, (long long) s->compiled, (long long) s->fallback, s->collect_seconds,
                s->wait_seconds, s->run_seconds, s->fallback_seconds);
    }
    if (hipep_opts().profile) {
        std::vector<std::pair<std::string, hipep_profile>> rows(s->profile.begin(), s->profile.end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second.seconds > b.second.seconds; });
        for (const auto & [what, p] : rows) {
            fprintf(stderr, "ggml-hip/hip-ep profile: %7lld runs %8.3f s %8.1f us/run  %s\n", (long long) p.runs,
                    p.seconds, 1e6 * p.seconds / std::max<int64_t>(p.runs, 1), what.c_str());
        }
    }
    {
        hipep_device &              d = hipep_device_of(s->device);
        std::lock_guard<std::mutex> lock(d.mutex);
        if (--d.backends == 0 && d.sdk != nullptr) {
            hipep_backend_close(d.sdk);
            d.sdk    = nullptr;
            d.opened = false;
        }
    }
    delete s;
}

// the split as it is now: what tells it from another
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

static bool hipep_is_view(const ggml_tensor * node) {
    return node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
           node->op == GGML_OP_TRANSPOSE;
}

static std::string hipep_ops_of(const hipep_split_nodes & split) {
    std::map<std::string, int> ops;
    for (const ggml_tensor * node : split.nodes) {
        if (!hipep_is_view(node)) {
            ++ops[ggml_op_desc(node)];
        }
    }
    std::string text;
    for (const auto & [op, n] : ops) {
        text += (text.empty() ? "" : " ") + op + (n > 1 ? "x" + std::to_string(n) : "");
    }
    return text;
}

static hipep_segment hipep_make_segment(int begin, int end, bool compiled) {
    hipep_segment segment;
    segment.begin    = begin;
    segment.end      = end;
    segment.compiled = compiled;
    return segment;
}

// nodes [begin, end) of `graph` as a graph of their own (ggml_graph_view, which ggml-base does not export)
static ggml_cgraph hipep_subgraph(ggml_cgraph * graph, int begin, int end) {
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

// the split cut into runs of what the library compiles and what it does not. a view goes with the run it is
// in. a compiled run hands on what the split does and what a later run reads
static std::vector<hipep_segment> hipep_segments_of(hipep_backend_ctx * sdk, const ggml_cgraph & graph) {
    std::vector<hipep_segment> segments;
    const hipep_split_nodes    whole = hipep_split_nodes_of(graph);
    if (sdk == nullptr || !whole.computable || whole.nodes.empty()) {
        segments.push_back(hipep_make_segment(0, graph.n_nodes, false));
        return segments;
    }
    std::unordered_map<const ggml_tensor *, bool> exported;
    for (size_t i = 0; i < whole.nodes.size(); ++i) {
        exported.emplace(whole.nodes[i], whole.exported[i]);
    }
    // per node: compiled (1) or not (0); a view, or a node nothing computes (-1), goes with the run it is in
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
            segments.push_back(hipep_make_segment(i, i + 1, kind == 1));
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
                segment.split.exported.push_back(found->second || (!hipep_is_view(node) && read_after.count(node) != 0));
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

// a graph only split - a probe of where a fused op goes - never runs: its splits are not compiled
void ggml_cuda_hipep_precompile(int device, void ** state, ggml_cgraph * cgraph,
                                const ggml_backend_graph_optimize_params * params) {
    if (!hipep_opts().enabled || !hipep_opts().precompile || (params != nullptr && params->split_only)) {
        return;
    }
    hipep_state * s = hipep_state_of(device, state);
    if (s->splits > 0 || s->sdk == nullptr) {
        return;
    }
    for (const hipep_segment & segment : hipep_segments_of(s->sdk, *cgraph)) {
        if (!segment.compiled || segment.failed) {
            continue;
        }
        const hipep_backend_split_desc desc = segment.tensors.desc();
        if (hipep_backend_find(s->sdk, &desc) == nullptr) {
            hipep_backend_compile(s->sdk, &desc, /*wait=*/true);
        }
    }
}

// one compiled run, by its code: false when there is none (yet) or it failed
static bool hipep_run_compiled(hipep_state * s, hipep_segment & segment, void * stream, std::string & why) {
    if (segment.code == nullptr && !segment.failed) {
        const hipep_backend_split_desc desc = segment.tensors.desc();
        segment.code                        = hipep_backend_find(s->sdk, &desc);
        if (segment.code == nullptr && !segment.asked) {
            segment.code  = hipep_backend_compile(s->sdk, &desc, hipep_opts().wait);
            segment.asked = true;
        }
        if (segment.code == nullptr) {
            why = hipep_backend_last_error(s->sdk);
        }
    }
    if (segment.code == nullptr) {
        return false;
    }
    for (size_t i = 0; i < segment.data.size(); ++i) {
        segment.data[i] = segment.tensors.originals[i]->data;
    }
    // the code runs on the library's stream, and returns when it is done: the work before it is waited for
    using clock                   = std::chrono::steady_clock;
    const clock::time_point start = clock::now();
    (void) hipStreamSynchronize(static_cast<hipStream_t>(stream));
    s->wait_seconds += std::chrono::duration<double>(clock::now() - start).count();
    if (hipep_backend_run(segment.code, segment.data.data(), nullptr) != 0) {
        // the code ran and failed: never again
        why            = hipep_backend_last_error(s->sdk);
        segment.code   = nullptr;
        segment.failed = true;
        GGML_LOG_ERROR("%s: split of %s: %s\n", __func__, segment.ops.c_str(), why.c_str());
        return false;
    }
    return true;
}

bool ggml_cuda_hipep_graph_compute(int device, void * stream, void ** state, ggml_cgraph * cgraph,
                                   const std::function<void(ggml_cgraph *)> & run_rest) {
    if (!hipep_opts().enabled || cgraph->n_nodes == 0) {
        return false;
    }
    using clock      = std::chrono::steady_clock;
    const auto since = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };

    hipep_state * s = hipep_state_of(device, state);
    if (s->sdk == nullptr) {
        return false;
    }
    clock::time_point      start       = clock::now();
    std::vector<uintptr_t> fingerprint = hipep_fingerprint(*cgraph);
    hipep_memo &           memo        = s->memos[cgraph];
    if (memo.fingerprint != fingerprint) {
        memo.fingerprint = std::move(fingerprint);
        memo.segments    = hipep_segments_of(s->sdk, *cgraph);
    }
    s->collect_seconds += since(start);
    if (std::none_of(memo.segments.begin(), memo.segments.end(),
                     [](const hipep_segment & segment) { return segment.compiled && !segment.failed; })) {
        return false;
    }
    ++s->splits;

    const bool timed = hipep_opts().log || hipep_opts().profile;
    for (hipep_segment & segment : memo.segments) {
        start = clock::now();
        std::string why;
        const bool  ran = segment.compiled && hipep_run_compiled(s, segment, stream, why);
        if (!ran) {
            ggml_cgraph rest = hipep_subgraph(cgraph, segment.begin, segment.end);
            run_rest(&rest);
            if (timed) {
                (void) hipStreamSynchronize(static_cast<hipStream_t>(stream));
            }
        }
        const double seconds = since(start);
        ++(ran ? s->compiled : s->fallback);
        (ran ? s->run_seconds : s->fallback_seconds) += seconds;
        if (hipep_opts().profile) {
            hipep_profile & p = s->profile[std::string(ran ? "compiled " : "ggml-hip ") +
                                           (segment.compiled ? segment.ops : std::string("(not compiled)"))];
            ++p.runs;
            p.seconds += seconds;
        }
        if (hipep_opts().log) {
            fprintf(stderr, "ggml-hip/hip-ep: run (%d nodes): %s%s%s\n", segment.end - segment.begin,
                    ran ? "compiled" : "ggml-hip", why.empty() ? "" : ": ", why.c_str());
        }
    }
    return true;
}
