#include "split-desc.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstring>
#include <unordered_set>

static_assert(GGML_MAX_SRC == 10, "hipep_backend_tensor_desc has 10 sources");
static_assert(GGML_MAX_OP_PARAMS == 16 * sizeof(int32_t), "hipep_backend_tensor_desc has 16 op_params");

static bool is_view(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool is_weight(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static uint8_t role_of(const ggml_tensor * t) {
    if (is_weight(t)) {
        return HIPEP_BACKEND_ROLE_WEIGHT;
    }
    if (t->op == GGML_OP_NONE && (t->flags & GGML_TENSOR_FLAG_INPUT) != 0) {
        return HIPEP_BACKEND_ROLE_INPUT;
    }
    if (t->op == GGML_OP_NONE && t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return HIPEP_BACKEND_ROLE_KV_CACHE;
    }
    return HIPEP_BACKEND_ROLE_NODE;
}

static uint32_t ggml_version_number() {
    static const uint32_t version = [] {
        unsigned major = 0, minor = 0, patch = 0;
        std::sscanf(ggml_version(), "%u.%u.%u", &major, &minor, &patch);
        return major * 10000 + minor * 100 + patch;
    }();
    return version;
}

// a tensor as a leaf: what it is, not what made it
static hipep_backend_tensor_desc leaf_of(const ggml_tensor * t) {
    hipep_backend_tensor_desc d = {};
    d.op   = "NONE";
    d.name = t->name;
    d.type = t->type;
    for (int i = 0; i < 4; ++i) {
        d.ne[i] = t->ne[i];
        d.nb[i] = t->nb[i];
    }
    for (int32_t & s : d.src) {
        s = -1;
    }
    d.view_src = -1;
    d.flags    = t->flags;
    d.role     = role_of(t);
    return d;
}

// what made it: its operation, parameters, sources and the tensor it views
static void set_op(hipep_backend_tensor_desc & d, const ggml_tensor * t, const int32_t * src, int32_t view_src) {
    d.op = ggml_op_name(t->op);
    std::memcpy(d.op_params, t->op_params, sizeof(d.op_params));
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        d.src[i] = src[i];
    }
    d.view_src  = view_src;
    d.view_offs = t->view_offs;
}

hipep_backend_split_desc hipep_split_tensors::desc() const {
    hipep_backend_split_desc d = {};
    d.abi          = HIPEP_BACKEND_ABI_VERSION;
    d.tensor_size  = sizeof(hipep_backend_tensor_desc);
    d.ggml_version = ggml_version_number();
    d.tensors      = tensors.data();
    d.n_tensors    = tensors.size();
    return d;
}

hipep_split_nodes hipep_split_nodes_of(const ggml_cgraph & split) {
    hipep_split_nodes result;
    std::unordered_set<const ggml_tensor *> empty;
    for (int i = 0; i < split.n_nodes; ++i) {
        const ggml_tensor * node = split.nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (ggml_nelements(node) == 0) {
            empty.insert(node);
            continue;
        }
        for (const ggml_tensor * src : node->src) {
            if (src != nullptr && empty.count(src) != 0) {
                result.computable = false;
            }
        }
        result.nodes.push_back(node);
    }
    const std::vector<const ggml_tensor *> & nodes = result.nodes;
    result.exported.assign(nodes.size(), false);
    // a split ending in one of the graph's outputs ends the graph (llama.cpp's
    // logits come last): nothing after it reads what it leaves, but its
    // outputs and the caches it writes
    const bool last = !nodes.empty() && (nodes.back()->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const ggml_tensor * n = nodes[i];
        if (is_view(n->op)) {
            continue;
        }
        bool         reused = false;
        const char * begin  = static_cast<const char *>(n->data);
        const char * end    = begin + ggml_nbytes(n);
        for (size_t j = i + 1; j < nodes.size() && !reused; ++j) {
            const ggml_tensor * m = nodes[j];
            if (is_view(m->op) || m->data == nullptr || m->buffer != n->buffer) {
                continue;
            }
            const char * b = static_cast<const char *>(m->data);
            reused         = b < end && begin < b + ggml_nbytes(m);
        }
        result.exported[i] = (n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                             (!reused && (!last || n->op == GGML_OP_SET_ROWS));
    }
    return result;
}

namespace {

struct collector {
    std::unordered_set<const ggml_tensor *>          nodes;
    hipep_split_tensors &                            out;
    std::unordered_map<const ggml_tensor *, int32_t> index;
    bool                                             ok = true;

    int32_t add(const ggml_tensor * t) {
        if (t == nullptr) {
            return -1;
        }
        if (auto found = index.find(t); found != index.end()) {
            return found->second;
        }
        if (nodes.count(t) != 0) {
            // read before the split computes it
            ok = false;
            return -1;
        }
        // from outside: a view of an outside tensor stays a view of what it
        // views (a cache read through a window), anything else is a leaf
        if (t->view_src != nullptr && t->op != GGML_OP_NONE) {
            return collect(t, false, false);
        }
        return push(t, leaf_of(t));
    }

    int32_t collect(const ggml_tensor * t, bool node, bool exported) {
        int32_t src[GGML_MAX_SRC];
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            // an operation that is its own operand (CPY's destination) names itself
            src[i] = t->src[i] == t ? -2 : add(t->src[i]);
        }
        const int32_t view_src = add(t->view_src);
        hipep_backend_tensor_desc d = leaf_of(t);
        set_op(d, t, src, view_src);
        d.in_split = node;
        d.exported = node && exported;
        return push(t, d);
    }

    int32_t push(const ggml_tensor * t, hipep_backend_tensor_desc d) {
        const int32_t i = static_cast<int32_t>(out.tensors.size());
        for (int32_t & s : d.src) {
            if (s == -2) {
                s = i;
            }
        }
        out.tensors.push_back(d);
        out.originals.push_back(t);
        index.emplace(t, i);
        return i;
    }
};

} // namespace

bool hipep_collect_split(const hipep_split_nodes & split, hipep_split_tensors & result) {
    result = {};
    collector d{ { split.nodes.begin(), split.nodes.end() }, result, {} };
    for (size_t i = 0; i < split.nodes.size() && d.ok; ++i) {
        d.collect(split.nodes[i], true, split.exported[i]);
    }
    return d.ok;
}

void hipep_collect_op(const ggml_tensor * op, hipep_split_tensors & result) {
    result = {};
    int32_t src[GGML_MAX_SRC];
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        src[i] = -1;
        if (op->src[i] != nullptr && op->src[i] != op) {
            src[i] = static_cast<int32_t>(result.tensors.size());
            result.tensors.push_back(leaf_of(op->src[i]));
            result.originals.push_back(op->src[i]);
        }
    }
    int32_t view_src = -1;
    if (op->view_src != nullptr) {
        view_src = static_cast<int32_t>(result.tensors.size());
        result.tensors.push_back(leaf_of(op->view_src));
        result.originals.push_back(op->view_src);
    }
    hipep_backend_tensor_desc d = leaf_of(op);
    set_op(d, op, src, view_src);
    d.in_split = 1;
    result.tensors.push_back(d);
    result.originals.push_back(op);
}
