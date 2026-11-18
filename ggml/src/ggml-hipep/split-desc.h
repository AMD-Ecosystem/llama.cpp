#pragma once

// a split's tensors for libhipep_backend (hipep_backend_split_desc): their
// metadata, never their data

#include "ggml.h"

#include <hipep_backend.h>

#include <vector>

// which of a split's nodes are computed, in order, and which of them are read
// after it: a node is, unless ggml-alloc gave its memory to a later node of
// the split (it reuses memory only after its last reader) - and in a split
// ending in a graph output (llama.cpp's logits, last), only the graph's
// outputs and the caches it writes are. not computed: a node of a branch the
// graph does not take, or one of no elements. `computable` is false when a
// computed node reads one of no elements.
struct hipep_split_nodes {
    std::vector<const ggml_tensor *> nodes;
    std::vector<bool>                exported;
    bool                             computable = true;
};

hipep_split_nodes hipep_split_nodes_of(const ggml_cgraph & split);

// the tensors of a hipep_backend_split_desc, and the ggml tensors they are
struct hipep_split_tensors {
    std::vector<hipep_backend_tensor_desc> tensors;
    std::vector<const ggml_tensor *>       originals;

    hipep_backend_split_desc desc() const;
};

// the split: for each node, in order, the tensors it reads not collected yet
// (a view of a tensor outside the split with what it views), then the node.
// false when a node is read before the split computes it.
bool hipep_collect_split(const hipep_split_nodes & split, hipep_split_tensors & result);

// one operation, its sources and the tensor it views, for supports
void hipep_collect_op(const ggml_tensor * op, hipep_split_tensors & result);
