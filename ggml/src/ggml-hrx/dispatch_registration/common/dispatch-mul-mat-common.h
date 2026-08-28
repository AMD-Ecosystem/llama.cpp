#pragma once

#include "../dispatch-registry.h"
#include "dispatch-mul-mat-weight-format.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace ggml::hrx {

struct CommonMulMatMatch {
    const Value *            input            = nullptr;
    const Value *            weight           = nullptr;
    const Value *            output           = nullptr;
    KernelCatalogRef         kernel           = {};
    int64_t                  input_size       = 0;
    int64_t                  output_size      = 0;
    int64_t                  token_count      = 0;
    CommonMulMatWeightFormat weight_format    = CommonMulMatWeightFormat::Q4K;
    UnaryKind                output_unary_op  = UnaryKind::Identity;
    size_t                   unary_node_index = 0;
    bool                     has_fused_unary  = false;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

inline const Value * common_graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

inline bool common_same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

inline bool common_is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

inline bool common_is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

inline bool common_is_supported_rmsnorm_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

inline bool common_is_supported_prefill_token_count(int64_t token_count) {
    return token_count > 1 && token_count <= 2048;
}

inline bool common_is_supported_decode_token_count(int64_t token_count) {
    return token_count == 1;
}

inline bool common_is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

inline std::string common_to_config_value(int64_t value) {
    return std::to_string(value);
}

inline std::string common_to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

inline int64_t common_ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline bool common_is_weight_shape(const Value & weight, int64_t hidden_size) {
    if (weight.ne[0] != hidden_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (weight.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

inline const GraphNode * common_find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    if (!graph.has_index()) {
        return nullptr;
    }
    const GraphNode * match = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != op) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = consumer;
    }
    return match;
}

inline const GraphNode * common_find_only_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    if (!graph.has_index()) {
        return nullptr;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    if (consumers.size() != 1 || consumers.front() == nullptr || consumers.front()->op != op) {
        return nullptr;
    }
    return consumers.front();
}

inline bool common_binary_node_is_mul(const GraphNode & node) {
    if (node.op != GGML_OP_MUL || node.inputs.size() != 2) {
        return false;
    }
    const BinaryParams * binary_params = op_params_as<BinaryParams>(node.params);
    return binary_params == nullptr || binary_params->op == BinaryKind::Mul;
}

inline bool common_binary_node_is_add(const GraphNode & node) {
    if (node.op != GGML_OP_ADD || node.inputs.size() != 2) {
        return false;
    }
    const BinaryParams * binary_params = op_params_as<BinaryParams>(node.params);
    return binary_params == nullptr || binary_params->op == BinaryKind::Add;
}

inline bool common_is_swiglu_params(const OpParams & params) {
    const BinaryParams * binary_params = op_params_as<BinaryParams>(params);
    if (binary_params != nullptr) {
        return binary_params->op == BinaryKind::SwiGLU;
    }

    const GluParams * glu_params = op_params_as<GluParams>(params);
    return glu_params != nullptr && glu_params->op == GGML_GLU_OP_SWIGLU;
}

inline CommonMulMatMatch common_match_mul_mat_any_format(const Graph &     graph,
                                                         const GraphNode * node,
                                                         KernelCatalogRef  kernel,
                                                         bool              decode) {
    CommonMulMatMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = common_graph_value(graph, node->inputs[0]);
    const Value * input  = common_graph_value(graph, node->inputs[1]);
    const Value * output = common_graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !common_is_2d(*weight) || !common_is_2d(*input) ||
        !common_is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    CommonMulMatWeightFormat format = CommonMulMatWeightFormat::Q4K;
    if (!common_mul_mat_format_for_type(weight->type, format)) {
        return {};
    }

    const int64_t input_size            = weight->ne[0];
    const int64_t output_size           = weight->ne[1];
    const int64_t token_count           = input->ne[1];
    const bool    token_count_supported = decode ? common_is_supported_decode_token_count(token_count) :
                                                   common_is_supported_prefill_token_count(token_count);
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !token_count_supported || !common_is_supported_dense_input_size(input_size) ||
        !common_is_supported_dense_output_size(output_size)) {
        return {};
    }

    match.input         = input;
    match.weight        = weight;
    match.output        = output;
    match.kernel        = kernel;
    match.input_size    = input_size;
    match.output_size   = output_size;
    match.token_count   = token_count;
    match.weight_format = format;
    return match;
}

}  // namespace ggml::hrx
