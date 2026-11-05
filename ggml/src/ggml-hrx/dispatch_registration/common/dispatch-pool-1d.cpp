#include "dispatch-pool-1d.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kFullMaxPoolF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_pool_1d_full_max_f32");

static constexpr KernelCatalogRef kAmaxScaleF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_amax_scale_f32");

static bool matches_unary(const GraphNode & node, UnaryKind kind) {
    const UnaryParams * params = op_params_as<UnaryParams>(node.params);
    if (params == nullptr) {
        return false;
    }
    return params->op == kind;
}

static bool matches_scale(const GraphNode & node, float factor) {
    const ScaleParams * params = op_params_as<ScaleParams>(node.params);
    if (params == nullptr) {
        return false;
    }
    const bool same_factor = params->scale == factor;
    const bool no_bias = params->bias == 0.0f;
    return same_factor && no_bias;
}

static bool match_amax_scale_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const Graph & graph = context.graph;
    // Intermediate values must stay private: this dispatch publishes only x/s and s.
    const ggml_op ops[] = { GGML_OP_UNARY, GGML_OP_POOL_1D, GGML_OP_SCALE, GGML_OP_LOG,
                           GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_CLAMP, GGML_OP_SCALE, GGML_OP_UNARY };
    std::vector<const GraphNode *> chain;
    for (size_t i = 0; i < 9; ++i) {
        const bool right_op = node->op == ops[i];
        const bool one_input = node->inputs.size() == 1;
        const bool expected_node = right_op && one_input;
        if (!expected_node) {
            return false;
        }
        const Value * value = graph.values().find(node->output);
        if (value == nullptr) {
            return false;
        }
        const bool f32 = value->type == GGML_TYPE_F32;
        const bool contiguous = value->contiguous;
        const bool supported_value = f32 && contiguous;
        if (!supported_value) {
            return false;
        }
        chain.push_back(node);
        if (i == 8) {
            break;
        }
        const auto & consumers = graph.index().consumers(node->output);
        const bool private_value = value->kind == ValueKind::Transient;
        const bool one_consumer = consumers.size() == 1;
        const bool can_elide = private_value && one_consumer;
        if (!can_elide) {
            return false;
        }
        node = consumers.front();
    }
    const bool abs_input = matches_unary(*chain[0], UnaryKind::Abs);
    const bool scale_amax = matches_scale(*chain[2], 1.0f / 1024.0f);
    const bool logarithm = matches_unary(*chain[3], UnaryKind::Log);
    const bool scale_log = matches_scale(*chain[4], 1.4426950408889634f);
    const bool ceil_log = matches_unary(*chain[5], UnaryKind::Ceil);
    const bool scale_exp = matches_scale(*chain[7], 0.6931471805599453f);
    const bool exponential = matches_unary(*chain[8], UnaryKind::Exp);
    const bool exact_math = abs_input && scale_amax && logarithm && scale_log && ceil_log && scale_exp && exponential;
    if (!exact_math) {
        return false;
    }
    const Pool1DParams * pool = op_params_as<Pool1DParams>(chain[1]->params);
    const ClampParams * clamp = op_params_as<ClampParams>(chain[6]->params);
    const Value * input = graph.values().find(chain[0]->inputs[0]);
    const bool has_pool = pool != nullptr;
    const bool has_clamp = clamp != nullptr;
    const bool has_input = input != nullptr;
    const bool all_present = has_pool && has_clamp && has_input;
    if (!all_present) {
        return false;
    }
    const bool max_pool = pool->op == GGML_OP_POOL_MAX;
    const bool full_width = pool->kernel == input->ne[0];
    const bool full_stride = pool->stride == pool->kernel;
    const bool no_padding = pool->padding == 0;
    const bool clamp_min = clamp->min == 0.0f;
    const bool clamp_max = clamp->max == 127.0f;
    const bool input_f32 = input->type == GGML_TYPE_F32;
    const bool bounded_width = input->ne[0] > 0 && input->ne[0] <= 32768;
    const bool bounded_elements = input->element_count > 0 && input->element_count <= 134217728;
    const bool supported = max_pool && full_width && full_stride && no_padding && clamp_min && clamp_max &&
                           input_f32 && input->contiguous && bounded_width && bounded_elements;
    if (!supported) {
        return false;
    }
    const Value * scale = graph.values().find(chain.back()->output);
    const int64_t rows = input->element_count / input->ne[0];
    const bool bounded_rows = rows <= 1048576;
    if (!bounded_rows) {
        return false;
    }
    for (size_t i = 0; i < chain.size(); ++i) {
        const Value * value = graph.values().find(chain[i]->output);
        auto expected_shape = input->ne;
        if (i > 0) {
            expected_shape[0] = 1;
        }
        const bool same_shape = value->ne == expected_shape;
        const bool separate_input = value->storage_root != input->storage_root;
        const bool valid_output = same_shape && separate_input;
        if (!valid_output) {
            return false;
        }
    }
    const GraphNode * division = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(scale->id)) {
        const bool is_div = consumer->op == GGML_OP_DIV;
        const bool two_inputs = consumer->inputs.size() == 2;
        const bool candidate = is_div && two_inputs;
        if (!candidate) {
            continue;
        }
        const bool numerator = consumer->inputs[0] == input->id;
        const bool denominator = consumer->inputs[1] == scale->id;
        const bool exact_division = numerator && denominator;
        if (exact_division) {
            division = consumer;
            break;
        }
    }
    if (division == nullptr) {
        return false;
    }
    const Value * output = graph.values().find(division->output);
    if (output == nullptr) {
        return false;
    }
    const bool output_f32 = output->type == GGML_TYPE_F32;
    const bool same_shape = output->ne == input->ne;
    const bool separate_input = output->storage_root != input->storage_root;
    const bool separate_scale = output->storage_root != scale->storage_root;
    const bool valid_output = output_f32 && output->contiguous && same_shape && separate_input && separate_scale;
    if (!valid_output) {
        return false;
    }
    chain.push_back(division);
    for (const GraphNode * covered : chain) {
        if (!append_covered_node_index_once(graph, context.covered_nodes, covered, match.covered_nodes)) {
            return false;
        }
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAmaxScaleF32Kernel);
    dispatch.kernel.integer_parameters.emplace("row_count", rows);
    dispatch.kernel.integer_parameters.emplace("row_width", input->ne[0]);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch.bindings.push_back({ scale->id, 0, scale->byte_count });
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_full_max_pool_1d_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool one_input = node->inputs.size() == 1;
    if (!one_input) {
        return false;
    }

    const Pool1DParams * params = op_params_as<Pool1DParams>(node->params);
    const Value * input = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    const bool has_params = params != nullptr;
    const bool has_input = input != nullptr;
    const bool has_output = output != nullptr;
    const bool all_present = has_params && has_input && has_output;
    if (!all_present) {
        return false;
    }

    const bool f32_values = input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32;
    const bool contiguous_values = input->contiguous && output->contiguous;
    const bool full_row = output->ne[0] == 1 && input->ne[0] == params->kernel;
    const bool matching_rows = input->ne[1] == output->ne[1] && input->ne[2] == output->ne[2] &&
                               input->ne[3] == output->ne[3];
    const bool exact_window = params->stride == params->kernel && params->padding == 0;
    const bool bounded_width = input->ne[0] > 0 && input->ne[0] <= 32768;
    const bool bounded_rows = output->element_count > 0 && output->element_count <= 1048576;
    const bool bounded_input = input->element_count <= 134217728;
    const bool bounded = bounded_width && bounded_rows && bounded_input;
    const bool separate_storage = input->storage_root != output->storage_root;
    const bool supported = params->op == GGML_OP_POOL_MAX && f32_values && contiguous_values && full_row &&
                           matching_rows && exact_window && bounded && separate_storage;
    if (!supported) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kFullMaxPoolF32Kernel);
    dispatch.kernel.integer_parameters.emplace("row_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("row_width", input->ne[0]);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_pool_1d_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.amax_scale_f32",
        GGML_OP_UNARY,
        DispatchMatchKind::Fused,
        400,
        DispatchSource::Common,
        match_amax_scale_f32,
    });
    registry.add({
        "common.pool_1d_full_max_f32",
        GGML_OP_POOL_1D,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_full_max_pool_1d_f32,
    });
}

}  // namespace ggml::hrx
