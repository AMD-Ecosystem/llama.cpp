#include "dispatch-mul-mat.h"

#include "dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMulMatF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatF32F32DecodeWave64Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_decode_wave64");
static constexpr KernelCatalogRef kMulMatBiasF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_bias_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatAddF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_add_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatBiasAddF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_bias_add_f32_f32_wmma");

struct MulMatPostOpsMatch {
    const Value *                  input             = nullptr;
    const Value *                  weight            = nullptr;
    const Value *                  projection_output = nullptr;
    const Value *                  bias              = nullptr;
    const Value *                  residual_input    = nullptr;
    const Value *                  residual_output   = nullptr;
    const GraphNode *              bias_add_node     = nullptr;
    const GraphNode *              residual_add_node = nullptr;
    std::vector<const GraphNode *> add_nodes;
    KernelCatalogRef               kernel        = {};
    CommonMulMatWeightFormat       weight_format = CommonMulMatWeightFormat::Q4K;
    int64_t                        input_size    = 0;
    int64_t                        output_size   = 0;
    int64_t                        token_count   = 0;
    bool                           has_bias      = false;
    bool                           has_residual  = false;

    bool matched() const {
        return input != nullptr && weight != nullptr && projection_output != nullptr && residual_output != nullptr &&
               kernel.id != kUncatalogedKernelId && (has_bias || has_residual);
    }
};

static bool is_bias_shape(const Value & value, int64_t output_size) {
    if (value.type != GGML_TYPE_F32 || !value.contiguous || value.ne[0] != output_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static MulMatPostOpsMatch match_mul_mat_postops(const DispatchMatchContext & context) {
    MulMatPostOpsMatch      match;
    const CommonMulMatMatch root =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, false);
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const Value * current = root.output;
    for (int add_index = 0; add_index < 2; ++add_index) {
        const GraphNode * add_node = common_find_only_consumer_with_op(context.graph, current->id, GGML_OP_ADD);
        if (add_node == nullptr || !common_binary_node_is_add(*add_node)) {
            break;
        }

        const bool current_is_lhs = add_node->inputs[0] == current->id;
        const bool current_is_rhs = add_node->inputs[1] == current->id;
        if (!current_is_lhs && !current_is_rhs) {
            return {};
        }

        const ValueId other_id = current_is_lhs ? add_node->inputs[1] : add_node->inputs[0];
        const Value * other    = common_graph_value(context.graph, other_id);
        const Value * output   = common_graph_value(context.graph, add_node->output);
        if (other == nullptr || output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
            !common_same_shape(*root.output, *output)) {
            return {};
        }

        if (!match.has_bias && is_bias_shape(*other, root.output_size)) {
            match.bias          = other;
            match.bias_add_node = add_node;
            match.add_nodes.push_back(add_node);
            match.has_bias = true;
            current        = output;
            continue;
        }

        if (!match.has_residual && other->type == GGML_TYPE_F32 && other->contiguous &&
            common_same_shape(*root.output, *other)) {
            match.residual_input    = other;
            match.residual_output   = output;
            match.residual_add_node = add_node;
            match.add_nodes.push_back(add_node);
            match.has_residual = true;
            current            = output;
            continue;
        }

        break;
    }

    if (!match.has_bias && !match.has_residual) {
        return {};
    }

    match.residual_output = current;

    if (match.has_bias && match.has_residual) {
        match.kernel = kMulMatBiasAddF32F32WmmaKernel;
    } else if (match.has_residual) {
        match.kernel = kMulMatAddF32F32WmmaKernel;
    } else if (match.has_bias) {
        match.kernel = kMulMatBiasF32F32WmmaKernel;
    } else {
        return {};
    }

    match.input             = root.input;
    match.weight            = root.weight;
    match.projection_output = root.output;
    match.weight_format     = root.weight_format;
    match.input_size        = root.input_size;
    match.output_size       = root.output_size;
    match.token_count       = root.token_count;
    return match;
}

static bool try_match_fused_unary(const DispatchMatchContext & context, CommonMulMatMatch & match) {
    if (!match.matched()) {
        return false;
    }

    const std::vector<const GraphNode *> & consumers = context.graph.index().consumers(context.root_node->output);
    if (consumers.size() != 1 || consumers.front() == nullptr) {
        return false;
    }

    const GraphNode * unary       = consumers.front();
    size_t            unary_index = 0;
    if (!context.graph.index().node_index(unary, unary_index) || unary_index >= context.covered_nodes.size() ||
        context.covered_nodes[unary_index] || unary->inputs.size() != 1) {
        return false;
    }

    const UnaryParams * params = op_params_as<UnaryParams>(unary->params);
    if (params == nullptr || !unary_kind_supported(params->op)) {
        return false;
    }

    const Value * unary_input  = common_graph_value(context.graph, unary->inputs[0]);
    const Value * unary_output = common_graph_value(context.graph, unary->output);
    if (unary_input == nullptr || unary_output == nullptr || unary_input->id != match.output->id ||
        unary_output->type != GGML_TYPE_F32 || !unary_output->contiguous ||
        !common_same_shape(*match.output, *unary_output)) {
        return false;
    }

    match.output           = unary_output;
    match.output_unary_op  = params->op;
    match.unary_node_index = unary_index;
    match.has_fused_unary  = true;
    return true;
}

}  // namespace

static void build_mul_mat_dispatch(const CommonMulMatMatch & match, DispatchMatch & dispatch_match, size_t root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.input_size", common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_size", common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_unary_op",
                                               std::to_string(unary_kind_config_value(match.output_unary_op)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat.weight_format", common_to_config_value(common_mul_mat_format_config_value(match.weight_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    if (match.has_fused_unary) {
        dispatch_match.covered_nodes.push_back(match.unary_node_index);
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_mul_mat_postops_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const MulMatPostOpsMatch match = match_mul_mat_postops(context);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_postops.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_postops.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_postops.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.weight_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    if (match.has_bias) {
        dispatch.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
    }
    if (match.has_residual) {
        dispatch.bindings.push_back({ match.residual_input->id, 0, match.residual_input->byte_count });
    }
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    for (const GraphNode * add_node : match.add_nodes) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, add_node,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static void build_decode_mul_mat_dispatch(const CommonMulMatMatch & match,
                                          DispatchMatch &           dispatch_match,
                                          size_t                    root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
    dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.output_capacity",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_f32_f32_decode.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.weight_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, false);
    if (!match.matched()) {
        return false;
    }
    try_match_fused_unary(context, match);
    build_mul_mat_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_decode_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32DecodeWave64Kernel, true);
    if (!match.matched()) {
        return false;
    }
    build_decode_mul_mat_dispatch(match, dispatch_match, context.root_index);
    return true;
}

void register_mul_mat_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mul_mat_postops.f32_f32_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        180,
        DispatchSource::Common,
        match_mul_mat_postops_dispatch,
    });
    registry.add({
        "common.mul_mat.f32_f32_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        80,
        DispatchSource::Common,
        match_mul_mat_dispatch,
    });
    registry.add({
        "common.mul_mat.f32_f32_decode",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        60,
        DispatchSource::Common,
        match_decode_mul_mat_dispatch,
    });
}

}  // namespace ggml::hrx
