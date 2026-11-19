#include "dispatch-unary.h"

#include "dispatch-common.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kUnaryF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_unary_f32");
static constexpr KernelCatalogRef kSoftPlusF32Kernel  = GGML_HRX_KERNEL_REF("loom_libs", "ggml_softplus_f32");
static constexpr KernelCatalogRef kScaleBiasF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_scale_bias_f32");
static constexpr KernelCatalogRef kClampF32Kernel     = GGML_HRX_KERNEL_REF("loom_libs", "ggml_clamp_f32");

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool scale_storage_is_safe(const Value & input, const Value & output) {
    if (input.storage_root != output.storage_root) {
        return true;
    }
    if (input.storage_offset == output.storage_offset && input.byte_count == output.byte_count) {
        return true;
    }
    if (input.storage_offset > std::numeric_limits<size_t>::max() - input.byte_count ||
        output.storage_offset > std::numeric_limits<size_t>::max() - output.byte_count) {
        return false;
    }
    return input.storage_offset + input.byte_count <= output.storage_offset ||
           output.storage_offset + output.byte_count <= input.storage_offset;
}

static bool match_scale_bias_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }

    const ScaleParams * params = op_params_as<ScaleParams>(node->params);
    const Value *       input  = graph_value(context.graph, node->inputs[0]);
    const Value *       output = graph_value(context.graph, node->output);
    if (params == nullptr || input == nullptr || output == nullptr || !std::isfinite(params->scale) ||
        std::isnan(params->bias) || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input->contiguous || !output->contiguous || !same_shape(*input, *output) ||
        !scale_storage_is_safe(*input, *output) || output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kScaleBiasF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.scale.scale", float_config_value(params->scale));
    dispatch.kernel.compile_parameters.emplace("ggml.scale.bias", float_config_value(params->bias));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_unary_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }

    const UnaryParams * params = op_params_as<UnaryParams>(node->params);
    if (params == nullptr || (!unary_kind_supported(params->op) && params->op != UnaryKind::SoftPlus)) {
        return false;
    }

    const Value * output = graph_value(context.graph, node->output);
    const Value * input  = graph_value(context.graph, node->inputs[0]);
    if (output == nullptr || input == nullptr) {
        return false;
    }

    if (output->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || !same_shape(*output, *input) ||
        !output->contiguous || !input->contiguous || output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel =
        make_kernel_specialization(params->op == UnaryKind::SoftPlus ? kSoftPlusF32Kernel : kUnaryF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    if (params->op != UnaryKind::SoftPlus) {
        dispatch.kernel.compile_parameters.emplace("ggml.unary_f32.op",
                                                   std::to_string(unary_kind_config_value(params->op)));
    }
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_clamp_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }
    const auto *  params = op_params_as<ClampParams>(node->params);
    const Value * input  = graph_value(context.graph, node->inputs[0]);
    const Value * output = graph_value(context.graph, node->output);
    if (params == nullptr || input == nullptr || output == nullptr || std::isnan(params->min) ||
        std::isnan(params->max) || params->min > params->max || input->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !input->contiguous || !output->contiguous || !same_shape(*input, *output) ||
        !scale_storage_is_safe(*input, *output) || output->element_count < 1 || output->element_count > 134217728) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kClampF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.clamp.min", float_config_value(params->min));
    dispatch.kernel.compile_parameters.emplace("ggml.clamp.max", float_config_value(params->max));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_fill(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_FILL) {
        return false;
    }
    const auto *  params = op_params_as<FillParams>(node->params);
    const Value * output = context.graph.values().find(node->output);
    if (params == nullptr || output == nullptr || std::isnan(params->value) || !output->contiguous ||
        (output->type != GGML_TYPE_F32 && output->type != GGML_TYPE_F16) || output->element_count < 1 ||
        output->element_count > 134217728) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_fill"));
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.fill.value", float_config_value(params->value));
    dispatch.kernel.compile_parameters.emplace("ggml.fill.format", output->type == GGML_TYPE_F16 ? "16" : "32");
    dispatch.bindings = {
        { output->id, 0, output->byte_count }
    };
    match.dispatches.push_back(std::move(dispatch));
    match.covered_nodes.push_back(context.root_index);
    return true;
}

static void register_unary_dispatch_for(DispatchRegistryBuilder & registry, ggml_op root_op) {
    registry.add({
        "common.unary_f32",
        root_op,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_unary_f32_dispatch,
    });
}

}  // namespace

void register_unary_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({ "common.fill", GGML_OP_FILL, DispatchMatchKind::SingleOp, 0, DispatchSource::Common, match_fill });
    registry.add({
        "common.clamp_f32",
        GGML_OP_CLAMP,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_clamp_f32_dispatch,
    });
    registry.add({
        "common.scale_bias_f32",
        GGML_OP_SCALE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_scale_bias_f32_dispatch,
    });
    register_unary_dispatch_for(registry, GGML_OP_UNARY);
    register_unary_dispatch_for(registry, GGML_OP_SQR);
    register_unary_dispatch_for(registry, GGML_OP_SQRT);
    register_unary_dispatch_for(registry, GGML_OP_LOG);
    register_unary_dispatch_for(registry, GGML_OP_SIN);
    register_unary_dispatch_for(registry, GGML_OP_COS);
}

}  // namespace ggml::hrx
