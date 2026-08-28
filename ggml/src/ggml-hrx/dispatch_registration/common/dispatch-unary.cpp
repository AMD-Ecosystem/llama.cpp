#include "dispatch-unary.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kUnaryF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_unary_f32");

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

static bool match_unary_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }

    const UnaryParams * params = op_params_as<UnaryParams>(node->params);
    if (params == nullptr || !unary_kind_supported(params->op)) {
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
    dispatch.kernel = make_kernel_specialization(kUnaryF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.unary_f32.op",
                                               std::to_string(unary_kind_config_value(params->op)));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
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
    register_unary_dispatch_for(registry, GGML_OP_UNARY);
    register_unary_dispatch_for(registry, GGML_OP_SQR);
    register_unary_dispatch_for(registry, GGML_OP_SQRT);
    register_unary_dispatch_for(registry, GGML_OP_LOG);
    register_unary_dispatch_for(registry, GGML_OP_SIN);
    register_unary_dispatch_for(registry, GGML_OP_COS);
}

}  // namespace ggml::hrx
