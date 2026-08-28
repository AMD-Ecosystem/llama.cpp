#include "dispatch-binary.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kBinaryF32Kernel   = GGML_HRX_KERNEL_REF("loom_libs", "ggml_binary_f32");
static constexpr KernelCatalogRef kBinaryBcF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_binary_bc_f32");

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool positive_shape(const Value & value) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0) {
            return false;
        }
    }
    return value.element_count > 0;
}

static bool packed_f32_layout(const Value & value) {
    size_t expected_stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.nb[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool distinct_storage(const Value & lhs, const Value & rhs, const Value & output) {
    return lhs.storage != rhs.storage && lhs.storage != output.storage && rhs.storage != output.storage;
}

static bool supported_source_layout(const Graph & graph, const Value & value) {
    if (value.alias_source.value < 0) {
        return true;
    }
    if (value.storage_offset != 0) {
        return false;
    }
    const GraphNode * producer = graph.index().producer(value.id);
    return producer != nullptr && producer->op == GGML_OP_RESHAPE;
}

static bool broadcastable_to(const Value & source, const Value & output) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (source.ne[i] != output.ne[i] && source.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static bool binary_kind_allows_broadcast(BinaryKind kind, const Value & lhs, const Value & rhs, const Value & output) {
    const bool lhs_full = same_shape(lhs, output);
    const bool rhs_full = same_shape(rhs, output);
    if (!broadcastable_to(lhs, output) || !broadcastable_to(rhs, output) || (!lhs_full && !rhs_full)) {
        return false;
    }

    switch (kind) {
        case BinaryKind::Add:
        case BinaryKind::Mul:
            return true;
        case BinaryKind::Sub:
        case BinaryKind::Div:
            return lhs_full;
        case BinaryKind::SwiGLU:
            return lhs_full && rhs_full;
    }
    return false;
}

static uint32_t broadcast_dim_flag(const Value & source, const Value & output, int dim) {
    return source.ne[dim] == 1 && output.ne[dim] != 1 ? 1 : 0;
}

static void add_binary_shape_parameters(Dispatch &    dispatch,
                                        const Value & lhs,
                                        const Value & rhs,
                                        const Value & output) {
    dispatch.kernel.integer_parameters.emplace("element_count", output.element_count);
    dispatch.kernel.integer_parameters.emplace("ne0", output.ne[0]);
    dispatch.kernel.integer_parameters.emplace("ne1", output.ne[1]);
    dispatch.kernel.integer_parameters.emplace("ne2", output.ne[2]);
    dispatch.kernel.integer_parameters.emplace("ne3", output.ne[3]);
    dispatch.kernel.integer_parameters.emplace("src0_element_count", lhs.element_count);
    dispatch.kernel.integer_parameters.emplace("src1_element_count", rhs.element_count);
}

static void add_broadcast_config(Dispatch &    dispatch,
                                 const char *  config_prefix,
                                 const char *  source_prefix,
                                 const Value & source,
                                 const Value & output) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.compile_parameters.emplace(
            std::string(config_prefix) + source_prefix + "_broadcast_dim" + std::to_string(i),
            std::to_string(broadcast_dim_flag(source, output, i)));
    }
}

static void bind_binary_buffers(Dispatch & dispatch, const Value & lhs, const Value & rhs, const Value & output) {
    dispatch.bindings.push_back({ lhs.id, 0, lhs.byte_count });
    dispatch.bindings.push_back({ rhs.id, 0, rhs.byte_count });
    dispatch.bindings.push_back({ output.id, 0, output.byte_count });
}

static bool match_binary_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 2) {
        return false;
    }

    const BinaryParams * params = op_params_as<BinaryParams>(node->params);
    if (params == nullptr || !binary_kind_supported(params->op)) {
        return false;
    }

    const Value * output = graph_value(context.graph, node->output);
    const Value * lhs    = graph_value(context.graph, node->inputs[0]);
    const Value * rhs    = graph_value(context.graph, node->inputs[1]);
    if (output == nullptr || lhs == nullptr || rhs == nullptr) {
        return false;
    }

    if (output->type != GGML_TYPE_F32 || lhs->type != GGML_TYPE_F32 || rhs->type != GGML_TYPE_F32 ||
        !positive_shape(*output) || !output->contiguous || !lhs->contiguous || !rhs->contiguous ||
        !packed_f32_layout(*output) || !packed_f32_layout(*lhs) || !packed_f32_layout(*rhs) ||
        output->alias_source.value >= 0 || !supported_source_layout(context.graph, *lhs) ||
        !supported_source_layout(context.graph, *rhs) || !distinct_storage(*lhs, *rhs, *output) ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    if (same_shape(*lhs, *output) && same_shape(*rhs, *output)) {
        dispatch.kernel = make_kernel_specialization(kBinaryF32Kernel);
        dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
        dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.op",
                                                   std::to_string(binary_kind_config_value(params->op)));
    } else {
        if (!binary_kind_allows_broadcast(params->op, *lhs, *rhs, *output)) {
            return false;
        }
        dispatch.kernel = make_kernel_specialization(kBinaryBcF32Kernel);
        add_binary_shape_parameters(dispatch, *lhs, *rhs, *output);
        dispatch.kernel.compile_parameters.emplace("ggml.binary_bc_f32.op",
                                                   std::to_string(binary_kind_config_value(params->op)));
        add_broadcast_config(dispatch, "ggml.binary_bc_f32.", "src0", *lhs, *output);
        add_broadcast_config(dispatch, "ggml.binary_bc_f32.", "src1", *rhs, *output);
    }
    bind_binary_buffers(dispatch, *lhs, *rhs, *output);

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static void register_binary_dispatch_for(DispatchRegistryBuilder & registry, ggml_op root_op) {
    registry.add({
        "common.binary_f32",
        root_op,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_binary_f32_dispatch,
    });
}

}  // namespace

void register_binary_dispatch(DispatchRegistryBuilder & registry) {
    register_binary_dispatch_for(registry, GGML_OP_ADD);
    register_binary_dispatch_for(registry, GGML_OP_SUB);
    register_binary_dispatch_for(registry, GGML_OP_MUL);
    register_binary_dispatch_for(registry, GGML_OP_DIV);
    register_binary_dispatch_for(registry, GGML_OP_GLU);
}

}  // namespace ggml::hrx
