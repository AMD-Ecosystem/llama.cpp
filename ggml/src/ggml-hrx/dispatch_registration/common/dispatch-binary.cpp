#include "dispatch-binary.h"

#include "dispatch-layout-utils.h"
#include "dispatch-mul-mat-common.h"
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
static constexpr KernelCatalogRef kBinarySwiGluSymmetricI4K32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_binary_swiglu_symmetric_i4_k32");

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

static bool storage_ranges_disjoint(const Value & lhs, size_t lhs_byte_count, const Value & rhs, size_t rhs_byte_count) {
    if (lhs.storage != rhs.storage) {
        return true;
    }
    if (lhs.storage_offset > std::numeric_limits<size_t>::max() - lhs_byte_count ||
        rhs.storage_offset > std::numeric_limits<size_t>::max() - rhs_byte_count) {
        return false;
    }
    return lhs.storage_offset + lhs_byte_count <= rhs.storage_offset ||
           rhs.storage_offset + rhs_byte_count <= lhs.storage_offset;
}

static bool storage_ranges_disjoint(const Value & lhs, const Value & rhs) {
    return storage_ranges_disjoint(lhs, lhs.byte_count, rhs, rhs.byte_count);
}

static bool binary_output_storage_is_safe(const Value & lhs, const Value & rhs, const Value & output) {
    return storage_ranges_disjoint(lhs, output) && storage_ranges_disjoint(rhs, output);
}

static bool binary_noalias_storage_is_safe(const Value & lhs, const Value & rhs, const Value & output) {
    return storage_ranges_disjoint(lhs, rhs) && binary_output_storage_is_safe(lhs, rhs, output);
}

static bool binary_output_storage_is_safe(const Value & lhs,
                                          size_t        lhs_byte_count,
                                          const Value & rhs,
                                          size_t        rhs_byte_count,
                                          const Value & output) {
    return storage_ranges_disjoint(lhs, lhs_byte_count, output, output.byte_count) &&
           storage_ranges_disjoint(rhs, rhs_byte_count, output, output.byte_count);
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
        case BinaryKind::GeGLU:
        case BinaryKind::RegLU:
        case BinaryKind::GeGLUErf:
        case BinaryKind::GeGLUQuick:
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

static void bind_binary_source_buffer(Dispatch & dispatch, const Value & value, size_t byte_count) {
    dispatch.bindings.push_back({ value.storage_root, value.storage_offset, byte_count });
}

static void bind_binary_buffers(Dispatch & dispatch,
                                const Value & lhs,
                                size_t        lhs_byte_count,
                                const Value & rhs,
                                size_t        rhs_byte_count,
                                const Value & output) {
    bind_binary_source_buffer(dispatch, lhs, lhs_byte_count);
    bind_binary_source_buffer(dispatch, rhs, rhs_byte_count);
    dispatch.bindings.push_back({ output.id, 0, output.byte_count });
}

static void add_binary_strided_parameters(Dispatch & dispatch,
                                          const Value & lhs,
                                          const Value & rhs,
                                          const Value & output,
                                          size_t        lhs_byte_count,
                                          size_t        rhs_byte_count) {
    dispatch.kernel.integer_parameters.emplace("element_count", output.element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne0", std::to_string(output.ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne1", std::to_string(output.ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne2", std::to_string(output.ne[2]));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride1",
                                               std::to_string(lhs.nb[1] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride2",
                                               std::to_string(lhs.nb[2] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride3",
                                               std::to_string(lhs.nb[3] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride1",
                                               std::to_string(rhs.nb[1] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride2",
                                               std::to_string(rhs.nb[2] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride3",
                                               std::to_string(rhs.nb[3] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_span",
                                               std::to_string(lhs_byte_count / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_span",
                                               std::to_string(rhs_byte_count / sizeof(float)));
}

static bool match_binary_swiglu_symmetric_i4_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_GLU || node->inputs.size() != 2 ||
        !common_is_swiglu_params(node->params)) {
        return false;
    }

    const Value * lhs    = graph_value(context.graph, node->inputs[0]);
    const Value * rhs    = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (lhs == nullptr || rhs == nullptr || output == nullptr || lhs->type != GGML_TYPE_F32 ||
        rhs->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !same_shape(*lhs, *output) ||
        !same_shape(*rhs, *output) || !packed_f32_layout(*lhs) || !packed_f32_layout(*rhs) ||
        !packed_f32_layout(*output) || output->alias_source.value >= 0 ||
        !supported_source_layout(context.graph, *lhs) || !supported_source_layout(context.graph, *rhs) ||
        !binary_noalias_storage_is_safe(*lhs, *rhs, *output) || output->ne[0] < 256 || output->ne[0] > 32768 ||
        output->ne[0] % 64 != 0 || output->element_count <= 0 || output->element_count % output->ne[0] != 0 ||
        !common_has_symmetric_i4_lowrow_consumer(context.graph, *output)) {
        return false;
    }

    const int64_t input_size  = output->ne[0];
    const int64_t token_count = output->element_count / input_size;
    if (token_count < 1 || token_count > 16) {
        return false;
    }
    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(input_size, token_count);
    if (activation_layout.total_bytes == 0) {
        return false;
    }
    const ValueId activation = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kBinarySwiGluSymmetricI4K32Kernel);
    dispatch.kernel.compile_parameters.emplace("ggml.binary_swiglu_symmetric_i4.input_size",
                                               std::to_string(input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_swiglu_symmetric_i4.token_count",
                                               std::to_string(token_count));
    bind_binary_buffers(dispatch, *lhs, lhs->byte_count, *rhs, rhs->byte_count, *output);
    dispatch.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    Status metadata_status;
    if (!match.metadata.append_alternate_value({ output->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
                                                 kCommonSymmetricI4K32ActivationAlternateName },
                                               metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back(
        { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });
    return match.status.success();
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

    size_t lhs_byte_count = 0;
    size_t rhs_byte_count = 0;
    if (output->type != GGML_TYPE_F32 || lhs->type != GGML_TYPE_F32 || rhs->type != GGML_TYPE_F32 ||
        !positive_shape(*output) || !output->contiguous || !packed_f32_layout(*output) ||
        !strided_f32_storage_span_bytes(*lhs, lhs_byte_count) ||
        !strided_f32_storage_span_bytes(*rhs, rhs_byte_count) ||
        output->alias_source.value >= 0 ||
        !binary_output_storage_is_safe(*lhs, lhs_byte_count, *rhs, rhs_byte_count, *output) ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    if (same_shape(*lhs, *output) && same_shape(*rhs, *output)) {
        dispatch.kernel = make_kernel_specialization(kBinaryF32Kernel);
        add_binary_strided_parameters(dispatch, *lhs, *rhs, *output, lhs_byte_count, rhs_byte_count);
        dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.op",
                                                   std::to_string(binary_kind_config_value(params->op)));
    } else {
        if (!lhs->contiguous || !rhs->contiguous || !packed_f32_layout(*lhs) || !packed_f32_layout(*rhs)) {
            return false;
        }
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
    bind_binary_buffers(dispatch, *lhs, lhs_byte_count, *rhs, rhs_byte_count, *output);

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
    registry.add({
        "common.binary_swiglu_symmetric_i4_k32",
        GGML_OP_GLU,
        DispatchMatchKind::Fused,
        100,
        DispatchSource::Common,
        match_binary_swiglu_symmetric_i4_dispatch,
    });
    register_binary_dispatch_for(registry, GGML_OP_ADD);
    register_binary_dispatch_for(registry, GGML_OP_SUB);
    register_binary_dispatch_for(registry, GGML_OP_MUL);
    register_binary_dispatch_for(registry, GGML_OP_DIV);
    register_binary_dispatch_for(registry, GGML_OP_GLU);
}

}  // namespace ggml::hrx
