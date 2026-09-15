#include "dispatch-mul-mat.h"

#include "dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMulMatF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatF32F32NarrowSplitK4Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_narrow_split_k4");
static constexpr KernelCatalogRef kMulMatF32F32DecodeWave64Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_decode_wave64");
static constexpr KernelCatalogRef kMulMatAddF32F32DecodeWave64Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_add_f32_f32_decode_wave64");
static constexpr KernelCatalogRef kMulMatBiasF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_bias_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatAddF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_add_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatBiasAddF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_bias_add_f32_f32_wmma");
static constexpr KernelCatalogRef kQuantizeF32SymmetricI4K64PlaneKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_quantize_f32_symmetric_i4_k64_plane");
static constexpr KernelCatalogRef kMulMatSymmetricI4WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_wmma");
static constexpr KernelCatalogRef kQuantizeF32SymmetricI4K32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_quantize_f32_symmetric_i4_k32");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowWmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_wmma");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowSplitK2WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_wmma");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowSplitK2DirectDotKernels[] = {
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c1"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c2"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c3"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c4"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c5"),
};
static constexpr KernelCatalogRef kQuantizeF32SymmetricI8K256Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_quantize_f32_symmetric_i8_k256");
static constexpr KernelCatalogRef kMulMatQ5KSymmetricI8WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q5_k_symmetric_i8_wmma");
static constexpr KernelCatalogRef kMulMatQ5KIQ4XSQ8_1X4WmmaToken256Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256");
static constexpr KernelCatalogRef kMulMatQ4KF16WmmaPrefillWave32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q4_k_f16_wmma_prefill_wave32");
static constexpr KernelCatalogRef kMulMatQ6KPackedToken1F16WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_packed_token1_f16_wmma");
static constexpr int64_t          kMulMatQ6KPackedMaxOutputSize = 262144;
static constexpr KernelCatalogRef kMulMatQ6KI8PrepackedF16WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_i8_prepacked_f16_wmma");
static constexpr KernelCatalogRef kMulMatQ6KF32WmmaPrefillWave32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_f32_wmma_prefill_wave32");
static constexpr KernelCatalogRef kMulMatQ6KF16WmmaPrefillWave32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_f16_wmma_prefill_wave32");
static constexpr KernelCatalogRef kSelectSymmetricI4K32GroupsKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_select_symmetric_i4_k32_groups");
static constexpr KernelCatalogRef kMulMatQ6KSymmetricI2ScanToken1Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_symmetric_i2_scan_token1");
static constexpr KernelCatalogRef kTopK8F32PartitionsRegisterKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_top_k8_f32_partitions_register");
static constexpr KernelCatalogRef kTopK128F32ReduceGatherRegisterKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_top_k128_f32_reduce_gather_register");
static constexpr KernelCatalogRef kFillNegativeF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_fill_negative_f32");
static constexpr KernelCatalogRef kMulMatQ6KPackedSelectedRefineToken1Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_packed_selected_refine_token1");

struct MulMatPostOpsMatch {
    const Value *                  input             = nullptr;
    const Value *                  weight            = nullptr;
    const Value *                  projection_output = nullptr;
    const Value *                  bias              = nullptr;
    const Value *                  residual_input    = nullptr;
    const Value *                  residual_output   = nullptr;
    const GraphNode *              bias_add_node     = nullptr;
    const GraphNode *              residual_add_node = nullptr;
    const GraphNode *              layout_node       = nullptr;
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

struct DecodeMulMatAddMatch {
    CommonMulMatMatch root;
    const Value *     residual_input  = nullptr;
    const Value *     residual_output = nullptr;
    const GraphNode * add_node        = nullptr;

    bool matched() const {
        return root.matched() && residual_input != nullptr && residual_output != nullptr && add_node != nullptr;
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

static bool has_normalized_rope_consumer(const Graph & graph, const Value & value) {
    if (!graph.has_index()) {
        return false;
    }
    for (const GraphNode * layout : graph.index().consumers(value.id)) {
        if (layout == nullptr || (layout->op != GGML_OP_RESHAPE && layout->op != GGML_OP_VIEW)) {
            continue;
        }
        const GraphNode * rms = common_find_single_consumer_with_op(graph, layout->output, GGML_OP_RMS_NORM);
        const GraphNode * mul =
            rms != nullptr ? common_find_single_consumer_with_op(graph, rms->output, GGML_OP_MUL) : nullptr;
        if (mul != nullptr && common_find_single_consumer_with_op(graph, mul->output, GGML_OP_ROPE) != nullptr) {
            return true;
        }
    }
    return false;
}

static bool has_reshaped_unary_mul_consumer(const Graph & graph, const Value & value) {
    const GraphNode * reshape = common_find_single_consumer_with_op(graph, value.id, GGML_OP_RESHAPE);
    const GraphNode * unary =
        reshape != nullptr ? common_find_single_consumer_with_op(graph, reshape->output, GGML_OP_UNARY) : nullptr;
    return unary != nullptr && common_find_single_consumer_with_op(graph, unary->output, GGML_OP_MUL) != nullptr;
}

static bool has_reshaped_transpose_concat_consumer(const Graph & graph, const Value & value) {
    const GraphNode * reshape = common_find_single_consumer_with_op(graph, value.id, GGML_OP_RESHAPE);
    const GraphNode * transpose =
        reshape != nullptr ? common_find_single_consumer_with_op(graph, reshape->output, GGML_OP_TRANSPOSE) : nullptr;
    return transpose != nullptr &&
           common_find_single_consumer_with_op(graph, transpose->output, GGML_OP_CONCAT) != nullptr;
}

static const Value * match_scaled_normalized_branch(const Graph & graph, ValueId branch_id, int64_t hidden_size) {
    const Value *     branch = common_graph_value(graph, branch_id);
    const GraphNode * mul    = graph.index().producer(branch_id);
    if (branch == nullptr || mul == nullptr || !common_binary_node_is_mul(*mul) || branch->type != GGML_TYPE_F32 ||
        !common_is_2d(*branch) || branch->ne[0] != hidden_size || branch->ne[1] != 1) {
        return nullptr;
    }

    const GraphNode * rms   = nullptr;
    const Value *     scale = nullptr;
    for (ValueId input_id : mul->inputs) {
        const GraphNode * producer = graph.index().producer(input_id);
        if (producer != nullptr && producer->op == GGML_OP_RMS_NORM) {
            if (rms != nullptr) {
                return nullptr;
            }
            rms = producer;
        } else {
            scale = common_graph_value(graph, input_id);
        }
    }
    if (rms == nullptr || rms->inputs.size() != 1 || scale == nullptr || scale->type != GGML_TYPE_F32 ||
        !common_is_2d(*scale) || scale->ne[0] != hidden_size || scale->ne[1] != 1) {
        return nullptr;
    }

    const Value * normalized = common_graph_value(graph, rms->output);
    const Value * source     = common_graph_value(graph, rms->inputs[0]);
    if (normalized == nullptr || source == nullptr || normalized->type != GGML_TYPE_F32 ||
        source->type != GGML_TYPE_F32 || !common_is_2d(*normalized) || !common_is_2d(*source) ||
        normalized->ne[0] != hidden_size || normalized->ne[1] != 1 || source->ne[0] != hidden_size ||
        source->ne[1] != 1) {
        return nullptr;
    }
    return source;
}

static bool matches_embedded_external_concat_projection(const Graph &     graph,
                                                        const GraphNode & projection,
                                                        int64_t           hidden_size) {
    if (projection.op != GGML_OP_MUL_MAT || projection.inputs.size() != 2) {
        return false;
    }

    const Value * weight = common_graph_value(graph, projection.inputs[0]);
    const Value * input  = common_graph_value(graph, projection.inputs[1]);
    const Value * output = common_graph_value(graph, projection.output);
    if (weight == nullptr || input == nullptr || output == nullptr || !common_is_2d(*weight) || !common_is_2d(*input) ||
        !common_is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || weight->ne[0] != 2 * hidden_size ||
        weight->ne[1] != hidden_size || input->ne[0] != 2 * hidden_size || input->ne[1] != 1 ||
        output->ne[0] != hidden_size || output->ne[1] != 1) {
        return false;
    }

    const GraphNode * concat = graph.index().producer(input->id);
    if (concat == nullptr || concat->op != GGML_OP_CONCAT || concat->inputs.size() != 2) {
        return false;
    }
    const Value * first_source  = match_scaled_normalized_branch(graph, concat->inputs[0], hidden_size);
    const Value * second_source = match_scaled_normalized_branch(graph, concat->inputs[1], hidden_size);
    if (first_source == nullptr || second_source == nullptr) {
        return false;
    }

    const GraphNode * first_producer  = graph.index().producer(first_source->id);
    const GraphNode * second_producer = graph.index().producer(second_source->id);
    const bool        first_is_embedding =
        first_producer != nullptr && first_producer->op == GGML_OP_GET_ROWS && first_producer->inputs.size() == 2;
    const bool second_is_embedding =
        second_producer != nullptr && second_producer->op == GGML_OP_GET_ROWS && second_producer->inputs.size() == 2;
    const bool first_is_external  = first_producer == nullptr;
    const bool second_is_external = second_producer == nullptr;
    return (first_is_embedding && second_is_external) || (second_is_embedding && first_is_external);
}

static bool has_embedded_external_concat_projection_ancestor(const Graph & graph,
                                                             ValueId       endpoint_input,
                                                             int64_t       hidden_size) {
    if (!graph.has_index() || endpoint_input.value < 0 || hidden_size <= 0) {
        return false;
    }

    std::vector<bool>    visited(graph.values().size(), false);
    std::vector<ValueId> pending = { endpoint_input };
    while (!pending.empty()) {
        const ValueId current = pending.back();
        pending.pop_back();
        if (current.value < 0 || static_cast<size_t>(current.value) >= visited.size() || visited[current.value]) {
            continue;
        }
        visited[current.value]     = true;
        const GraphNode * producer = graph.index().producer(current);
        if (producer == nullptr) {
            continue;
        }
        if (matches_embedded_external_concat_projection(graph, *producer, hidden_size)) {
            return true;
        }
        pending.insert(pending.end(), producer->inputs.begin(), producer->inputs.end());
    }
    return false;
}

static bool match_symmetric_i4_low_row_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    // Disabled: this symmetric matmul route causes a substantial numeric performance regression.
    (void) context;
    (void) dispatch_match;
    return false;

    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSymmetricI4LowRowWmmaKernel, false);
    if (!match.matched()) {
        match =
            common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSymmetricI4LowRowWmmaKernel, true);
    }
    if (!match.matched() || (match.weight->type != GGML_TYPE_Q5_K && match.weight->type != GGML_TYPE_IQ4_XS) ||
        match.weight->alias_source.value >= 0 || match.token_count < 1 || match.token_count > 16 ||
        match.input_size % 64 != 0 || match.output_size % 64 != 0) {
        return false;
    }
    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, match.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate != nullptr && alternate->name != kCommonSymmetricI4K32ActivationAlternateName) {
        alternate = nullptr;
    }

    const ValueId activation = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
    const int32_t next_value = context.next_plan_value.value + (alternate == nullptr ? 1 : 0);

    if (alternate == nullptr) {
        dispatch_match.transients.push_back(
            { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K32Kernel);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.input_size",
                                                   common_to_config_value(match.input_size));
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.token_count",
                                                   common_to_config_value(match.token_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));

        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.input->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
                  kCommonSymmetricI4K32ActivationAlternateName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
    }

    const size_t materialized_weight_bytes =
        common_symmetric_i4_shared4_weight_byte_count(match.input_size, match.output_size);
    const bool split_k = match.input_size % 128 == 0 && materialized_weight_bytes >= size_t{ 16 } * 1024 * 1024 &&
                         match.output_size <= 2 * match.input_size;

    const ValueId  partial             = split_k ? ValueId(next_value) : ValueId{};
    const ValueId  completion_counters = split_k ? ValueId(next_value + 1) : ValueId{};
    const uint32_t completion_counter_count =
        static_cast<uint32_t>(common_ceil_div(match.output_size, 16) * common_ceil_div(match.token_count, 16));
    if (split_k) {
        dispatch_match.transients.push_back(
            { partial, "common.mul_mat.symmetric_i4_lowrow.split_k_partial", match.output->byte_count, 256 });
        dispatch_match.completion_counter_requests.push_back({
            completion_counters,
            "common.mul_mat.symmetric_i4_lowrow.split_k_completion_counters",
            completion_counter_count,
        });
    }

    const bool use_split_direct_dot = split_k && match.token_count <= 5 && match.input_size % 256 == 0;

    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(
        use_split_direct_dot ? kMulMatSymmetricI4LowRowSplitK2DirectDotKernels[match.token_count - 1] :
        split_k              ? kMulMatSymmetricI4LowRowSplitK2WmmaKernel :
                               kMulMatSymmetricI4LowRowWmmaKernel);
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.input_size",
                                                  common_to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.output_size",
                                                  common_to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.token_count",
                                                  common_to_config_value(match.token_count));
    contraction.kernel.compile_parameters.emplace(
        "ggml.mul_mat.symmetric_i4.lowrow.row_group_size",
        common_to_config_value(
            static_cast<int64_t>(common_symmetric_shared4_row_group_size(match.input_size, match.output_size, 4))));
    contraction.bindings.push_back(
        match.weight->type == GGML_TYPE_Q5_K && match.token_count == 1 ?
            common_symmetric_i4_shared4_multistart_weight_binding(*match.weight, match.input_size, match.output_size) :
            common_symmetric_i4_shared4_weight_binding(*match.weight, match.input_size, match.output_size));
    contraction.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    contraction.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    contraction.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    contraction.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
    if (split_k) {
        contraction.bindings.push_back({ partial, 0, match.output->byte_count });
        contraction.bindings.push_back({ completion_counters, 0, completion_counter_count * sizeof(int32_t) });
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(contraction));
    return dispatch_match.status.success();
}

static bool match_q6_k_token1_shortlist_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    constexpr int64_t selected_group_count  = 96;
    constexpr int64_t candidate_count       = 128;
    constexpr int64_t refine_tile_size      = 64;
    constexpr size_t  partition_entry_count = 1024;

    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ6KSymmetricI2ScanToken1Kernel, true);
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q6_K ||
        match.token_count != 1 || match.weight->alias_source.value >= 0 || match.input_size % 256 != 0 ||
        match.input_size > 8192 || match.output_size < 65536 || match.output_size % 64 != 0 ||
        match.output_size < 16 * match.input_size || !context.graph.index().consumers(match.output->id).empty() ||
        !has_embedded_external_concat_projection_ancestor(context.graph, match.input->id, match.input_size)) {
        return false;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, match.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate != nullptr && alternate->name != kCommonSymmetricI4K32ActivationAlternateName) {
        alternate = nullptr;
    }

    const ValueId activation     = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
    const int32_t transient_base = context.next_plan_value.value + (alternate == nullptr ? 1 : 0);
    const ValueId selected_groups(transient_base);
    const ValueId partial_values(transient_base + 1);
    const ValueId partial_ids(transient_base + 2);
    const ValueId candidates(transient_base + 3);
    const ValueId candidate_values(transient_base + 4);

    if (alternate == nullptr) {
        dispatch_match.transients.push_back(
            { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K32Kernel);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.input_size",
                                                   common_to_config_value(match.input_size));
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.token_count",
                                                   common_to_config_value(match.token_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));

        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.input->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
                  kCommonSymmetricI4K32ActivationAlternateName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
    }

    dispatch_match.transients.push_back({ selected_groups, "common.mul_mat.q6_k_shortlist.selected_groups",
                                          static_cast<size_t>(selected_group_count) * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back(
        { partial_values, "common.mul_mat.q6_k_shortlist.partial_values", partition_entry_count * sizeof(float), 256 });
    dispatch_match.transients.push_back(
        { partial_ids, "common.mul_mat.q6_k_shortlist.partial_ids", partition_entry_count * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back(
        { candidates, "common.mul_mat.q6_k_shortlist.candidates", candidate_count * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back(
        { candidate_values, "common.mul_mat.q6_k_shortlist.candidate_values", candidate_count * sizeof(float), 256 });

    Dispatch select_groups;
    select_groups.kernel = make_kernel_specialization(kSelectSymmetricI4K32GroupsKernel);
    select_groups.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.input_size",
                                                    common_to_config_value(match.input_size));
    select_groups.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.selected_group_count",
                                                    common_to_config_value(selected_group_count));
    select_groups.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    select_groups.bindings.push_back(
        { selected_groups, 0, static_cast<size_t>(selected_group_count) * sizeof(int32_t) });
    dispatch_match.dispatches.push_back(std::move(select_groups));

    const size_t symmetric_i2_bytes =
        common_symmetric_shared4_weight_byte_count(match.input_size, match.output_size, 2);
    if (symmetric_i2_bytes > std::numeric_limits<size_t>::max() - match.weight->byte_count) {
        return false;
    }
    const size_t materialized_weight_bytes = symmetric_i2_bytes + match.weight->byte_count;
    Dispatch     scan;
    scan.kernel = make_kernel_specialization(kMulMatQ6KSymmetricI2ScanToken1Kernel);
    scan.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.input_size",
                                           common_to_config_value(match.input_size));
    scan.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.output_size",
                                           common_to_config_value(match.output_size));
    scan.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.selected_group_count",
                                           common_to_config_value(selected_group_count));
    scan.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes,
                              kQ6KSymmetricI2PackedK256Row64ScaleRowLayout, match.weight->type, match.input_size,
                              match.output_size, match.weight->byte_count });
    scan.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    scan.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    scan.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    scan.bindings.push_back({ selected_groups, 0, static_cast<size_t>(selected_group_count) * sizeof(int32_t) });
    dispatch_match.dispatches.push_back(std::move(scan));

    Dispatch partition_top_k;
    partition_top_k.kernel = make_kernel_specialization(kTopK8F32PartitionsRegisterKernel);
    partition_top_k.kernel.integer_parameters.emplace("element_count", match.output_size);
    partition_top_k.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    partition_top_k.bindings.push_back({ partial_values, 0, partition_entry_count * sizeof(float) });
    partition_top_k.bindings.push_back({ partial_ids, 0, partition_entry_count * sizeof(int32_t) });
    dispatch_match.dispatches.push_back(std::move(partition_top_k));

    Dispatch reduce_top_k;
    reduce_top_k.kernel = make_kernel_specialization(kTopK128F32ReduceGatherRegisterKernel);
    reduce_top_k.kernel.integer_parameters.emplace("element_count", match.output_size);
    reduce_top_k.bindings.push_back({ partial_values, 0, partition_entry_count * sizeof(float) });
    reduce_top_k.bindings.push_back({ partial_ids, 0, partition_entry_count * sizeof(int32_t) });
    reduce_top_k.bindings.push_back({ candidates, 0, candidate_count * sizeof(int32_t) });
    reduce_top_k.bindings.push_back({ candidate_values, 0, candidate_count * sizeof(float) });
    dispatch_match.dispatches.push_back(std::move(reduce_top_k));

    Dispatch fill;
    fill.kernel = make_kernel_specialization(kFillNegativeF32Kernel);
    fill.kernel.integer_parameters.emplace("element_count", match.output_size);
    fill.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch_match.dispatches.push_back(std::move(fill));

    for (int64_t candidate_offset = 0; candidate_offset < candidate_count; candidate_offset += refine_tile_size) {
        Dispatch refine;
        refine.kernel = make_kernel_specialization(kMulMatQ6KPackedSelectedRefineToken1Kernel);
        refine.kernel.integer_parameters.emplace("token_count", match.token_count);
        refine.kernel.integer_parameters.emplace("candidate_count", refine_tile_size);
        refine.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.input_size",
                                                 common_to_config_value(match.input_size));
        refine.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_size",
                                                 common_to_config_value(match.output_size));
        refine.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_accumulation", "0");
        refine.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.weight_offset",
                                                 common_to_config_value(static_cast<int64_t>(symmetric_i2_bytes)));
        refine.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        refine.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes,
                                    kQ6KSymmetricI2PackedK256Row64ScaleRowLayout, match.weight->type, match.input_size,
                                    match.output_size, match.weight->byte_count });
        refine.bindings.push_back({ candidates, static_cast<size_t>(candidate_offset) * sizeof(int32_t),
                                    static_cast<size_t>(refine_tile_size) * sizeof(int32_t) });
        refine.bindings.push_back({ candidate_values, static_cast<size_t>(candidate_offset) * sizeof(float),
                                    static_cast<size_t>(refine_tile_size) * sizeof(float) });
        refine.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch_match.dispatches.push_back(std::move(refine));
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    return dispatch_match.status.success();
}

static bool build_q6_k_aligned_skinny_dispatch(const CommonMulMatMatch & match,
                                               size_t                    root_index,
                                               KernelCatalogRef          kernel,
                                               DispatchMatch &           dispatch_match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_accumulation", "0");
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, kQ6KPackedK256Row64ScaleRowLayout,
                                  match.weight->type, match.input_size, match.output_size, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_q6_k_aligned_skinny_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ6KPackedToken1F16WmmaKernel, false);
    if (!match.matched()) {
        match = common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ6KPackedToken1F16WmmaKernel,
                                                true);
    }
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q6_K ||
        match.weight->alias_source.value >= 0 || match.token_count > 16 || match.input_size % 256 != 0 ||
        match.output_size % 64 != 0 || match.output_size > kMulMatQ6KPackedMaxOutputSize ||
        (match.token_count > 1 && match.output_size < 8 * match.input_size) ||
        !context.graph.index().consumers(match.output->id).empty()) {
        return false;
    }

    return build_q6_k_aligned_skinny_dispatch(match, context.root_index, kMulMatQ6KPackedToken1F16WmmaKernel,
                                              dispatch_match);
}

static bool match_q6_k_i8_prepacked_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ6KI8PrepackedF16WmmaKernel, false);
    if (!match.matched() || match.weight->type != GGML_TYPE_Q6_K || match.weight->alias_source.value >= 0 ||
        match.token_count <= 5 || match.token_count > 16 || match.input_size % 256 != 0 || match.output_size % 64 != 0 ||
        (match.output_size > match.input_size / 2 && match.token_count > 5)) {
        return false;
    }

    const size_t block_count               = static_cast<size_t>(match.input_size / ggml_blck_size(GGML_TYPE_Q6_K));
    size_t       materialized_weight_bytes = static_cast<size_t>(match.output_size) * block_count;
    if (materialized_weight_bytes > std::numeric_limits<size_t>::max() / size_t{ 274 }) {
        return false;
    }
    materialized_weight_bytes *= size_t{ 274 };

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatQ6KI8PrepackedF16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_accumulation", "0");
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kQ6KI8K32Row64Layout,
                                  match.weight->type, match.input_size, match.output_size, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool build_q6_k_prefill_wave32_dispatch(const CommonMulMatMatch & match,
                                               ValueId                   input,
                                               size_t                    input_bytes,
                                               KernelCatalogRef          kernel,
                                               size_t                    root_index,
                                               DispatchMatch &           dispatch_match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.bindings.push_back({ input, 0, input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_q6_k_prefill_wave32_dispatch(const DispatchMatchContext & context,
                                               DispatchMatch &              dispatch_match) {
    const CommonMulMatMatch match = common_match_mul_mat_any_format(
        context.graph, context.root_node, kMulMatQ6KF32WmmaPrefillWave32Kernel, false);
    if (!match.matched() || match.weight->type != GGML_TYPE_Q6_K || match.weight->alias_source.value >= 0 ||
        match.token_count < 128 || match.token_count > 2048 || match.token_count % 128 != 0 ||
        match.input_size % 256 != 0 || match.output_size % 64 != 0) {
        return false;
    }

    const bool packed_input = common_mul_mat_uses_k16_major_f16(
        match.weight_format, match.input_size, match.output_size, match.token_count);
    DispatchBinding activation;
    const bool prepared = packed_input ?
        common_prepare_k16_major_f16_input(context, *match.input, match.input_size, match.token_count,
                                           dispatch_match, activation) :
        common_prepare_f16_input(context, *match.input, match.input_size, match.token_count,
                                  dispatch_match, activation);
    if (!prepared) {
        return false;
    }
    return build_q6_k_prefill_wave32_dispatch(match, activation.value, activation.length,
                                              kMulMatQ6KF16WmmaPrefillWave32Kernel, context.root_index,
                                              dispatch_match);
}

static bool match_symmetric_i4_prefill_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    // Disabled: this symmetric matmul route causes a substantial numeric performance regression.
    (void) context;
    (void) dispatch_match;
    return false;

    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSymmetricI4WmmaKernel, false);
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q5_K ||
        match.weight->alias_source.value >= 0 || match.token_count < 128 || match.token_count % 128 != 0 ||
        match.input_size % 64 != 0 || match.output_size < match.input_size || match.output_size % 128 != 0) {
        return false;
    }

    const GraphNode * input_producer = context.graph.index().producer(match.input->id);
    if (input_producer != nullptr && input_producer->op == GGML_OP_GLU) {
        return false;
    }
    if (!has_normalized_rope_consumer(context.graph, *match.output) &&
        !has_reshaped_unary_mul_consumer(context.graph, *match.output) &&
        !has_reshaped_transpose_concat_consumer(context.graph, *match.output)) {
        return false;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const ValueId activation = context.next_plan_value;
    dispatch_match.transients.push_back(
        { activation, "common.mul_mat.symmetric_i4_k64.activation", activation_layout.total_bytes, 256 });

    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K64PlaneKernel);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k64.input_size",
                                               common_to_config_value(match.input_size));
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k64.token_count",
                                               common_to_config_value(match.token_count));
    quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    const size_t materialized_weight_bytes =
        static_cast<size_t>(match.output_size) * static_cast<size_t>(match.input_size / 256) * size_t{ 144 };
    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(kMulMatSymmetricI4WmmaKernel);
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.input_size",
                                                  common_to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.output_size",
                                                  common_to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.token_count",
                                                  common_to_config_value(match.token_count));
    contraction.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kSymmetricI4K64Row64Layout,
                                     match.weight->type, match.input_size, match.output_size,
                                     match.weight->byte_count });
    contraction.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    contraction.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    contraction.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    contraction.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(quantize));
    dispatch_match.dispatches.push_back(std::move(contraction));
    return true;
}

static bool match_q5_k_symmetric_i8_prefill_dispatch(const DispatchMatchContext & context,
                                                     DispatchMatch &              dispatch_match) {
    // Disabled: this symmetric matmul route causes a substantial numeric performance regression.
    (void) context;
    (void) dispatch_match;
    return false;

    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ5KSymmetricI8WmmaKernel, false);
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q5_K ||
        match.weight->alias_source.value >= 0 || match.token_count < 256 || match.token_count % 256 != 0 ||
        match.input_size % 256 != 0 || match.output_size % 64 != 0 || match.output_size > match.input_size) {
        return false;
    }

    const GraphNode * input_producer = context.graph.index().producer(match.input->id);
    if (input_producer != nullptr && input_producer->op == GGML_OP_GLU) {
        return false;
    }

    const size_t element_count = static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size);
    const size_t metadata_bytes =
        static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size / 256) * sizeof(int32_t);
    const size_t  quantized_bytes = element_count + metadata_bytes;
    const ValueId quantized       = context.next_plan_value;
    dispatch_match.transients.push_back(
        { quantized, "common.mul_mat.symmetric_i8_k256.activation", quantized_bytes, 256 });

    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI8K256Kernel);
    quantize.kernel.integer_parameters.emplace("token_count", match.token_count);
    quantize.kernel.integer_parameters.emplace("input_size_arg", match.input_size);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i8_k256.token_count",
                                               common_to_config_value(match.token_count));
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i8_k256.input_size",
                                               common_to_config_value(match.input_size));
    quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    quantize.bindings.push_back({ quantized, 0, quantized_bytes });

    const size_t materialized_weight_bytes =
        static_cast<size_t>(match.output_size) * static_cast<size_t>(match.input_size / 256) * size_t{ 258 };
    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(kMulMatQ5KSymmetricI8WmmaKernel);
    contraction.kernel.integer_parameters.emplace("token_count", match.token_count);
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i8.input_size",
                                                  common_to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i8.output_size",
                                                  common_to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i8.token_count",
                                                  common_to_config_value(match.token_count));
    contraction.bindings.push_back({ quantized, 0, quantized_bytes });
    contraction.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kQ5KSymmetricI8K256Row64Layout,
                                     match.weight->type, match.input_size, match.output_size,
                                     match.weight->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(quantize));
    dispatch_match.dispatches.push_back(std::move(contraction));
    return true;
}


static bool match_packed_q8_1_x4_prefill_dispatch(const DispatchMatchContext & context,
                                                  DispatchMatch &              dispatch_match) {
    const CommonMulMatMatch match = common_match_mul_mat_any_format(context.graph, context.root_node,
                                                                    kMulMatQ5KIQ4XSQ8_1X4WmmaToken256Kernel, false);
    if (!match.matched() || !context.graph.has_index() ||
        (match.weight->type != GGML_TYPE_Q4_K && match.weight->type != GGML_TYPE_Q5_K &&
         match.weight->type != GGML_TYPE_IQ4_XS) ||
        match.token_count < 256 || match.token_count > 2048 || match.token_count % 256 != 0 ||
        match.input_size % 256 != 0 || match.output_size % 64 != 0) {
        return false;
    }

    for (const GraphNode * consumer : context.graph.index().consumers(match.output->id)) {
        if (match.weight->type != GGML_TYPE_Q4_K && consumer != nullptr && consumer->op == GGML_OP_GLU) {
            return false;
        }
    }

    const bool use_f16 = match.weight->type == GGML_TYPE_Q4_K && match.weight->alias_source.value < 0 &&
                         match.output_size >= match.input_size / 4;
    DispatchBinding activation;
    const bool packed_input = use_f16 && common_mul_mat_uses_k16_major_f16(
        match.weight_format, match.input_size, match.output_size, match.token_count,
        context.plan.metadata.find_generated_resource(match.input->id, GeneratedResourceRole::F16K16Major) != nullptr);
    if (use_f16) {
        const bool prepared = packed_input ?
            common_prepare_k16_major_f16_input(context, *match.input, match.input_size, match.token_count,
                                               dispatch_match, activation) :
            common_prepare_f16_input(context, *match.input, match.input_size, match.token_count,
                                      dispatch_match, activation);
        if (!prepared) {
            return false;
        }
    } else if (!common_prepare_q8_1_x4_input(context, *match.input, match.input_size, match.token_count,
                                             dispatch_match, activation, match.weight->type == GGML_TYPE_Q4_K)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(use_f16 ? kMulMatQ4KF16WmmaPrefillWave32Kernel :
                                                            kMulMatQ5KIQ4XSQ8_1X4WmmaToken256Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace(use_f16 ? "ggml.mul_mat.input_size" :
                                                         "ggml.mul_mat_q8_1_x4.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace(use_f16 ? "ggml.mul_mat.output_size" :
                                                         "ggml.mul_mat_q8_1_x4.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(use_f16 ? "ggml.workload.token_capacity" :
                                                         "ggml.mul_mat_q8_1_x4.token_capacity",
                                               common_to_config_value(match.token_count));
    const bool pack_q4 = match.weight_format == CommonMulMatWeightFormat::Q4K &&
                         match.weight->alias_source.value < 0;
    if (use_f16) {
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.f16_input_layout", packed_input ? "1" : "0");
    } else {
        dispatch.kernel.compile_parameters.emplace(
            "ggml.mul_mat_q8_1_x4.weight_format",
            common_to_config_value(common_mul_mat_format_config_value(pack_q4 ? CommonMulMatWeightFormat::Q4KRow64 :
                                                                               match.weight_format)));
    }
    dispatch.bindings.push_back(activation);
    if (pack_q4) {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, kQ4KPackedK256Row64Layout,
                                      match.weight->type, match.input_size, match.output_size,
                                      match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_q4_k_q8_1_x4_prefill_dispatch(const DispatchMatchContext & context,
                                                 DispatchMatch &              dispatch_match) {
    if (context.root_node == nullptr || context.root_node->inputs.empty()) {
        return false;
    }
    const Value * weight = common_graph_value(context.graph, context.root_node->inputs[0]);
    return weight != nullptr && weight->type == GGML_TYPE_Q4_K &&
           match_packed_q8_1_x4_prefill_dispatch(context, dispatch_match);
}

static MulMatPostOpsMatch match_mul_mat_postops(const DispatchMatchContext & context) {
    MulMatPostOpsMatch      match;
    CommonMulMatMatch root =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, false);
    if (!root.matched()) {
        root = common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, true);
    }
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const bool lowtoken_contraction = root.token_count <= 5 && root.weight->alias_source.value < 0 &&
                                      (root.weight_format == CommonMulMatWeightFormat::Q4K ||
                                       root.weight_format == CommonMulMatWeightFormat::Q6K) &&
                                      root.input_size >= 4096 && root.output_size >= 4096 &&
                                      root.output_size <= root.input_size && root.output_size % 64 == 0;
    if (root.token_count == 1 && !lowtoken_contraction) {
        return {};
    }

    const Value * current = root.output;
    if (lowtoken_contraction) {
        const GraphNode * reshape = common_find_only_consumer_with_op(context.graph, current->id, GGML_OP_RESHAPE);
        const Value * reshaped = reshape != nullptr ? common_graph_value(context.graph, reshape->output) : nullptr;
        if (reshaped != nullptr && is_layout_alias_node(context.graph, *reshape) && reshaped->contiguous &&
            same_full_value_range(*current, *reshaped) && common_same_shape(*current, *reshaped)) {
            match.layout_node = reshape;
            current = reshaped;
        }
    }
    const int add_limit = root.token_count == 1 || match.layout_node != nullptr ? 1 : 2;
    for (int add_index = 0; add_index < add_limit; ++add_index) {
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

static bool build_mul_mat_dispatch(const DispatchMatchContext & context,
                                   const CommonMulMatMatch &  match,
                                   DispatchMatch &            dispatch_match) {
    const bool split_k = match.weight_format == CommonMulMatWeightFormat::Q4K && !match.has_fused_unary &&
                         match.input_size >= 4096 && match.input_size % 1024 == 0 &&
                         match.output_size <= 64 && match.output_size % 4 == 0 &&
                         match.token_count >= 128 && match.token_count % 32 == 0 &&
                         common_ceil_div(match.token_count, 32) < 32;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(split_k ? kMulMatF32F32NarrowSplitK4Kernel : match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.input_size", common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_size", common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_unary_op",
                                               std::to_string(unary_kind_config_value(match.output_unary_op)));
    const bool pack_rows = match.token_count <= 5 && match.output_size % 64 == 0 &&
                           match.weight->alias_source.value < 0;
    const bool pack_q4 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q4K;
    const bool pack_q6 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q6K;
    const CommonMulMatWeightFormat format = pack_q4 ? CommonMulMatWeightFormat::Q4KRow64 :
                                           pack_q6 ? CommonMulMatWeightFormat::Q6KRow64 : match.weight_format;
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(format)));
    DispatchBinding activation = { match.input->id, 0, match.input->byte_count };
    if (pack_q4 || pack_q6) {
        if (!common_prepare_q8_1_x4_input(context, *match.input, match.input_size, match.token_count,
                                          dispatch_match, activation)) {
            return false;
        }
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.activation_format",
                                                    std::to_string(GGML_TYPE_Q8_1));
    }
    dispatch.bindings.push_back(activation);
    if (pack_q4 || pack_q6) {
        const char * layout = pack_q4 ? kQ4KPackedK256Row64Layout : kQ6KPackedK256Row64ScaleRowLayout;
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, layout,
                                      match.weight->type, match.input_size, match.output_size,
                                      match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    if (split_k) {
        const ValueId partial = context.next_plan_value;
        const ValueId counters(partial.value + 1);
        const size_t partial_bytes = 4 * match.output->byte_count;
        const uint32_t counter_count = static_cast<uint32_t>(match.token_count / 32);
        dispatch_match.transients.push_back(
            { partial, "common.mul_mat.narrow.split_k_partial", partial_bytes, 256 });
        dispatch_match.completion_counter_requests.push_back(
            { counters, "common.mul_mat.narrow.split_k_counters", counter_count });
        dispatch.bindings.push_back({ partial, 0, partial_bytes });
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch.bindings.push_back({ counters, 0, counter_count * sizeof(int32_t) });
    } else {
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.has_fused_unary) {
        dispatch_match.covered_nodes.push_back(match.unary_node_index);
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
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
    const bool pack_rows = match.token_count <= 5 && match.output_size % 64 == 0 &&
                           match.weight->alias_source.value < 0;
    const bool pack_q4 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q4K;
    const bool pack_q6 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q6K;
    const CommonMulMatWeightFormat format = pack_q4 ? CommonMulMatWeightFormat::Q4KRow64 :
                                           pack_q6 ? CommonMulMatWeightFormat::Q6KRow64 : match.weight_format;
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_postops.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(format)));
    DispatchBinding activation = { match.input->id, 0, match.input->byte_count };
    if (pack_q4 || pack_q6) {
        if (!common_prepare_q8_1_x4_input(context, *match.input, match.input_size, match.token_count,
                                          dispatch_match, activation)) {
            return false;
        }
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.activation_format",
                                                    std::to_string(GGML_TYPE_Q8_1));
    }
    dispatch.bindings.push_back(activation);
    if (pack_q4 || pack_q6) {
        const char * layout = pack_q4 ? kQ4KPackedK256Row64Layout : kQ6KPackedK256Row64ScaleRowLayout;
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, layout,
                                      match.weight->type, match.input_size, match.output_size,
                                      match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
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
    if (match.layout_node != nullptr &&
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.layout_node,
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

static DecodeMulMatAddMatch match_decode_mul_mat_add(const DispatchMatchContext & context) {
    DecodeMulMatAddMatch match;
    CommonMulMatMatch    root =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatAddF32F32DecodeWave64Kernel, true);
    if (!root.matched() || !context.graph.has_index() || root.token_count != 1) {
        return match;
    }

    const GraphNode * add_node = common_find_only_consumer_with_op(context.graph, root.output->id, GGML_OP_ADD);
    if (add_node == nullptr || !common_binary_node_is_add(*add_node)) {
        return match;
    }

    const bool root_is_lhs = add_node->inputs[0] == root.output->id;
    const bool root_is_rhs = add_node->inputs[1] == root.output->id;
    if (!root_is_lhs && !root_is_rhs) {
        return match;
    }

    const ValueId residual_id = root_is_lhs ? add_node->inputs[1] : add_node->inputs[0];
    const Value * residual    = common_graph_value(context.graph, residual_id);
    const Value * output      = common_graph_value(context.graph, add_node->output);
    if (residual == nullptr || output == nullptr || residual->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !residual->contiguous || !output->contiguous || !common_same_shape(*root.output, *residual) ||
        !common_same_shape(*root.output, *output)) {
        return match;
    }

    size_t add_index = 0;
    if (!context.graph.index().node_index(add_node, add_index) || add_index >= context.covered_nodes.size() ||
        context.covered_nodes[add_index]) {
        return match;
    }

    // Keep ADD -> RMS_NORM available for the stronger next-rmsnorm fusion.
    if (common_find_single_consumer_with_op(context.graph, output->id, GGML_OP_RMS_NORM) != nullptr) {
        return match;
    }

    root.output           = output;
    match.root            = root;
    match.residual_input  = residual;
    match.residual_output = output;
    match.add_node        = add_node;
    return match;
}

static bool build_decode_mul_mat_add_dispatch(const DecodeMulMatAddMatch & match,
                                              const DispatchMatchContext & context,
                                              DispatchMatch &              dispatch_match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatAddF32F32DecodeWave64Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.root.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", match.root.input_size);
    dispatch.kernel.integer_parameters.emplace("output_size", match.root.output_size);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.token_capacity",
                                               common_to_config_value(match.root.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.output_capacity",
                                               common_to_config_value(match.root.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_f32_f32_decode.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.root.weight_format)));
    dispatch.bindings.push_back({ match.root.input->id, 0, match.root.input->byte_count });
    dispatch.bindings.push_back({ match.root.weight->id, 0, match.root.weight->byte_count });
    dispatch.bindings.push_back({ match.residual_input->id, 0, match.residual_input->byte_count });
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.add_node,
                                        dispatch_match.covered_nodes)) {
        return false;
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
    return build_mul_mat_dispatch(context, match, dispatch_match);
}

static bool match_mul_mat_unary_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, false);
    if (!match.matched() || !try_match_fused_unary(context, match)) {
        return false;
    }
    return build_mul_mat_dispatch(context, match, dispatch_match);
}

static bool match_decode_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32DecodeWave64Kernel, true);
    if (!match.matched()) {
        return false;
    }
    if ((match.weight_format == CommonMulMatWeightFormat::Q4K ||
         match.weight_format == CommonMulMatWeightFormat::Q6K) && match.output_size % 64 == 0 &&
        common_is_supported_dense_output_size(match.output_size) && match.weight->alias_source.value < 0) {
        match.kernel = kMulMatF32F32WmmaKernel;
        return build_mul_mat_dispatch(context, match, dispatch_match);
    } else {
        build_decode_mul_mat_dispatch(match, dispatch_match, context.root_index);
    }
    return true;
}

static bool match_decode_mul_mat_add_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const DecodeMulMatAddMatch match = match_decode_mul_mat_add(context);
    if (!match.matched()) {
        return false;
    }
    return build_decode_mul_mat_add_dispatch(match, context, dispatch_match);
}

void register_mul_mat_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mul_mat.q6_k_token1_shortlist",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        310,
        DispatchSource::Common,
        match_q6_k_token1_shortlist_dispatch,
    });
    registry.add({
        "common.mul_mat.q6_k_prefill_wave32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        295,
        DispatchSource::Common,
        match_q6_k_prefill_wave32_dispatch,
    });
    registry.add({
        "common.mul_mat.q6_k_aligned_skinny",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_q6_k_aligned_skinny_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_iq4_xs_symmetric_i4_lowrow",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        285,
        DispatchSource::Common,
        match_symmetric_i4_low_row_dispatch,
    });
    registry.add({
        "common.mul_mat.q6_k_i8_prepacked",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        285,
        DispatchSource::Common,
        match_q6_k_i8_prepacked_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_iq4_xs_q8_1_x4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        260,
        DispatchSource::Common,
        match_packed_q8_1_x4_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat.q4_k_q8_1_x4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        305,
        DispatchSource::Common,
        match_q4_k_q8_1_x4_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_symmetric_i4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        270,
        DispatchSource::Common,
        match_symmetric_i4_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_symmetric_i8_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        280,
        DispatchSource::Common,
        match_q5_k_symmetric_i8_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat_unary.f32_f32_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        290,
        DispatchSource::Common,
        match_mul_mat_unary_dispatch,
    });
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
        "common.mul_mat_add.f32_f32_decode",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        70,
        DispatchSource::Common,
        match_decode_mul_mat_add_dispatch,
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
