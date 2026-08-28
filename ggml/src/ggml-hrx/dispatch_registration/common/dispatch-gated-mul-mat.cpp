#include "dispatch-gated-mul-mat.h"

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
static constexpr KernelCatalogRef kMulMatSwiGLUF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_swiglu_f32_f32_wmma");

struct MulMatSwiGLUMatch {
    const Value *            input       = nullptr;
    const Value *            gate_weight = nullptr;
    const Value *            up_weight   = nullptr;
    const Value *            gate_output = nullptr;
    const Value *            up_output   = nullptr;
    const Value *            output      = nullptr;
    const GraphNode *        gate_node   = nullptr;
    const GraphNode *        up_node     = nullptr;
    const GraphNode *        glu_node    = nullptr;
    CommonMulMatWeightFormat gate_format = CommonMulMatWeightFormat::Q4K;
    CommonMulMatWeightFormat up_format   = CommonMulMatWeightFormat::Q4K;
    int64_t                  input_size  = 0;
    int64_t                  output_size = 0;
    int64_t                  token_count = 0;

    bool matched() const {
        return input != nullptr && gate_weight != nullptr && up_weight != nullptr && gate_output != nullptr &&
               up_output != nullptr && output != nullptr && gate_node != nullptr && up_node != nullptr &&
               glu_node != nullptr && token_count > 1;
    }
};

static MulMatSwiGLUMatch match_mul_mat_swiglu(const DispatchMatchContext & context) {
    MulMatSwiGLUMatch       match;
    const CommonMulMatMatch root =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, false);
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const std::vector<const GraphNode *> & root_consumers = context.graph.index().consumers(context.root_node->output);
    if (root_consumers.size() != 1 || root_consumers.front() == nullptr || root_consumers.front()->op != GGML_OP_GLU) {
        return {};
    }

    const GraphNode * glu_node = root_consumers.front();
    if (glu_node->inputs.size() != 2 || !common_is_swiglu_params(glu_node->params)) {
        return {};
    }

    size_t glu_index = 0;
    if (!context.graph.index().node_index(glu_node, glu_index) || glu_index >= context.covered_nodes.size() ||
        context.covered_nodes[glu_index]) {
        return {};
    }

    const bool root_is_gate = glu_node->inputs[0] == context.root_node->output;
    const bool root_is_up   = glu_node->inputs[1] == context.root_node->output;
    if (!root_is_gate && !root_is_up) {
        return {};
    }

    const ValueId     peer_output_id = root_is_gate ? glu_node->inputs[1] : glu_node->inputs[0];
    const GraphNode * peer_node      = context.graph.index().producer(peer_output_id);
    if (peer_node == nullptr || peer_node == context.root_node || peer_node->op != GGML_OP_MUL_MAT) {
        return {};
    }

    size_t peer_index = 0;
    if (!context.graph.index().node_index(peer_node, peer_index) || peer_index >= context.covered_nodes.size() ||
        context.covered_nodes[peer_index]) {
        return {};
    }

    const std::vector<const GraphNode *> & peer_consumers = context.graph.index().consumers(peer_output_id);
    if (peer_consumers.size() != 1 || peer_consumers.front() != glu_node) {
        return {};
    }

    const CommonMulMatMatch peer =
        common_match_mul_mat_any_format(context.graph, peer_node, kMulMatF32F32WmmaKernel, false);
    if (!peer.matched() || peer.input->id != root.input->id || peer.input_size != root.input_size ||
        peer.output_size != root.output_size || peer.token_count != root.token_count ||
        !common_same_shape(*root.output, *peer.output)) {
        return {};
    }

    const Value * output = common_graph_value(context.graph, glu_node->output);
    if (output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !common_same_shape(*output, *root.output)) {
        return {};
    }

    match.input       = root.input;
    match.gate_weight = root_is_gate ? root.weight : peer.weight;
    match.up_weight   = root_is_gate ? peer.weight : root.weight;
    match.gate_output = root_is_gate ? root.output : peer.output;
    match.up_output   = root_is_gate ? peer.output : root.output;
    match.output      = output;
    match.gate_node   = root_is_gate ? context.root_node : peer_node;
    match.up_node     = root_is_gate ? peer_node : context.root_node;
    match.glu_node    = glu_node;
    match.gate_format = root_is_gate ? root.weight_format : peer.weight_format;
    match.up_format   = root_is_gate ? peer.weight_format : root.weight_format;
    match.input_size  = root.input_size;
    match.output_size = root.output_size;
    match.token_count = root.token_count;
    return match;
}

static bool match_mul_mat_swiglu_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const MulMatSwiGLUMatch match = match_mul_mat_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatSwiGLUF32F32WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_swiglu.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_swiglu.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_swiglu.gate_weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.gate_format)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_swiglu.up_weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.up_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.gate_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.up_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.glu_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_gated_mul_mat_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mul_mat_swiglu.f32_f32_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Common,
        match_mul_mat_swiglu_dispatch,
    });
}

}  // namespace ggml::hrx
