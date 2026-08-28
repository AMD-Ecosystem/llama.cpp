#include "dispatch-rmsnorm.h"

#include "../qwen/dispatch-llm-profiles.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kRmsNormBinaryF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_f32");
static constexpr KernelCatalogRef kRmsNormBinaryQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_q8_1_x4");
static constexpr KernelCatalogRef kRmsNormF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_f32");
static constexpr int64_t kQwenHiddenSize      = kQwen30BMoeDispatchProfile.hidden_size;
static constexpr int64_t kQwenVocabularyCount = 151936;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return is_llm_supported_query_length(kQwen30BMoeDispatchProfile, token_count);
}

static bool is_binary_op(ggml_op op) {
    return op == GGML_OP_ADD || op == GGML_OP_SUB || op == GGML_OP_MUL || op == GGML_OP_DIV;
}

static bool binary_kind_requires_order(BinaryKind kind) {
    return kind == BinaryKind::Sub || kind == BinaryKind::Div;
}

static bool is_qwen_q6k_q8_consumer(const Graph & graph, const GraphNode * consumer, const Value & input) {
    if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
        consumer->inputs[1] != input.id) {
        return false;
    }
    const Value * weight = graph_value(graph, consumer->inputs[0]);
    const Value * output = graph_value(graph, consumer->output);
    if (weight == nullptr || output == nullptr || weight->type != GGML_TYPE_Q6_K || output->type != GGML_TYPE_F32 ||
        !weight->contiguous || !output->contiguous) {
        return false;
    }
    return input.ne[0] == kQwenHiddenSize && input.ne[1] == 1 && input.ne[2] == 1 && input.ne[3] == 1 &&
           weight->ne[0] == kQwenHiddenSize && weight->ne[1] == kQwenVocabularyCount && weight->ne[2] == 1 &&
           weight->ne[3] == 1 && output->ne[0] == kQwenVocabularyCount && output->ne[1] == 1 && output->ne[2] == 1 &&
           output->ne[3] == 1;
}

static bool has_only_qwen_q6k_q8_consumers(const Graph & graph, const Value & value) {
    if (!graph.has_index()) {
        return false;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value.id);
    if (consumers.empty()) {
        return false;
    }
    for (const GraphNode * consumer : consumers) {
        if (!is_qwen_q6k_q8_consumer(graph, consumer, value)) {
            return false;
        }
    }
    return true;
}

static size_t q8_1_x4_byte_count(int64_t token_count, int64_t hidden_size) {
    if (token_count <= 0 || hidden_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static bool is_weight_shape(const Value & weight, int64_t hidden_size) {
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

struct RmsNormBinaryMatch {
    const GraphNode * rms_node          = nullptr;
    const GraphNode * binary_node       = nullptr;
    const Value *     input             = nullptr;
    const Value *     rhs               = nullptr;
    const Value *     output            = nullptr;
    size_t            rms_node_index    = 0;
    size_t            binary_node_index = 0;
    int64_t           hidden_size       = 0;
    int64_t           token_count       = 0;
    BinaryKind        op                = BinaryKind::Add;
    float             epsilon           = 0.0f;

    bool matched() const {
        return rms_node != nullptr && binary_node != nullptr && input != nullptr && rhs != nullptr && output != nullptr;
    }
};

struct RmsNormMatch {
    const GraphNode * rms_node       = nullptr;
    const Value *     input          = nullptr;
    const Value *     output         = nullptr;
    size_t            rms_node_index = 0;
    int64_t           hidden_size    = 0;
    int64_t           token_count    = 0;
    float             epsilon        = 0.0f;

    bool matched() const { return rms_node != nullptr && input != nullptr && output != nullptr; }
};

static RmsNormMatch match_rmsnorm_f32(const Graph & graph, const GraphNode * node, size_t node_index) {
    RmsNormMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1) {
        return match;
    }

    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(node->params);
    if (rms_params == nullptr || !std::isfinite(rms_params->eps) || rms_params->eps <= 0.0f) {
        return {};
    }

    const Value * input  = graph_value(graph, node->inputs[0]);
    const Value * output = graph_value(graph, node->output);
    if (input == nullptr || output == nullptr) {
        return {};
    }
    if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !input->contiguous || !output->contiguous ||
        !same_shape(*input, *output)) {
        return {};
    }

    const int64_t hidden_size = output->ne[0];
    if (!is_supported_hidden_size(hidden_size)) {
        return {};
    }
    if (hidden_size == 0 || output->element_count <= 0 || output->element_count % hidden_size != 0) {
        return {};
    }
    const int64_t token_count = output->element_count / hidden_size;
    if (!is_supported_token_count(token_count)) {
        return {};
    }

    match.rms_node       = node;
    match.input          = input;
    match.output         = output;
    match.rms_node_index = node_index;
    match.hidden_size    = hidden_size;
    match.token_count    = token_count;
    match.epsilon        = rms_params->eps;
    return match;
}

static RmsNormBinaryMatch match_rmsnorm_binary_f32(const Graph & graph, const GraphNode * node, size_t node_index) {
    RmsNormBinaryMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1 || !graph.has_index()) {
        return match;
    }

    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(node->params);
    if (rms_params == nullptr || !std::isfinite(rms_params->eps) || rms_params->eps <= 0.0f) {
        return {};
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(node->output);
    if (consumers.size() != 1) {
        return {};
    }
    const GraphNode * binary_node = consumers.front();
    size_t            binary_node_index;
    if (binary_node == nullptr || !is_binary_op(binary_node->op) || binary_node->inputs.size() != 2 ||
        !graph.index().node_index(binary_node, binary_node_index)) {
        return {};
    }
    const BinaryParams * binary_params = op_params_as<BinaryParams>(binary_node->params);
    if (binary_params == nullptr || !binary_kind_supported(binary_params->op)) {
        return {};
    }

    const bool rms_is_lhs = binary_node->inputs[0] == node->output;
    const bool rms_is_rhs = binary_node->inputs[1] == node->output;
    if (!rms_is_lhs && !rms_is_rhs) {
        return {};
    }
    if (rms_is_rhs && binary_kind_requires_order(binary_params->op)) {
        return {};
    }

    const ValueId rhs_id = rms_is_lhs ? binary_node->inputs[1] : binary_node->inputs[0];
    const Value * input  = graph_value(graph, node->inputs[0]);
    const Value * rms    = graph_value(graph, node->output);
    const Value * rhs    = graph_value(graph, rhs_id);
    const Value * output = graph_value(graph, binary_node->output);
    if (input == nullptr || rms == nullptr || rhs == nullptr || output == nullptr) {
        return {};
    }
    if (input->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || rhs->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32) {
        return {};
    }
    if (!input->contiguous || !rms->contiguous || !rhs->contiguous || !output->contiguous) {
        return {};
    }
    if (!same_shape(*input, *rms) || !same_shape(*input, *output)) {
        return {};
    }

    const int64_t hidden_size = output->ne[0];
    if (!is_supported_hidden_size(hidden_size) || !is_weight_shape(*rhs, hidden_size)) {
        return {};
    }
    if (hidden_size == 0 || output->element_count <= 0 || output->element_count % hidden_size != 0) {
        return {};
    }
    const int64_t token_count = output->element_count / hidden_size;
    if (!is_supported_token_count(token_count)) {
        return {};
    }

    match.rms_node          = node;
    match.binary_node       = binary_node;
    match.input             = input;
    match.rhs               = rhs;
    match.output            = output;
    match.rms_node_index    = node_index;
    match.binary_node_index = binary_node_index;
    match.hidden_size       = hidden_size;
    match.token_count       = token_count;
    match.op                = binary_params->op;
    match.epsilon           = rms_params->eps;
    return match;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static std::string to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

}  // namespace

static bool match_rmsnorm_binary_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormBinaryMatch rms_match =
        match_rmsnorm_binary_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.binary_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[rms_match.rms_node_index] || context.covered_nodes[rms_match.binary_node_index]) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormBinaryF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_f32.hidden_size",
                                               to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_f32.rms_epsilon",
                                               to_config_value(rms_match.epsilon));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_f32.op",
                                               std::to_string(binary_kind_config_value(rms_match.op)));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.rhs->id, 0, rms_match.rhs->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.binary_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_rmsnorm_binary_q8_1_x4_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormBinaryMatch rms_match =
        match_rmsnorm_binary_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.binary_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[rms_match.rms_node_index] || context.covered_nodes[rms_match.binary_node_index] ||
        !has_only_qwen_q6k_q8_consumers(context.graph, *rms_match.output)) {
        return false;
    }

    const size_t q8_byte_count = q8_1_x4_byte_count(rms_match.token_count, rms_match.hidden_size);
    if (q8_byte_count == 0) {
        return false;
    }

    const ValueId q8_value = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormBinaryQ8_1X4Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_q8_1_x4.hidden_size",
                                               to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_q8_1_x4.rms_epsilon",
                                               to_config_value(rms_match.epsilon));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_q8_1_x4.op",
                                               std::to_string(binary_kind_config_value(rms_match.op)));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.rhs->id, 0, rms_match.rhs->byte_count });
    dispatch.bindings.push_back({ q8_value, 0, q8_byte_count });

    Status metadata_status;
    if (!match.metadata.append_alternate_value(
            { rms_match.output->id, q8_value, GGML_TYPE_Q8_1, q8_byte_count, "common.rmsnorm_binary.q8_1_x4" },
            metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.binary_node_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back({ q8_value, "common.rmsnorm_binary.q8_1_x4", q8_byte_count, 256 });
    return match.status.success();
}

static bool match_rmsnorm_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormMatch rms_match = match_rmsnorm_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[rms_match.rms_node_index]) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.hidden_size",
                                               to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.rms_epsilon",
                                               to_config_value(rms_match.epsilon));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_rmsnorm_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.rmsnorm_binary_q8_1_x4",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        150,
        DispatchSource::Common,
        match_rmsnorm_binary_q8_1_x4_dispatch,
    });
    registry.add({
        "common.rmsnorm_binary_f32",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        100,
        DispatchSource::Common,
        match_rmsnorm_binary_f32_dispatch,
    });
    registry.add({
        "common.rmsnorm_f32",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_rmsnorm_f32_dispatch,
    });
}

}  // namespace ggml::hrx
