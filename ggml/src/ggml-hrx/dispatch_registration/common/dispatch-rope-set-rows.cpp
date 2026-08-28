#include "dispatch-rope-set-rows.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kRopeF32Kernel        = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_f32");
static constexpr KernelCatalogRef kSetRowsKernel        = GGML_HRX_KERNEL_REF("loom_libs", "ggml_set_rows");
static constexpr KernelCatalogRef kRopeSetRowsF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_set_rows_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_1d_shape(const Value & value, int64_t ne0) {
    return value.ne[0] == ne0 && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d_shape(const Value & value, int64_t ne0, int64_t ne1) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_rope_shape(const Value & value) {
    return value.ne[0] >= 4 && value.ne[0] <= 1024 && value.ne[0] % 4 == 0 && value.ne[1] >= 1 && value.ne[1] <= 64 &&
           value.ne[2] >= 1 && value.ne[2] <= 2048 && value.ne[3] == 1;
}

static bool is_supported_rope_params(const RopeParams & params, int64_t head_size) {
    const bool supported_mode = params.mode == GGML_ROPE_TYPE_NORMAL || params.mode == GGML_ROPE_TYPE_NEOX;
    return params.n_dims == head_size && supported_mode && std::isfinite(params.freq_base) && params.freq_base > 0.0f &&
           std::isfinite(params.freq_scale) && params.freq_scale > 0.0f && params.ext_factor == 0.0f &&
           params.attn_factor == 1.0f;
}

static bool build_rope_theta_table(const GraphNode & rope, int64_t head_size, std::vector<uint8_t> & data) {
    const RopeParams * params = op_params_as<RopeParams>(rope.params);
    if (params == nullptr || !is_supported_rope_params(*params, head_size)) {
        return false;
    }

    data.resize(static_cast<size_t>(head_size / 2) * sizeof(float));
    const float theta_scale = std::pow(params->freq_base, -2.0f / static_cast<float>(head_size));
    float       theta       = params->freq_scale;
    for (int64_t i = 0; i < head_size / 2; ++i) {
        std::memcpy(data.data() + static_cast<size_t>(i) * sizeof(float), &theta, sizeof(theta));
        theta *= theta_scale;
    }
    return true;
}

static void build_unit_frequency_factors(int64_t head_size, std::vector<uint8_t> & data) {
    data.resize(static_cast<size_t>(head_size / 2) * sizeof(float));
    const float one = 1.0f;
    for (int64_t i = 0; i < head_size / 2; ++i) {
        std::memcpy(data.data() + static_cast<size_t>(i) * sizeof(float), &one, sizeof(one));
    }
}

static bool format_value(ggml_type type, int64_t & value) {
    switch (type) {
        case GGML_TYPE_F16:
            value = 16;
            return true;
        case GGML_TYPE_F32:
            value = 32;
            return true;
        default:
            return false;
    }
}

static bool supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool supported_cache_row_count(int64_t row_count) {
    return row_count >= 1 && row_count <= 1048576;
}

static bool supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 4 && hidden_size <= 32768 && hidden_size % 4 == 0;
}

static ValueId next_match_transient_value(const DispatchMatchContext & context, const DispatchMatch & dispatch_match) {
    return ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) +
                   static_cast<int32_t>(dispatch_match.completion_counter_requests.size()));
}

struct RopeMatch {
    const GraphNode *    node               = nullptr;
    const Value *        input              = nullptr;
    const Value *        positions          = nullptr;
    const Value *        freq_factors       = nullptr;
    const Value *        output             = nullptr;
    int64_t              token_count        = 0;
    int64_t              head_count         = 0;
    int64_t              head_size          = 0;
    int64_t              mode               = 0;
    size_t               theta_bytes        = 0;
    size_t               freq_factors_bytes = 0;
    std::vector<uint8_t> theta_data;
    std::vector<uint8_t> freq_factors_data;

    bool matched() const { return node != nullptr && input != nullptr && positions != nullptr && output != nullptr; }
};

struct SetRowsMatch {
    const GraphNode * node            = nullptr;
    const Value *     rows            = nullptr;
    const Value *     indices         = nullptr;
    const Value *     cache           = nullptr;
    const Value *     output          = nullptr;
    int64_t           row_format      = 0;
    int64_t           output_format   = 0;
    int64_t           token_count     = 0;
    int64_t           cache_row_count = 0;
    int64_t           hidden_size     = 0;

    bool matched() const { return node != nullptr && rows != nullptr && indices != nullptr && output != nullptr; }
};

struct RopeSetRowsMatch {
    RopeMatch         rope;
    SetRowsMatch      set_rows;
    const GraphNode * layout       = nullptr;
    const Value *     cache_rows   = nullptr;
    size_t            set_rows_idx = 0;
    size_t            layout_idx   = 0;

    bool matched() const { return rope.matched() && set_rows.matched() && cache_rows != nullptr; }
};

static RopeMatch match_rope_f32(const Graph & graph, const GraphNode * node) {
    RopeMatch match;
    if (node == nullptr || node->op != GGML_OP_ROPE || node->inputs.size() < 2 || node->inputs.size() > 3) {
        return match;
    }

    const Value * input     = graph_value(graph, node->inputs[0]);
    const Value * positions = graph_value(graph, node->inputs[1]);
    const Value * output    = graph_value(graph, node->output);
    if (input == nullptr || positions == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || positions->type != GGML_TYPE_I32 || !input->contiguous ||
        !output->contiguous || !positions->contiguous || !is_rope_shape(*input) || !same_shape(*input, *output)) {
        return {};
    }

    const int64_t      head_size   = input->ne[0];
    const int64_t      head_count  = input->ne[1];
    const int64_t      token_count = input->ne[2];
    const RopeParams * params      = op_params_as<RopeParams>(node->params);
    if (params == nullptr || !is_supported_rope_params(*params, head_size) || !is_1d_shape(*positions, token_count)) {
        return {};
    }

    std::vector<uint8_t> theta_data;
    if (!build_rope_theta_table(*node, head_size, theta_data)) {
        return {};
    }

    const Value *        freq_factors       = nullptr;
    size_t               freq_factors_bytes = 0;
    std::vector<uint8_t> freq_factors_data;
    if (node->inputs.size() == 3) {
        freq_factors = graph_value(graph, node->inputs[2]);
        if (freq_factors == nullptr || freq_factors->type != GGML_TYPE_F32 || !freq_factors->contiguous ||
            !is_1d_shape(*freq_factors, head_size / 2)) {
            return {};
        }
        freq_factors_bytes = freq_factors->byte_count;
    } else {
        build_unit_frequency_factors(head_size, freq_factors_data);
        freq_factors_bytes = freq_factors_data.size();
    }

    match.node               = node;
    match.input              = input;
    match.positions          = positions;
    match.freq_factors       = freq_factors;
    match.output             = output;
    match.token_count        = token_count;
    match.head_count         = head_count;
    match.head_size          = head_size;
    match.mode               = params->mode;
    match.theta_bytes        = theta_data.size();
    match.freq_factors_bytes = freq_factors_bytes;
    match.theta_data         = std::move(theta_data);
    match.freq_factors_data  = std::move(freq_factors_data);
    return match;
}

static SetRowsMatch match_set_rows_2d(const Graph & graph, const GraphNode * node) {
    SetRowsMatch match;
    if (node == nullptr || node->op != GGML_OP_SET_ROWS || node->inputs.size() != 3) {
        return match;
    }

    const Value * rows    = graph_value(graph, node->inputs[0]);
    const Value * indices = graph_value(graph, node->inputs[1]);
    const Value * cache   = graph_value(graph, node->inputs[2]);
    const Value * output  = graph_value(graph, node->output);
    if (rows == nullptr || indices == nullptr || cache == nullptr || output == nullptr || !rows->contiguous ||
        !indices->contiguous || !cache->contiguous || indices->type != GGML_TYPE_I64 || output->type != cache->type ||
        !same_shape(*cache, *output) || !graph.values().same_storage(cache->id, output->id)) {
        return {};
    }

    int64_t row_format    = 0;
    int64_t output_format = 0;
    if (!format_value(rows->type, row_format) || !format_value(output->type, output_format)) {
        return {};
    }
    if (rows->type == GGML_TYPE_F16 && output->type != GGML_TYPE_F16) {
        return {};
    }

    const int64_t hidden_size     = rows->ne[0];
    const int64_t token_count     = rows->ne[1];
    const int64_t cache_row_count = cache->ne[1];
    if (!is_2d_shape(*rows, hidden_size, token_count) || !is_1d_shape(*indices, token_count) ||
        !is_2d_shape(*cache, hidden_size, cache_row_count) || !supported_hidden_size(hidden_size) ||
        !supported_token_count(token_count) || !supported_cache_row_count(cache_row_count)) {
        return {};
    }

    match.node            = node;
    match.rows            = rows;
    match.indices         = indices;
    match.cache           = cache;
    match.output          = output;
    match.row_format      = row_format;
    match.output_format   = output_format;
    match.token_count     = token_count;
    match.cache_row_count = cache_row_count;
    match.hidden_size     = hidden_size;
    return match;
}

static void add_rope_compile_parameters(Dispatch & dispatch, const RopeMatch & match) {
    dispatch.kernel.compile_parameters.emplace("ggml.rope_f32.head_size", to_config_value(match.head_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rope_f32.head_count", to_config_value(match.head_count));
    dispatch.kernel.compile_parameters.emplace("ggml.rope_f32.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.rope_f32.mode", to_config_value(match.mode));
}

static void add_set_rows_compile_parameters(Dispatch & dispatch, const SetRowsMatch & match) {
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.hidden_capacity", to_config_value(match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.input_format", to_config_value(match.row_format));
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.output_format", to_config_value(match.output_format));
}

static ValueId add_constant_binding(const DispatchMatchContext & context,
                                    DispatchMatch &              dispatch_match,
                                    const char *                 name,
                                    size_t                       byte_count,
                                    const std::vector<uint8_t> & data) {
    const ValueId value = next_match_transient_value(context, dispatch_match);
    dispatch_match.transients.push_back({ value, name, byte_count, 256 });
    dispatch_match.constant_initializations.push_back({
        value,
        name,
        0,
        data,
    });
    return value;
}

static std::pair<ValueId, ValueId> add_rope_frequency_bindings(const DispatchMatchContext & context,
                                                               const RopeMatch &            match,
                                                               DispatchMatch &              dispatch_match,
                                                               const char *                 prefix) {
    const std::string theta_name = std::string(prefix) + ".theta";
    const ValueId     theta =
        add_constant_binding(context, dispatch_match, theta_name.c_str(), match.theta_bytes, match.theta_data);
    if (match.freq_factors != nullptr) {
        return { theta, match.freq_factors->id };
    }

    const std::string factors_name = std::string(prefix) + ".freq_factors";
    const ValueId     freq_factors = add_constant_binding(context, dispatch_match, factors_name.c_str(),
                                                          match.freq_factors_bytes, match.freq_factors_data);
    return { theta, freq_factors };
}

static Dispatch make_rope_dispatch(const DispatchMatchContext & context,
                                   const RopeMatch &            match,
                                   DispatchMatch &              dispatch_match) {
    const auto [theta, freq_factors] = add_rope_frequency_bindings(context, match, dispatch_match, "common.rope");
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRopeF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    add_rope_compile_parameters(dispatch, match);
    dispatch.bindings.push_back({ match.positions->id, 0, match.positions->byte_count });
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ theta, 0, match.theta_bytes });
    dispatch.bindings.push_back({ freq_factors, 0, match.freq_factors_bytes });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    return dispatch;
}

static Dispatch make_set_rows_dispatch(const SetRowsMatch & match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSetRowsKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("cache_row_count", match.cache_row_count);
    dispatch.kernel.integer_parameters.emplace("hidden_size", match.hidden_size);
    add_set_rows_compile_parameters(dispatch, match);
    dispatch.bindings.push_back({ match.rows->id, 0, match.rows->byte_count });
    dispatch.bindings.push_back({ match.indices->id, 0, match.indices->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    return dispatch;
}

static Dispatch make_rope_set_rows_dispatch(const DispatchMatchContext & context,
                                            const RopeSetRowsMatch &     match,
                                            DispatchMatch &              dispatch_match) {
    const auto [theta, freq_factors] =
        add_rope_frequency_bindings(context, match.rope, dispatch_match, "common.rope_set_rows");
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRopeSetRowsF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.rope.token_count);
    dispatch.kernel.integer_parameters.emplace("cache_row_count", match.set_rows.cache_row_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rope_set_rows_f32.head_size",
                                               to_config_value(match.rope.head_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rope_set_rows_f32.head_count",
                                               to_config_value(match.rope.head_count));
    dispatch.kernel.compile_parameters.emplace("ggml.rope_set_rows_f32.token_capacity",
                                               to_config_value(match.rope.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.rope_set_rows_f32.output_format",
                                               to_config_value(match.set_rows.output_format));
    dispatch.kernel.compile_parameters.emplace("ggml.rope_set_rows_f32.mode", to_config_value(match.rope.mode));
    dispatch.bindings.push_back({ match.rope.positions->id, 0, match.rope.positions->byte_count });
    dispatch.bindings.push_back({ match.set_rows.indices->id, 0, match.set_rows.indices->byte_count });
    dispatch.bindings.push_back({ match.rope.input->id, 0, match.rope.input->byte_count });
    dispatch.bindings.push_back({ theta, 0, match.rope.theta_bytes });
    dispatch.bindings.push_back({ freq_factors, 0, match.rope.freq_factors_bytes });
    dispatch.bindings.push_back({ match.set_rows.output->id, 0, match.set_rows.output->byte_count });
    return dispatch;
}

static bool find_rope_set_rows_consumer(const Graph & graph, const GraphNode & rope, RopeSetRowsMatch & match) {
    if (!graph.has_index() || !graph.index().has_single_consumer(rope.output)) {
        return false;
    }

    const GraphNode * consumer = graph.index().consumers(rope.output).front();
    if (consumer == nullptr) {
        return false;
    }

    if (consumer->op == GGML_OP_SET_ROWS) {
        match.set_rows_idx = 0;
        match.layout       = nullptr;
        return graph.index().node_index(consumer, match.set_rows_idx) &&
               (match.set_rows = match_set_rows_2d(graph, consumer)).matched();
    }

    if (!is_layout_alias_node(graph, *consumer) || !graph.index().has_single_consumer(consumer->output)) {
        return false;
    }

    const GraphNode * set_rows = graph.index().consumers(consumer->output).front();
    if (set_rows == nullptr || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    match.layout = consumer;
    return graph.index().node_index(consumer, match.layout_idx) &&
           graph.index().node_index(set_rows, match.set_rows_idx) &&
           (match.set_rows = match_set_rows_2d(graph, set_rows)).matched();
}

static RopeSetRowsMatch match_rope_set_rows_f32(const Graph & graph, const GraphNode * node) {
    RopeSetRowsMatch match;
    match.rope = match_rope_f32(graph, node);
    if (!match.rope.matched() || !find_rope_set_rows_consumer(graph, *node, match)) {
        return {};
    }

    match.cache_rows = match.set_rows.rows;
    if (match.set_rows.row_format != 32 || match.set_rows.hidden_size != match.rope.head_size * match.rope.head_count ||
        match.set_rows.token_count != match.rope.token_count || match.cache_rows->type != GGML_TYPE_F32) {
        return {};
    }
    return match;
}

static bool match_rope_set_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RopeSetRowsMatch match = match_rope_set_rows_f32(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        (match.layout != nullptr && !append_covered_node_index_once(context.graph, context.covered_nodes, match.layout,
                                                                    dispatch_match.covered_nodes)) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.set_rows.node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }

    dispatch_match.dispatches.push_back(make_rope_set_rows_dispatch(context, match, dispatch_match));
    return true;
}

static bool match_rope_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RopeMatch match = match_rope_f32(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(make_rope_dispatch(context, match, dispatch_match));
    return true;
}

static bool match_set_rows_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const SetRowsMatch match = match_set_rows_2d(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(make_set_rows_dispatch(match));
    return true;
}

}  // namespace

void register_rope_set_rows_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.rope_set_rows.f32",
        GGML_OP_ROPE,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Common,
        match_rope_set_rows_f32_dispatch,
    });
    registry.add({
        "common.rope.f32",
        GGML_OP_ROPE,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_rope_f32_dispatch,
    });
    registry.add({
        "common.set_rows",
        GGML_OP_SET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_set_rows_dispatch,
    });
}

}  // namespace ggml::hrx
