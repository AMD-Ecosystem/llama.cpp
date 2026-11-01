#include "dispatch-mrope-text.h"

#include "dispatch-layout-utils.h"
#include "dispatch-rope-utils.h"
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

static constexpr KernelCatalogRef kMropeF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_mrope_f32");

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool supported_shape(const Value & value) {
    const bool supported_head_size  = value.ne[0] >= 4 && value.ne[0] <= 1024 && value.ne[0] % 4 == 0;
    const bool supported_head_count = value.ne[1] >= 1 && value.ne[1] <= 64;
    const bool supported_tokens     = value.ne[2] >= 1 && value.ne[2] <= 2048 && value.ne[3] == 1;
    return supported_head_size && supported_head_count && supported_tokens;
}

static bool supported_input(const Value & value, size_t & span_elements) {
    const bool supported_type = value.type == GGML_TYPE_F32 && value.nb[0] == sizeof(float);
    const bool aligned_stride = value.nb[1] % sizeof(float) == 0 && value.nb[2] % sizeof(float) == 0;
    const bool supported_layout = supported_type && supported_shape(value) && aligned_stride;
    if (!supported_layout) {
        return false;
    }
    if (!strided_f32_storage_span_elements(value, span_elements)) {
        return false;
    }
    const size_t stride1 = value.nb[1] / sizeof(float);
    const size_t stride2 = value.nb[2] / sizeof(float);
    const bool nonoverlap_heads  = stride1 >= static_cast<size_t>(value.ne[0]);
    const bool nonoverlap_tokens = stride2 >= static_cast<size_t>(value.ne[1]) * stride1;
    const bool nonoverlap_layout = nonoverlap_heads && nonoverlap_tokens;
    if (!nonoverlap_layout) {
        return false;
    }
    const size_t required = static_cast<size_t>(value.ne[2] - 1) * stride2 +
                            static_cast<size_t>(value.ne[1] - 1) * stride1 + static_cast<size_t>(value.ne[0]);
    return required <= span_elements && span_elements <= 1073741824;
}

static bool supported_output(const Value & value) {
    const bool supported_type = value.type == GGML_TYPE_F32;
    const bool packed_element = value.nb[0] == sizeof(float);
    const bool supported_layout = supported_type && supported_shape(value) && packed_element;
    if (!supported_layout) {
        return false;
    }
    const size_t stride1 = static_cast<size_t>(value.ne[0]) * sizeof(float);
    const size_t stride2 = static_cast<size_t>(value.ne[1]) * stride1;
    return value.nb[1] == stride1 && value.nb[2] == stride2 && value.nb[3] == static_cast<size_t>(value.ne[2]) * stride2;
}

static bool supported_positions(const Value & value, int64_t token_count) {
    const bool supported_type  = value.type == GGML_TYPE_I32 && value.contiguous;
    const bool supported_shape = value.ne[0] == 4 * token_count && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
    return supported_type && supported_shape;
}

static bool supported_params(const RopeParams & params, int64_t head_size) {
    const bool supported_dims = params.n_dims >= 4 && params.n_dims <= head_size && params.n_dims % 4 == 0;
    const bool supported_mode = params.mode == GGML_ROPE_TYPE_IMROPE && params.n_offs == 0;
    const bool supported_freq = std::isfinite(params.freq_base) && params.freq_base > 0.0f &&
                                std::isfinite(params.freq_scale) && params.freq_scale > 0.0f;
    const bool supported_scale = params.ext_factor == 0.0f && std::isfinite(params.attn_factor) && params.attn_factor > 0.0f;
    int64_t section_count = 0;
    for (int section : params.sections) {
        if (section < 0) {
            return false;
        }
        section_count += section;
    }
    const bool supported_sections = section_count > 0 && section_count <= head_size;
    return supported_dims && supported_mode && supported_freq && supported_scale && supported_sections;
}

static int32_t position_plane(int64_t pair, const RopeParams & params) {
    int64_t section_count = 0;
    for (int section : params.sections) {
        section_count += section;
    }
    const int64_t sector = pair % section_count;
    const int64_t axis   = sector % 3;
    const bool height_pair = axis == 1 && sector < 3 * params.sections[1];
    const bool width_pair  = axis == 2 && sector < 3 * params.sections[2];
    const bool time_pair   = axis == 0 && sector < 3 * params.sections[0];
    if (height_pair) {
        return 1;
    }
    if (width_pair) {
        return 2;
    }
    if (time_pair) {
        return 0;
    }
    return 3;
}

static ValueId add_constant(const DispatchMatchContext & context,
                            DispatchMatch &              match,
                            const char *                 name,
                            const std::vector<uint8_t> & data) {
    const ValueId id(context.next_plan_value.value + static_cast<int32_t>(match.transients.size()) +
                     static_cast<int32_t>(match.completion_counter_requests.size()));
    match.transients.push_back({ id, name, data.size(), 256 });
    match.constant_initializations.push_back({ id, name, 0, data });
    return id;
}

static bool match_mrope_text(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool rope_op = node->op == GGML_OP_ROPE;
    const bool supported_arity = node->inputs.size() >= 2 && node->inputs.size() <= 3;
    const bool supported_root = rope_op && supported_arity;
    if (!supported_root) {
        return false;
    }
    const RopeParams * params = op_params_as<RopeParams>(node->params);
    const Value * input = context.graph.values().find(node->inputs[0]);
    const Value * positions = context.graph.values().find(node->inputs[1]);
    const Value * output = context.graph.values().find(node->output);
    size_t input_span = 0;
    const bool complete_graph = params != nullptr && input != nullptr && positions != nullptr && output != nullptr;
    if (!complete_graph) {
        return false;
    }
    const bool supported_input_layout  = supported_input(*input, input_span);
    const bool supported_output_layout = supported_output(*output);
    const bool matching_shape          = same_shape(*input, *output);
    const bool valid_positions         = supported_positions(*positions, input->ne[2]);
    const bool valid_params            = supported_params(*params, input->ne[0]);
    const bool supported_graph = supported_input_layout && supported_output_layout && matching_shape && valid_positions && valid_params;
    if (!supported_graph) {
        return false;
    }
    const bool input_aliases_output = context.graph.values().same_storage(input->id, output->id);
    const bool positions_alias_output = context.graph.values().same_storage(positions->id, output->id);
    const bool output_aliases_inputs = input_aliases_output || positions_alias_output;
    if (output_aliases_inputs) {
        return false;
    }

    const Value * factors = nullptr;
    if (node->inputs.size() == 3) {
        factors = context.graph.values().find(node->inputs[2]);
        if (factors == nullptr) {
            return false;
        }
        const bool factors_type = factors->type == GGML_TYPE_F32 && factors->contiguous;
        const bool factors_shape = factors->ne[0] == params->n_dims / 2 && factors->ne[1] == 1 &&
                                   factors->ne[2] == 1 && factors->ne[3] == 1;
        const bool supported_factors = factors_type && factors_shape;
        if (!supported_factors) {
            return false;
        }
        if (context.graph.values().same_storage(factors->id, output->id)) {
            return false;
        }
    }

    RopeFrequencyTable frequency;
    if (!build_rope_frequency_table(*params, params->n_dims, frequency)) {
        return false;
    }
    std::vector<uint8_t> selectors(static_cast<size_t>(params->n_dims / 2) * sizeof(int32_t));
    std::vector<uint8_t> unit_factors;
    for (int64_t pair = 0; pair < params->n_dims / 2; ++pair) {
        const int32_t plane = position_plane(pair, *params);
        std::memcpy(selectors.data() + static_cast<size_t>(pair) * sizeof(plane), &plane, sizeof(plane));
    }
    if (factors == nullptr) {
        unit_factors.resize(frequency.data.size());
        const float one = 1.0f;
        for (int64_t pair = 0; pair < params->n_dims / 2; ++pair) {
            std::memcpy(unit_factors.data() + static_cast<size_t>(pair) * sizeof(one), &one, sizeof(one));
        }
    }

    const ValueId theta_id = add_constant(context, match, "common.mrope_text.theta", frequency.data);
    const ValueId selector_id = add_constant(context, match, "common.mrope_text.selectors", selectors);
    const ValueId factors_id = factors == nullptr ? add_constant(context, match, "common.mrope_text.factors", unit_factors) : factors->id;
    const size_t factors_bytes = factors == nullptr ? unit_factors.size() : factors->byte_count;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMropeF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("token_count", input->ne[2]);
    dispatch.kernel.integer_parameters.emplace("input_span", input_span);
    dispatch.kernel.compile_parameters.emplace("ggml.mrope_f32.head_size", std::to_string(input->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.mrope_f32.n_dims", std::to_string(params->n_dims));
    dispatch.kernel.compile_parameters.emplace("ggml.mrope_f32.head_count", std::to_string(input->ne[1]));
    int64_t token_capacity = 1;
    while (token_capacity < input->ne[2]) {
        token_capacity *= 2;
    }
    dispatch.kernel.compile_parameters.emplace("ggml.mrope_f32.token_capacity", std::to_string(token_capacity));
    dispatch.kernel.compile_parameters.emplace("ggml.mrope_f32.input_stride1", std::to_string(input->nb[1] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.mrope_f32.input_stride2", std::to_string(input->nb[2] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.mrope_f32.mscale", rope_mscale_config_value(frequency.mscale));
    dispatch.bindings.push_back({ positions->id, 0, positions->byte_count });
    dispatch.bindings.push_back({ input->storage_root, input->storage_offset, input_span * sizeof(float) });
    dispatch.bindings.push_back({ theta_id, 0, frequency.data.size() });
    dispatch.bindings.push_back({ factors_id, 0, factors_bytes });
    dispatch.bindings.push_back({ selector_id, 0, selectors.size() });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_mrope_text_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mrope_text.f32",
        GGML_OP_ROPE,
        DispatchMatchKind::SingleOp,
        110,
        DispatchSource::Common,
        match_mrope_text,
    });
}

}  // namespace ggml::hrx
