#include "dispatch-scale.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kScaleF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_scale_f32");

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

static bool distinct_storage(const Value & input, const Value & output) {
    return input.storage != output.storage;
}

static bool supported_source_layout(const Graph & graph, const Value & value) {
    if (value.alias_source.value < 0) {
        return true;
    }
    if (value.storage_offset != 0) {
        return false;
    }
    const GraphNode * producer = graph.index().producer(value.id);
    return producer != nullptr && (producer->op == GGML_OP_RESHAPE || producer->op == GGML_OP_CLAMP);
}

static std::string format_float_config(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

static bool match_scale_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SCALE || node->inputs.size() != 1) {
        return false;
    }

    const ScaleParams * params = op_params_as<ScaleParams>(node->params);
    if (params == nullptr) {
        return false;
    }

    const Value * output = graph_value(context.graph, node->output);
    const Value * input  = graph_value(context.graph, node->inputs[0]);
    if (output == nullptr || input == nullptr) {
        return false;
    }

    if (output->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || !same_shape(*output, *input) ||
        !positive_shape(*output) || !output->contiguous || !input->contiguous || !packed_f32_layout(*output) ||
        !packed_f32_layout(*input) || output->alias_source.value >= 0 ||
        !supported_source_layout(context.graph, *input) || !distinct_storage(*input, *output) ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kScaleF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.scale_f32.scale", format_float_config(params->scale));
    dispatch.kernel.compile_parameters.emplace("ggml.scale_f32.bias", format_float_config(params->bias));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_scale_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.scale_f32",
        GGML_OP_SCALE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_scale_f32_dispatch,
    });
}

}  // namespace ggml::hrx
