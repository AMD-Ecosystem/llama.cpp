#include "dispatch-im2col.h"

#include "dispatch-layout-utils.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kIm2ColF32F16Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_im2col_f32_f16");
static constexpr KernelCatalogRef kIm2ColF32F32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_im2col_f32_f32");

static bool within_config_range(int64_t value, int64_t minimum, int64_t maximum) {
    return value >= minimum && value <= maximum;
}

static bool match_im2col_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool is_im2col = node->op == GGML_OP_IM2COL;
    const bool has_inputs = node->inputs.size() == 2;
    const bool matches_op = is_im2col && has_inputs;
    if (!matches_op) {
        return false;
    }

    const Im2ColParams * params = op_params_as<Im2ColParams>(node->params);
    const Value * kernel = context.graph.values().find(node->inputs[0]);
    const Value * source = context.graph.values().find(node->inputs[1]);
    const Value * output = context.graph.values().find(node->output);
    const bool has_parameters = params != nullptr;
    const bool has_values = kernel != nullptr && source != nullptr && output != nullptr;
    const bool has_operands = has_parameters && has_values;
    if (!has_operands) {
        return false;
    }

    const bool supported_types = source->type == GGML_TYPE_F32 &&
                                 (output->type == GGML_TYPE_F16 || output->type == GGML_TYPE_F32);
    const bool valid_dimensions =
        within_config_range(kernel->ne[0], 1, 65536) && within_config_range(kernel->ne[1], 1, 65536) &&
        within_config_range(source->ne[0], 1, 65536) && within_config_range(source->ne[1], 1, 65536) &&
        within_config_range(source->ne[2], 1, 65536) && within_config_range(source->ne[3], 1, 65536) &&
        kernel->ne[2] == source->ne[2];
    const bool valid_parameters =
        params->is_2d && within_config_range(params->stride_x, 1, 65536) &&
        within_config_range(params->stride_y, 1, 65536) && within_config_range(params->pad_x, 0, 65536) &&
        within_config_range(params->pad_y, 0, 65536) && within_config_range(params->dilation_x, 1, 65536) &&
        within_config_range(params->dilation_y, 1, 65536);
    const bool supported_input = supported_types && valid_dimensions && valid_parameters;
    if (!supported_input) {
        return false;
    }

    const int64_t output_width =
        (source->ne[0] + 2 * params->pad_x - params->dilation_x * (kernel->ne[0] - 1) - 1) / params->stride_x + 1;
    const int64_t output_height =
        (source->ne[1] + 2 * params->pad_y - params->dilation_y * (kernel->ne[1] - 1) - 1) / params->stride_y + 1;
    const int64_t patch_width = kernel->ne[0] * kernel->ne[1] * source->ne[2];
    const bool valid_output = within_config_range(output_width, 1, 65536) &&
                              within_config_range(output_height, 1, 65536) &&
                              output->ne[0] == patch_width && output->ne[1] == output_width &&
                              output->ne[2] == output_height && output->ne[3] == source->ne[3] &&
                              within_config_range(output->element_count, 1, 134217728) && output->contiguous;
    const bool supported_source_layout = source->nb[0] == sizeof(float) &&
                                         source->nb[1] == static_cast<size_t>(source->ne[0]) * sizeof(float) &&
                                         source->nb[2] >= sizeof(float) && source->nb[3] >= sizeof(float) &&
                                         source->nb[2] % sizeof(float) == 0 &&
                                         source->nb[3] % sizeof(float) == 0 &&
                                         source->nb[2] / sizeof(float) <= 1073741824 &&
                                         source->nb[3] / sizeof(float) <= 1073741824;
    const bool distinct_storage = source->storage != output->storage;
    const bool supported_layout = valid_output && supported_source_layout && distinct_storage;
    if (!supported_layout) {
        return false;
    }

    size_t source_span = 0;
    const bool valid_source_span = strided_f32_storage_span_bytes(*source, source_span);
    const bool supported_source_span = valid_source_span && source_span / sizeof(float) <= 1073741824;
    if (!supported_source_span) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(output->type == GGML_TYPE_F16 ? kIm2ColF32F16Kernel : kIm2ColF32F32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", source_span / sizeof(float));
    static constexpr const char * kPrefix = "ggml.im2col.";
    auto & compile = dispatch.kernel.compile_parameters;
    compile.emplace(std::string(kPrefix) + "kernel_width", std::to_string(kernel->ne[0]));
    compile.emplace(std::string(kPrefix) + "kernel_height", std::to_string(kernel->ne[1]));
    compile.emplace(std::string(kPrefix) + "channels", std::to_string(source->ne[2]));
    compile.emplace(std::string(kPrefix) + "output_width", std::to_string(output_width));
    compile.emplace(std::string(kPrefix) + "output_height", std::to_string(output_height));
    compile.emplace(std::string(kPrefix) + "input_width", std::to_string(source->ne[0]));
    compile.emplace(std::string(kPrefix) + "input_height", std::to_string(source->ne[1]));
    compile.emplace(std::string(kPrefix) + "channel_stride", std::to_string(source->nb[2] / sizeof(float)));
    compile.emplace(std::string(kPrefix) + "batch_stride", std::to_string(source->nb[3] / sizeof(float)));
    compile.emplace(std::string(kPrefix) + "stride_x", std::to_string(params->stride_x));
    compile.emplace(std::string(kPrefix) + "stride_y", std::to_string(params->stride_y));
    compile.emplace(std::string(kPrefix) + "pad_x", std::to_string(params->pad_x));
    compile.emplace(std::string(kPrefix) + "pad_y", std::to_string(params->pad_y));
    compile.emplace(std::string(kPrefix) + "dilation_x", std::to_string(params->dilation_x));
    compile.emplace(std::string(kPrefix) + "dilation_y", std::to_string(params->dilation_y));
    dispatch.bindings.push_back({ source->storage_root, source->storage_offset, source_span });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_im2col_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.im2col_f32",
        GGML_OP_IM2COL,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_im2col_dispatch,
    });
}

}  // namespace ggml::hrx
