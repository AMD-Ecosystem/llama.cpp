#include "dispatch-upscale.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kBilinearF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_upscale_bilinear_f32");
static constexpr size_t kMaximumElements = 134217728;

static bool packed_f32_layout(const Value & value) {
    size_t elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const bool positive_dimension = value.ne[i] > 0;
        const bool expected_stride = value.nb[i] == elements * sizeof(float);
        if (!positive_dimension || !expected_stride) {
            return false;
        }
        if (static_cast<size_t>(value.ne[i]) > kMaximumElements / elements) {
            return false;
        }
        elements *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static bool f32_storage_span(const Value & value, size_t & span) {
    if (value.type != GGML_TYPE_F32) {
        return false;
    }
    size_t last_offset = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const bool positive_dimension = value.ne[i] > 0;
        const bool bounded_dimension = static_cast<uint64_t>(value.ne[i]) <= kMaximumElements;
        const bool element_stride = value.nb[i] % sizeof(float) == 0;
        const bool bounded_stride = value.nb[i] / sizeof(float) <= kMaximumElements;
        if (!positive_dimension || !bounded_dimension || !element_stride || !bounded_stride) {
            return false;
        }
        const size_t extent = static_cast<size_t>(value.ne[i] - 1);
        const bool has_extent = extent != 0;
        const bool extent_fits = !has_extent ||
            value.nb[i] <= (kMaximumElements * sizeof(float) - last_offset - sizeof(float)) / extent;
        if (!extent_fits) {
            return false;
        }
        last_offset += extent * value.nb[i];
    }
    span = last_offset + sizeof(float);
    const bool offset_fits = value.storage_offset <= value.storage_byte_count;
    return offset_fits && span <= value.storage_byte_count - value.storage_offset;
}

static std::string float_config(float value) {
    std::ostringstream out;
    out.precision(std::numeric_limits<float>::max_digits10);
    out << value;
    return out.str();
}

static float axis_scale(int64_t source_size, int64_t output_size, bool align_corners) {
    const bool source_has_corners = source_size > 1;
    const bool output_has_corners = output_size > 1;
    if (align_corners && source_has_corners && output_has_corners) {
        return static_cast<float>(output_size - 1) / static_cast<float>(source_size - 1);
    }
    return static_cast<float>(output_size) / static_cast<float>(source_size);
}

static bool match_bilinear_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    const bool upscale_node = node != nullptr && node->op == GGML_OP_UPSCALE;
    const bool single_input = node != nullptr && node->inputs.size() == 1;
    if (!upscale_node || !single_input) {
        return false;
    }
    const UpscaleParams * params = op_params_as<UpscaleParams>(node->params);
    if (params == nullptr) {
        return false;
    }
    const uint32_t allowed_flags = GGML_SCALE_FLAG_ALIGN_CORNERS;
    const bool bilinear = (params->mode & 0xffu) == GGML_SCALE_MODE_BILINEAR;
    const bool supported_flags = (params->mode & ~(0xffu | allowed_flags)) == 0;
    if (!bilinear || !supported_flags) {
        return false;
    }
    const Value * input = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    const bool has_input = input != nullptr;
    const bool has_output = output != nullptr;
    if (!has_input || !has_output) {
        return false;
    }
    const bool supported_types = input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32;
    const bool distinct_storage = input->storage != output->storage;
    const bool output_packed = output->contiguous && packed_f32_layout(*output);
    const bool output_nonempty = output->element_count > 0;
    const bool output_bounded = static_cast<uint64_t>(output->element_count) <= kMaximumElements;
    const bool supported_output = output_packed && output_nonempty && output_bounded;
    size_t input_span = 0;
    const bool supported_input = f32_storage_span(*input, input_span);
    if (!supported_types || !distinct_storage || !supported_output || !supported_input) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kBilinearF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", input_span / sizeof(float));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.compile_parameters.emplace("ggml.upscale_bilinear_f32.src_ne" + std::to_string(i),
                                                   std::to_string(input->ne[i]));
        dispatch.kernel.compile_parameters.emplace("ggml.upscale_bilinear_f32.dst_ne" + std::to_string(i),
                                                   std::to_string(output->ne[i]));
        dispatch.kernel.compile_parameters.emplace("ggml.upscale_bilinear_f32.stride" + std::to_string(i),
                                                   std::to_string(input->nb[i] / sizeof(float)));
    }
    const bool align_corners = (params->mode & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
    dispatch.kernel.compile_parameters.emplace("ggml.upscale_bilinear_f32.scale0",
                                               float_config(axis_scale(input->ne[0], output->ne[0], align_corners)));
    dispatch.kernel.compile_parameters.emplace("ggml.upscale_bilinear_f32.scale1",
                                               float_config(axis_scale(input->ne[1], output->ne[1], align_corners)));
    dispatch.kernel.compile_parameters.emplace("ggml.upscale_bilinear_f32.pixel_offset",
                                               align_corners ? "0.0" : "0.5");
    dispatch.bindings.push_back({ input->storage_root, input->storage_offset, input_span });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_upscale_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({ "common.upscale_bilinear_f32", GGML_OP_UPSCALE, DispatchMatchKind::SingleOp, 0,
                   DispatchSource::Common, match_bilinear_f32 });
}

}  // namespace ggml::hrx
