#include "dispatch-soft-max.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMaskedSoftMaxF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_masked_soft_max_f32");

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (lhs.ne[dim] != rhs.ne[dim]) {
            return false;
        }
    }
    return true;
}

static bool disjoint_storage(const Value & lhs, const Value & rhs) {
    if (lhs.storage_root != rhs.storage_root) {
        return true;
    }
    const size_t lhs_begin = lhs.storage_offset;
    const size_t rhs_begin = rhs.storage_offset;
    const bool lhs_fits = lhs.byte_count <= std::numeric_limits<size_t>::max() - lhs_begin;
    const bool rhs_fits = rhs.byte_count <= std::numeric_limits<size_t>::max() - rhs_begin;
    if (!(lhs_fits && rhs_fits)) {
        return false;
    }
    const size_t lhs_end   = lhs_begin + lhs.byte_count;
    const size_t rhs_end   = rhs_begin + rhs.byte_count;
    return lhs_end <= rhs_begin || rhs_end <= lhs_begin;
}

static bool match_masked_soft_max_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool is_soft_max = node->op == GGML_OP_SOFT_MAX;
    const bool has_mask = node->inputs.size() == 2;
    if (!(is_soft_max && has_mask)) {
        return false;
    }

    const SoftMaxParams * params = op_params_as<SoftMaxParams>(node->params);
    const Value * scores = context.graph.values().find(node->inputs[0]);
    const Value * mask   = context.graph.values().find(node->inputs[1]);
    const Value * output = context.graph.values().find(node->output);
    const bool has_params = params != nullptr;
    const bool has_values = scores != nullptr && mask != nullptr && output != nullptr;
    if (!(has_params && has_values)) {
        return false;
    }

    const bool f32_values = scores->type == GGML_TYPE_F32 && mask->type == GGML_TYPE_F32 &&
                            output->type == GGML_TYPE_F32;
    const bool packed_values = scores->contiguous && mask->contiguous && output->contiguous;
    const bool matching_scores = same_shape(*scores, *output);
    const bool broadcast_mask = mask->ne[0] == scores->ne[0] && mask->ne[1] >= scores->ne[1] &&
                                mask->ne[2] > 0 && mask->ne[3] > 0 &&
                                scores->ne[2] % mask->ne[2] == 0 && scores->ne[3] % mask->ne[3] == 0;
    const bool supported_width = scores->ne[0] >= 64 && scores->ne[0] <= 4096;
    const bool supported_rows = supported_width && output->element_count > 0 &&
                                output->element_count / scores->ne[0] <= 1048576;
    const bool supported_mask_rows = mask->ne[1] > 0 && mask->ne[1] <= 1048576;
    const bool supported_params = std::isfinite(params->scale) && params->max_bias == 0.0f;
    const bool safe_storage = disjoint_storage(*mask, *output) &&
                              (disjoint_storage(*scores, *output) ||
                               (scores->storage_root == output->storage_root &&
                                scores->storage_offset == output->storage_offset &&
                                scores->byte_count == output->byte_count));
    const bool supported = f32_values && packed_values && matching_scores && broadcast_mask && supported_width &&
                           supported_rows && supported_mask_rows && supported_params && safe_storage;
    if (!supported) {
        return false;
    }

    std::ostringstream scale;
    scale.precision(std::numeric_limits<float>::max_digits10);
    scale << params->scale;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMaskedSoftMaxF32Kernel);
    dispatch.kernel.integer_parameters.emplace("row_count", output->element_count / scores->ne[0]);
    dispatch.kernel.compile_parameters.emplace("ggml.masked_soft_max_f32.width", std::to_string(scores->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.masked_soft_max_f32.ne1", std::to_string(scores->ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.masked_soft_max_f32.ne2", std::to_string(scores->ne[2]));
    dispatch.kernel.compile_parameters.emplace("ggml.masked_soft_max_f32.mask_ne1", std::to_string(mask->ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.masked_soft_max_f32.mask_ne2", std::to_string(mask->ne[2]));
    dispatch.kernel.compile_parameters.emplace("ggml.masked_soft_max_f32.mask_ne3", std::to_string(mask->ne[3]));
    dispatch.kernel.compile_parameters.emplace("ggml.masked_soft_max_f32.scale", scale.str());
    dispatch.bindings.push_back({ scores->id, 0, scores->byte_count });
    dispatch.bindings.push_back({ mask->id, 0, mask->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_soft_max_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.masked_soft_max_f32",
        GGML_OP_SOFT_MAX,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_masked_soft_max_f32,
    });
}

}  // namespace ggml::hrx
