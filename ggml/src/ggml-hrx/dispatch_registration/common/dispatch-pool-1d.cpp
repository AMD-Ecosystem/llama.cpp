#include "dispatch-pool-1d.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kFullMaxPoolF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_pool_1d_full_max_f32");

static bool match_full_max_pool_1d_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool one_input = node->inputs.size() == 1;
    if (!one_input) {
        return false;
    }

    const Pool1DParams * params = op_params_as<Pool1DParams>(node->params);
    const Value * input = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    const bool has_params = params != nullptr;
    const bool has_input = input != nullptr;
    const bool has_output = output != nullptr;
    const bool all_present = has_params && has_input && has_output;
    if (!all_present) {
        return false;
    }

    const bool f32_values = input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32;
    const bool contiguous_values = input->contiguous && output->contiguous;
    const bool full_row = output->ne[0] == 1 && input->ne[0] == params->kernel;
    const bool matching_rows = input->ne[1] == output->ne[1] && input->ne[2] == output->ne[2] &&
                               input->ne[3] == output->ne[3];
    const bool exact_window = params->stride == params->kernel && params->padding == 0;
    const bool bounded_width = input->ne[0] > 0 && input->ne[0] <= 32768;
    const bool bounded_rows = output->element_count > 0 && output->element_count <= 1048576;
    const bool bounded_input = input->element_count <= 134217728;
    const bool bounded = bounded_width && bounded_rows && bounded_input;
    const bool separate_storage = input->storage_root != output->storage_root;
    const bool supported = params->op == GGML_OP_POOL_MAX && f32_values && contiguous_values && full_row &&
                           matching_rows && exact_window && bounded && separate_storage;
    if (!supported) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kFullMaxPoolF32Kernel);
    dispatch.kernel.integer_parameters.emplace("row_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("row_width", input->ne[0]);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_pool_1d_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.pool_1d_full_max_f32",
        GGML_OP_POOL_1D,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_full_max_pool_1d_f32,
    });
}

}  // namespace ggml::hrx
