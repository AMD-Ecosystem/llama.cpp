#include "dispatch-norm.h"

#include "ggml.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kNormF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_norm_f32");

static bool match_norm_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool is_norm    = node->op == GGML_OP_NORM;
    const bool has_input  = node->inputs.size() == 1;
    const bool valid_node = is_norm && has_input;
    if (!valid_node) {
        return false;
    }
    if (context.root_index >= context.covered_nodes.size()) {
        return false;
    }
    if (context.covered_nodes[context.root_index]) {
        return false;
    }

    const NormParams * params = op_params_as<NormParams>(node->params);
    if (params == nullptr) {
        return false;
    }
    const bool finite_epsilon      = std::isfinite(params->eps);
    const bool nonnegative_epsilon = params->eps >= 0.0f;
    const bool valid_epsilon       = finite_epsilon && nonnegative_epsilon;
    if (!valid_epsilon) {
        return false;
    }

    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr) {
        return false;
    }
    if (output == nullptr) {
        return false;
    }
    const bool input_f32         = input->type == GGML_TYPE_F32;
    const bool output_f32        = output->type == GGML_TYPE_F32;
    const bool input_contiguous  = input->contiguous;
    const bool output_contiguous = output->contiguous;
    const bool same_shape        = input->ne == output->ne;
    const bool separate_storage  = input->storage_root != output->storage_root;
    const bool supported_tensors = input_f32 && output_f32 && input_contiguous && output_contiguous && same_shape &&
                                   separate_storage;
    if (!supported_tensors) {
        return false;
    }

    const int64_t hidden_size     = input->ne[0];
    const bool    hidden_positive = hidden_size >= 1;
    const bool    hidden_bounded  = hidden_size <= 32768;
    const bool supported_hidden_size = hidden_positive && hidden_bounded;
    if (!supported_hidden_size) {
        return false;
    }
    const bool has_elements = input->element_count > 0;
    if (!has_elements) {
        return false;
    }
    const bool complete_rows = input->element_count % hidden_size == 0;
    if (!complete_rows) {
        return false;
    }
    const int64_t row_count     = input->element_count / hidden_size;
    const bool    rows_positive = row_count >= 1;
    const bool    rows_bounded  = row_count <= 1048576;
    const bool supported_row_count = rows_positive && rows_bounded;
    if (!supported_row_count) {
        return false;
    }

    std::ostringstream epsilon;
    epsilon.precision(9);
    epsilon << params->eps;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kNormF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("row_count", row_count);
    dispatch.kernel.compile_parameters.emplace("ggml.norm_f32.hidden_size", std::to_string(hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.norm_f32.input_stride", std::to_string(hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.norm_f32.epsilon", epsilon.str());
    dispatch.bindings.push_back({ input->storage_root, input->storage_offset, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_norm_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.norm_f32",
        GGML_OP_NORM,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_norm_f32_dispatch,
    });
}

}  // namespace ggml::hrx
