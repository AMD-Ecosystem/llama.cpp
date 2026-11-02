#include "dispatch-reduction.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {
constexpr KernelCatalogRef kSumRowsF32 = GGML_HRX_KERNEL_REF("loom_libs", "ggml_sum_rows_f32");
constexpr KernelCatalogRef kSoftMaxF32 = GGML_HRX_KERNEL_REF("loom_libs", "ggml_softmax_f32");

bool match_argmax(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_ARGMAX || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_I32 ||
        !input->contiguous || !output->contiguous || input->ne[0] < 1 || input->ne[0] > 1048576 || input->ne[1] < 1 ||
        input->ne[1] > 2048 || input->ne[2] != 1 || input->ne[3] != 1 || output->element_count != input->ne[1] ||
        input->storage == output->storage) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_argmax_f32"));
    dispatch.kernel.integer_parameters.emplace("row_count", input->ne[1]);
    dispatch.kernel.compile_parameters.emplace("ggml.argmax.width", std::to_string(input->ne[0]));
    dispatch.bindings = {
        { input->id,  0, input->byte_count  },
        { output->id, 0, output->byte_count }
    };
    match.dispatches.push_back(std::move(dispatch));
    match.covered_nodes.push_back(context.root_index);
    return true;
}

bool match_top_k_radix(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_TOP_K || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_I32 ||
        !input->contiguous || !output->contiguous || input->ne[0] <= 512 || input->ne[0] > 131072 ||
        output->ne[0] < 1 || output->ne[0] > 512 || output->ne[0] > input->ne[0] || input->storage == output->storage) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (input->ne[dim] != output->ne[dim]) {
            return false;
        }
    }
    const int64_t rows = input->element_count / input->ne[0];
    if (rows < 1 || rows > 2048) {
        return false;
    }
    const int64_t count = output->ne[0];
    if (rows <= 16 && input->ne[0] >= 4096 && count <= 32) {
        int64_t width     = input->ne[0];
        ValueId ids       = input->id;
        size_t  ids_bytes = input->byte_count;
        int     stage     = 0;
        do {
            const int64_t next_width = width / 256 * count + std::min(count, width % 256);
            const bool    last       = next_width == count;
            const size_t  bytes      = static_cast<size_t>(rows * next_width) * sizeof(int32_t);
            const ValueId target     = last ? output->id : ValueId(context.next_plan_value.value + stage);
            if (!last) {
                match.transients.push_back({ target, "common.top_k.partitions", bytes, 256 });
            }
            Dispatch dispatch;
            dispatch.kernel =
                make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_top_k_radix_partitions_f32"));
            dispatch.kernel.integer_parameters.emplace("row_count", rows);
            dispatch.kernel.compile_parameters.emplace("ggml.top_k.width", std::to_string(input->ne[0]));
            dispatch.kernel.compile_parameters.emplace("ggml.top_k.count", std::to_string(count));
            dispatch.kernel.compile_parameters.emplace("ggml.top_k.stage_width", std::to_string(width));
            dispatch.kernel.compile_parameters.emplace("ggml.top_k.first_pass", stage == 0 ? "1" : "0");
            dispatch.kernel.compile_parameters.emplace("ggml.top_k.last_pass", last ? "1" : "0");
            dispatch.bindings = {
                { input->id, 0, input->byte_count },
                { ids,       0, ids_bytes         },
                { target,    0, bytes             }
            };
            match.dispatches.push_back(std::move(dispatch));
            ids       = target;
            ids_bytes = bytes;
            width     = next_width;
            ++stage;
        } while (width > count);
        match.covered_nodes.push_back(context.root_index);
        return true;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_top_k_radix_f32"));
    dispatch.kernel.integer_parameters.emplace("row_count", rows);
    dispatch.kernel.compile_parameters.emplace("ggml.top_k.width", std::to_string(input->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.top_k.count", std::to_string(output->ne[0]));
    dispatch.bindings = {
        { input->id,  0, input->byte_count  },
        { output->id, 0, output->byte_count }
    };
    match.dispatches.push_back(std::move(dispatch));
    match.covered_nodes.push_back(context.root_index);
    return true;
}

bool match_argsort(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || (node->op != GGML_OP_ARGSORT && node->op != GGML_OP_TOP_K) || node->inputs.size() != 1) {
        return false;
    }
    const bool    top_k  = node->op == GGML_OP_TOP_K;
    const auto *  params = op_params_as<ArgsortParams>(node->params);
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if ((!top_k && params == nullptr) || input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_I32 || !input->contiguous || !output->contiguous ||
        (!top_k && input->ne != output->ne) || output->ne[0] < 1 || output->ne[0] > input->ne[0] || input->ne[0] < 1 ||
        input->ne[0] > 2048 || input->element_count / input->ne[0] > 1048576 || input->storage == output->storage) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (input->ne[dim] != output->ne[dim]) {
            return false;
        }
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_argsort_f32"));
    dispatch.kernel.integer_parameters.emplace("row_count", input->element_count / input->ne[0]);
    dispatch.kernel.compile_parameters.emplace("ggml.argsort.width", std::to_string(input->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.argsort.descending",
                                               top_k || params->order == GGML_SORT_ORDER_DESC ? "1" : "0");
    dispatch.kernel.compile_parameters.emplace("ggml.argsort.output_width", std::to_string(output->ne[0]));
    dispatch.bindings = {
        { input->id,  0, input->byte_count  },
        { output->id, 0, output->byte_count }
    };
    match.dispatches.push_back(std::move(dispatch));
    match.covered_nodes.push_back(context.root_index);
    return true;
}

bool match_softmax(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SOFT_MAX || node->inputs.size() != 1) {
        return false;
    }
    const auto *  params = op_params_as<SoftMaxParams>(node->params);
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (params == nullptr || input == nullptr || output == nullptr || !std::isfinite(params->scale) ||
        params->max_bias != 0.0f || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input->contiguous || !output->contiguous || input->ne != output->ne || input->ne[0] < 1 ||
        input->ne[0] > 32768 || input->element_count < 1 || input->element_count > 134217728 ||
        (input->storage == output->storage && input->storage_offset != output->storage_offset)) {
        return false;
    }
    std::ostringstream scale;
    scale << std::setprecision(std::numeric_limits<float>::max_digits10) << params->scale;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSoftMaxF32);
    dispatch.kernel.integer_parameters.emplace("row_count", input->element_count / input->ne[0]);
    dispatch.kernel.compile_parameters.emplace("ggml.softmax.width", std::to_string(input->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.softmax.scale", scale.str());
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.dispatches.push_back(std::move(dispatch));
    match.covered_nodes.push_back(context.root_index);
    return true;
}

bool match_sum_rows(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || (node->op != GGML_OP_SUM_ROWS && node->op != GGML_OP_SUM) || node->inputs.size() != 1) {
        return false;
    }
    const bool    full_sum = node->op == GGML_OP_SUM;
    const Value * input    = context.graph.values().find(node->inputs[0]);
    const Value * output   = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !output->contiguous || input->ne[0] < 1 || (!full_sum && input->ne[0] > 32768) || output->ne[0] != 1 ||
        (full_sum && (!input->contiguous || input->element_count > 16777216 || output->element_count != 1)) ||
        output->element_count < 1 || output->element_count > 1048576 || input->storage == output->storage ||
        input->byte_count == 0 || input->byte_count % sizeof(float) != 0 ||
        input->byte_count / sizeof(float) > 1073741824) {
        return false;
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (input->ne[dim] < 1 || input->nb[dim] % sizeof(float) != 0 || input->nb[dim] / sizeof(float) > 1073741824 ||
            (!full_sum && dim > 0 && input->ne[dim] != output->ne[dim])) {
            return false;
        }
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSumRowsF32);
    dispatch.kernel.integer_parameters.emplace("row_count", output->element_count);
    auto & config = dispatch.kernel.compile_parameters;
    config.emplace("ggml.sum_rows.width", std::to_string(full_sum ? input->element_count : input->ne[0]));
    config.emplace("ggml.sum_rows.ne1", std::to_string(full_sum ? 1 : input->ne[1]));
    config.emplace("ggml.sum_rows.ne2", std::to_string(full_sum ? 1 : input->ne[2]));
    config.emplace("ggml.sum_rows.source_span", std::to_string(input->byte_count / sizeof(float)));
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        config.emplace("ggml.sum_rows.stride" + std::to_string(dim), std::to_string(input->nb[dim] / sizeof(float)));
    }
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.dispatches.push_back(std::move(dispatch));
    match.covered_nodes.push_back(context.root_index);
    return true;
}
}  // namespace

void register_reduction_dispatches(DispatchRegistryBuilder & registry) {
    registry.add(
        { "common.argmax.f32", GGML_OP_ARGMAX, DispatchMatchKind::SingleOp, 0, DispatchSource::Common, match_argmax });
    registry.add({ "common.top_k.radix.f32", GGML_OP_TOP_K, DispatchMatchKind::SingleOp, 20, DispatchSource::Common,
                   match_top_k_radix });
    registry.add(
        { "common.top_k.f32", GGML_OP_TOP_K, DispatchMatchKind::SingleOp, 0, DispatchSource::Common, match_argsort });
    registry.add(
        { "common.sum.f32", GGML_OP_SUM, DispatchMatchKind::SingleOp, 0, DispatchSource::Common, match_sum_rows });
    registry.add({ "common.argsort.f32", GGML_OP_ARGSORT, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_argsort });
    registry.add({ "common.softmax.f32", GGML_OP_SOFT_MAX, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_softmax });
    registry.add({ "common.sum_rows.f32", GGML_OP_SUM_ROWS, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_sum_rows });
}
}  // namespace ggml::hrx
