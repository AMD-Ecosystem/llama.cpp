#include "dispatch-copy.h"

#include "dispatch-layout-utils.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr uint64_t         kMaximumCopyElements = uint64_t{ 1 } << 30;
static constexpr KernelCatalogRef kCopyF32Kernel       = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f32");
static constexpr KernelCatalogRef kCopyStridedSourceF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_source_f32");
static constexpr KernelCatalogRef kCopyF32F16Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f32_f16");
static constexpr KernelCatalogRef kCopyStridedSourceF32F16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_source_f32_f16");
static constexpr KernelCatalogRef kCopyStridedSourceF16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_source_f16");
static constexpr KernelCatalogRef kConcatDim0F32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_concat_dim0_f32");
static constexpr KernelCatalogRef kConcatDim0StridedSourceF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_concat_dim0_strided_source_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool packed_layout(const Value & value, size_t element_size) {
    size_t expected_stride = element_size;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.nb[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static bool packed_f32_layout(const Value & value) {
    return packed_layout(value, sizeof(float));
}

static bool packed_f16_layout(const Value & value) {
    return packed_layout(value, sizeof(ggml_fp16_t));
}

static bool make_copy_f16_dispatch(const Value & source, const Value & output, Dispatch & dispatch) {
    const bool f16_types = source.type == GGML_TYPE_F16 && output.type == GGML_TYPE_F16;
    const bool valid_output = output.contiguous && packed_f16_layout(output);
    const bool valid_count = source.element_count > 0 && source.element_count == output.element_count &&
                             static_cast<uint64_t>(source.element_count) <= kMaximumCopyElements;
    if (!f16_types || !valid_output || !valid_count || source.storage == output.storage) {
        return false;
    }

    size_t source_span = 0;
    if (!strided_f16_storage_span_bytes(source, source_span)) {
        return false;
    }
    const size_t source_span_elements = source_span / sizeof(ggml_fp16_t);
    if (source_span_elements > kMaximumCopyElements) {
        return false;
    }

    dispatch.kernel = make_kernel_specialization(kCopyStridedSourceF16Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", source.element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", source_span_elements);
    dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f16.ne0", std::to_string(source.ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f16.ne1", std::to_string(source.ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f16.ne2", std::to_string(source.ne[2]));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f16.stride" + std::to_string(i),
                                                   std::to_string(source.nb[i] / sizeof(ggml_fp16_t)));
    }
    dispatch.bindings.push_back({ source.storage_root, source.storage_offset, source_span });
    dispatch.bindings.push_back({ output.id, 0, output.byte_count });
    return true;
}

static bool match_cont_or_dup_f16_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool is_cont = node->op == GGML_OP_CONT;
    const bool is_dup = node->op == GGML_OP_DUP;
    const bool has_source = node->inputs.size() == 1;
    const bool matches = (is_cont || is_dup) && has_source;
    if (!matches) {
        return false;
    }
    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || output == nullptr) {
        return false;
    }
    Dispatch dispatch;
    if (!make_copy_f16_dispatch(*source, *output, dispatch)) {
        return false;
    }
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool f32_element_strides(const Value & value) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0 || value.nb[i] <= 0 || value.nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    return value.byte_count % sizeof(float) == 0;
}

static void add_concat_strided_source_parameters(KernelSpecialization & kernel, const Value & lhs, const Value & rhs) {
    static constexpr const char * kPrefix = "ggml.concat_dim0_strided_source_f32.";
    kernel.integer_parameters.emplace("lhs_span", lhs.byte_count / sizeof(float));
    kernel.integer_parameters.emplace("rhs_span", rhs.byte_count / sizeof(float));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "lhs_width", std::to_string(lhs.ne[0]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "rhs_width", std::to_string(rhs.ne[0]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "row_count",
                                      std::to_string(lhs.ne[1] * lhs.ne[2] * lhs.ne[3]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "row_ne1", std::to_string(lhs.ne[1]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "row_ne2", std::to_string(lhs.ne[2]));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        kernel.compile_parameters.emplace(std::string(kPrefix) + "lhs_stride" + std::to_string(i),
                                          std::to_string(lhs.nb[i] / sizeof(float)));
        kernel.compile_parameters.emplace(std::string(kPrefix) + "rhs_stride" + std::to_string(i),
                                          std::to_string(rhs.nb[i] / sizeof(float)));
    }
}

static bool make_copy_f32_dispatch(const Value & source, const Value & output, Dispatch & dispatch) {
    if (source.type != GGML_TYPE_F32 || output.type != GGML_TYPE_F32 || !output.contiguous ||
        !packed_f32_layout(output) || !f32_element_strides(source) || source.element_count <= 0 ||
        source.element_count != output.element_count ||
        static_cast<uint64_t>(source.element_count) > kMaximumCopyElements || source.storage == output.storage) {
        return false;
    }

    size_t source_span = source.byte_count;
    if (packed_f32_layout(source)) {
        dispatch.kernel = make_kernel_specialization(kCopyF32Kernel);
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    } else {
        if (!strided_f32_storage_span_bytes(source, source_span)) {
            return false;
        }
        const size_t source_span_elements = source_span / sizeof(float);
        if (source_span_elements > kMaximumCopyElements) {
            return false;
        }
        dispatch.kernel = make_kernel_specialization(kCopyStridedSourceF32Kernel);
        dispatch.kernel.integer_parameters.emplace("source_span", source_span_elements);
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne0", std::to_string(source.ne[0]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne1", std::to_string(source.ne[1]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne2", std::to_string(source.ne[2]));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.stride" + std::to_string(i),
                                                       std::to_string(source.nb[i] / sizeof(float)));
        }
    }
    dispatch.kernel.integer_parameters.emplace("element_count", source.element_count);
    dispatch.bindings.push_back({ source.storage_root, source.storage_offset, source_span });
    dispatch.bindings.push_back({ output.id, 0, output.byte_count });
    return true;
}

static bool match_copy_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CPY || node->inputs.size() != 2) {
        return false;
    }

    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * target = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || target == nullptr || output == nullptr || target->type != GGML_TYPE_F32 ||
        !target->contiguous || !packed_f32_layout(*target) || target->element_count != output->element_count ||
        target->byte_count != output->byte_count || source->storage == target->storage) {
        return false;
    }

    Dispatch dispatch;
    if (!make_copy_f32_dispatch(*source, *output, dispatch)) {
        return false;
    }

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_copy_f32_f16_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CPY || node->inputs.size() != 2) {
        return false;
    }

    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * target = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || target == nullptr || output == nullptr) {
        return false;
    }
    const bool has_expected_types = source->type == GGML_TYPE_F32 && target->type == GGML_TYPE_F16 &&
                                    output->type == GGML_TYPE_F16;
    const bool has_packed_output = output->contiguous && packed_f16_layout(*output) && target->contiguous &&
                                   packed_f16_layout(*target);
    const bool has_matching_size = source->element_count == output->element_count &&
                                   target->element_count == output->element_count &&
                                   target->byte_count == output->byte_count;
    const bool has_safe_size = output->element_count > 0 &&
                               static_cast<uint64_t>(output->element_count) <= kMaximumCopyElements;
    const bool has_distinct_storage = source->storage != target->storage;
    const bool has_f32_source_strides = f32_element_strides(*source);
    const bool can_dispatch = has_expected_types && has_packed_output && has_matching_size && has_safe_size &&
                              has_distinct_storage && has_f32_source_strides;
    if (!can_dispatch) {
        return false;
    }

    size_t source_span = 0;
    if (!strided_f32_storage_span_bytes(*source, source_span)) {
        return false;
    }
    const size_t source_span_elements = source_span / sizeof(float);
    if (source_span_elements > kMaximumCopyElements) {
        return false;
    }

    Dispatch dispatch;
    const bool packed_source = packed_f32_layout(*source);
    const bool vector_count = output->element_count >= 4 && output->element_count % 4 == 0;
    const bool use_vector_copy = packed_source && vector_count;
    if (use_vector_copy) {
        dispatch.kernel = make_kernel_specialization(kCopyF32F16Kernel);
    } else {
        dispatch.kernel = make_kernel_specialization(kCopyStridedSourceF32F16Kernel);
        dispatch.kernel.integer_parameters.emplace("source_span", source_span_elements);
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne0", std::to_string(source->ne[0]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne1", std::to_string(source->ne[1]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne2", std::to_string(source->ne[2]));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.stride" + std::to_string(i),
                                                       std::to_string(source->nb[i] / sizeof(float)));
        }
    }
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ source->storage_root, source->storage_offset, source_span });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_cont_or_dup_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool is_cont = node->op == GGML_OP_CONT;
    const bool is_dup = node->op == GGML_OP_DUP;
    const bool is_copy = is_cont || is_dup;
    const bool has_source = node->inputs.size() == 1;
    const bool matches = is_copy && has_source;
    if (!matches) {
        return false;
    }

    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || output == nullptr) {
        return false;
    }

    Dispatch dispatch;
    if (!make_copy_f32_dispatch(*source, *output, dispatch)) {
        return false;
    }

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_concat_dim0_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2) {
        return false;
    }

    const Value * lhs    = graph_value(context.graph, node->inputs[0]);
    const Value * rhs    = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (lhs == nullptr || rhs == nullptr || output == nullptr || lhs->type != GGML_TYPE_F32 ||
        rhs->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !packed_f32_layout(*output) || !f32_element_strides(*lhs) || !f32_element_strides(*rhs) || lhs->ne[0] <= 0 ||
        lhs->ne[0] > 65536 || rhs->ne[0] <= 0 || rhs->ne[0] > 65536 || output->ne[0] != lhs->ne[0] + rhs->ne[0] ||
        output->element_count <= 0 || static_cast<uint64_t>(output->element_count) > kMaximumCopyElements ||
        lhs->storage == rhs->storage || lhs->storage == output->storage || rhs->storage == output->storage) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (lhs->ne[dim] != rhs->ne[dim] || lhs->ne[dim] != output->ne[dim]) {
            return false;
        }
    }

    const int64_t row_count = output->element_count / output->ne[0];
    if (row_count < 1 || row_count > 1048576) {
        return false;
    }

    Dispatch dispatch;
    if (packed_f32_layout(*lhs) && packed_f32_layout(*rhs)) {
        dispatch.kernel = make_kernel_specialization(kConcatDim0F32Kernel);
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
        dispatch.kernel.integer_parameters.emplace("row_count", row_count);
        dispatch.kernel.integer_parameters.emplace("lhs_span", lhs->element_count);
        dispatch.kernel.integer_parameters.emplace("rhs_span", rhs->element_count);
        dispatch.kernel.compile_parameters.emplace("ggml.concat_dim0_f32.lhs_width", std::to_string(lhs->ne[0]));
        dispatch.kernel.compile_parameters.emplace("ggml.concat_dim0_f32.rhs_width", std::to_string(rhs->ne[0]));
    } else {
        const size_t lhs_span = lhs->byte_count / sizeof(float);
        const size_t rhs_span = rhs->byte_count / sizeof(float);
        if (lhs_span > kMaximumCopyElements || rhs_span > kMaximumCopyElements) {
            return false;
        }
        dispatch.kernel = make_kernel_specialization(kConcatDim0StridedSourceF32Kernel);
        add_concat_strided_source_parameters(dispatch.kernel, *lhs, *rhs);
    }
    dispatch.bindings.push_back({ lhs->id, 0, lhs->byte_count });
    dispatch.bindings.push_back({ rhs->id, 0, rhs->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_copy_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.copy_f32_f16",
        GGML_OP_CPY,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_copy_f32_f16_dispatch,
    });
    registry.add({
        "common.copy_f32",
        GGML_OP_CPY,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_copy_f32_dispatch,
    });
    registry.add({
        "common.cont_f32",
        GGML_OP_CONT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_cont_or_dup_f32_dispatch,
    });
    registry.add({
        "common.cont_f16",
        GGML_OP_CONT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_cont_or_dup_f16_dispatch,
    });
    registry.add({
        "common.dup_f32",
        GGML_OP_DUP,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_cont_or_dup_f32_dispatch,
    });
    registry.add({
        "common.dup_f16",
        GGML_OP_DUP,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_cont_or_dup_f16_dispatch,
    });
    registry.add({
        "common.concat_dim0_f32",
        GGML_OP_CONCAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_concat_dim0_f32_dispatch,
    });
}

}  // namespace ggml::hrx
