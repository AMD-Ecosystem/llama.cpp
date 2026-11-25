#include "dispatch-lightning-indexer.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kLightningIndexerKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_lightning_indexer_f32_f16");

static bool match_lightning_indexer(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_LIGHTNING_INDEXER || node->inputs.size() != 4) {
        return false;
    }
    const Value * q = context.graph.values().find(node->inputs[0]);
    const Value * k = context.graph.values().find(node->inputs[1]);
    const Value * w = context.graph.values().find(node->inputs[2]);
    const Value * m = context.graph.values().find(node->inputs[3]);
    const Value * y = context.graph.values().find(node->output);
    if (!q || !k || !w || !m || !y || !q->contiguous || !k->contiguous || !w->contiguous || !m->contiguous ||
        !y->contiguous || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || w->type != GGML_TYPE_F32 ||
        m->type != GGML_TYPE_F16 || y->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t nq         = q->ne[2];
    const int64_t nk         = k->ne[2];
    const int64_t head_size  = q->ne[0];
    const int64_t head_count = q->ne[1];
    // Keep staged operands within the existing workgroup memory budget.
    if (nq < 1 || nq > 2048 || nk < 1 || nk > 131072 || head_size < 16 || head_size > 256 || head_size % 16 != 0 ||
        head_count < 1 || head_count > 64 || ((head_count + 15) / 16) * 16 * head_size > 8192 || q->ne[3] != 1 ||
        k->ne != std::array<int64_t, 4>{ head_size, 1, nk, 1 } ||
        w->ne != std::array<int64_t, 4>{ head_count, nq, 1, 1 } || m->ne != std::array<int64_t, 4>{ nk, nq, 1, 1 } ||
        y->ne != std::array<int64_t, 4>{ nk, nq, 1, 1 }) {
        return false;
    }
    for (const Value * input : { q, k, w, m }) {
        if (input->storage == y->storage) {
            return false;
        }
    }
    Dispatch dispatch;
    dispatch.kernel                         = make_kernel_specialization(kLightningIndexerKernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("query_count", nq);
    dispatch.kernel.integer_parameters.emplace("key_count", nk);
    dispatch.kernel.compile_parameters.emplace("ggml.indexer.head_size", std::to_string(head_size));
    dispatch.kernel.compile_parameters.emplace("ggml.indexer.head_count", std::to_string(head_count));
    for (const Value * value : { q, k, w, m, y }) {
        dispatch.bindings.push_back({ value->id, 0, value->byte_count });
    }
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_lightning_indexer_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.lightning_indexer_f32_f16",
        GGML_OP_LIGHTNING_INDEXER,
        DispatchMatchKind::SingleOp,
        50,
        DispatchSource::Common,
        match_lightning_indexer,
    });
}

}  // namespace ggml::hrx
