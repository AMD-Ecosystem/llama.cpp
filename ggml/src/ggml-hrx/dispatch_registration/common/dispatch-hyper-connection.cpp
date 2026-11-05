#include "dispatch-hyper-connection.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <utility>

namespace ggml::hrx {
namespace {
static constexpr KernelCatalogRef kComb = GGML_HRX_KERNEL_REF("loom_libs", "ggml_hc_comb_f32");
static constexpr KernelCatalogRef kPre  = GGML_HRX_KERNEL_REF("loom_libs", "ggml_hc_pre_f32");
static constexpr KernelCatalogRef kPost = GGML_HRX_KERNEL_REF("loom_libs", "ggml_hc_post_f32");

static bool match_hyper_connection(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr) {
        return false;
    }
    const bool comb = node->op == GGML_OP_DSV4_HC_COMB;
    const bool pre  = node->op == GGML_OP_DSV4_HC_PRE;
    const bool post = node->op == GGML_OP_DSV4_HC_POST;
    const bool identity = post && node->inputs.size() == 3;
    if ((!comb && !pre && !post) || node->inputs.size() != (comb ? 3 : pre ? 2 : identity ? 3 : 4)) {
        return false;
    }
    const Value * output = context.graph.values().find(node->output);
    if (!output || output->type != GGML_TYPE_F32 || !output->contiguous || output->ne[3] != 1) {
        return false;
    }
    const Value * inputs[4] = {};
    for (size_t i = 0; i < node->inputs.size(); ++i) {
        inputs[i] = context.graph.values().find(node->inputs[i]);
        if (!inputs[i] || inputs[i]->type != GGML_TYPE_F32 || !inputs[i]->contiguous || inputs[i]->ne[3] != 1 ||
            inputs[i]->storage == output->storage) {
            return false;
        }
    }
    const int64_t tokens  = pre ? inputs[0]->ne[2] : inputs[0]->ne[1];
    const int64_t embd    = comb ? 1 : inputs[0]->ne[0];
    const int64_t streams = comb ? 4 : pre ? inputs[0]->ne[1] : inputs[1]->ne[1];
    if (tokens < 1 || tokens > 32768 || embd < 1 || embd > 32768 || streams < 1 || streams > 128 ||
        tokens * embd > 134217728 / streams) {
        return false;
    }
    const auto * pre_params = pre ? op_params_as<HyperConnectionPreParams>(node->params) : nullptr;
    if (comb) {
        if (inputs[0]->ne[0] != 24 || inputs[0]->ne[2] != 1 || inputs[1]->ne[0] < 3 || inputs[1]->ne[1] != 1 ||
            inputs[1]->ne[2] != 1 || inputs[2]->element_count != 24 || output->ne[0] != 4 || output->ne[1] != 4 ||
            output->ne[2] != tokens) {
            return false;
        }
    } else if (pre) {
        if (pre_params == nullptr || !std::isfinite(pre_params->scale) || output->ne[0] != embd ||
            output->ne[1] != tokens || output->ne[2] != 1 ||
            (pre_params->gated ? inputs[1]->ne != inputs[0]->ne :
                                 inputs[1]->ne != std::array<int64_t, 4>{ streams, tokens, 1, 1 })) {
            return false;
        }
    } else if (inputs[0]->ne[2] != 1 || inputs[1]->ne[0] != embd || inputs[1]->ne[2] != tokens ||
               inputs[2]->ne != std::array<int64_t, 4>{ streams, tokens, 1, 1 } ||
               (!identity && inputs[3]->ne != std::array<int64_t, 4>{ streams, streams, tokens, 1 }) ||
               output->ne != inputs[1]->ne) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(comb ? kComb : pre ? kPre : kPost);
    dispatch.kernel.compile_parameters.emplace("ggml.hc.tokens", std::to_string(tokens));
    if (comb) {
        const auto * params = op_params_as<HyperConnectionCombParams>(node->params);
        if (!params || !std::isfinite(params->eps) || params->eps <= 0 || params->n_iter < 1 || params->n_iter > 128) {
            return false;
        }
        dispatch.kernel.compile_parameters.emplace("ggml.hc.eps", float_config_value(params->eps));
        dispatch.kernel.compile_parameters.emplace("ggml.hc.n_iter", std::to_string(params->n_iter));
    } else {
        dispatch.kernel.compile_parameters.emplace("ggml.hc.embd", std::to_string(embd));
        dispatch.kernel.compile_parameters.emplace("ggml.hc.streams", std::to_string(streams));
        if (pre) {
            dispatch.kernel.compile_parameters.emplace("ggml.hc.scale", float_config_value(pre_params->scale));
            dispatch.kernel.compile_parameters.emplace("ggml.hc.gated", pre_params->gated ? "1" : "0");
        } else {
            dispatch.kernel.compile_parameters.emplace("ggml.hc.identity", identity ? "1" : "0");
        }
    }
    for (size_t i = 0; i < node->inputs.size(); ++i) {
        dispatch.bindings.push_back({ inputs[i]->id, 0, inputs[i]->byte_count });
    }
    if (identity) {
        // The identity specialization does not read the mixing matrix.
        dispatch.bindings.push_back({ inputs[1]->id, 0, inputs[1]->byte_count });
    }
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}
}  // namespace

void register_hyper_connection_dispatches(DispatchRegistryBuilder & registry) {
    for (ggml_op op : { GGML_OP_DSV4_HC_COMB, GGML_OP_DSV4_HC_PRE, GGML_OP_DSV4_HC_POST }) {
        registry.add({ "common.hyper_connection", op, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                       match_hyper_connection });
    }
}
}  // namespace ggml::hrx
