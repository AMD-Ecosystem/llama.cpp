#include "dispatch-registry.h"

#include "backend-buffer-binding.h"

#include "common/dispatch-common.h"
#include "llm/dispatch-attention-qkv.h"
#include "llm/dispatch-gated-delta-net.h"
#include "llm/dispatch-ssm-conv.h"
#include "qwen/dispatch-llm-profiles.h"
#include "qwen/dispatch-qwen.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr size_t kOpCount = static_cast<size_t>(GGML_OP_COUNT);

static const std::vector<DispatchRegistration> kEmptyRegistrations;
static const std::vector<DispatchActivationConsumerRegistration> kEmptyActivationConsumers;

static bool valid_root_op(ggml_op op) {
    return op >= 0 && static_cast<size_t>(op) < kOpCount;
}

static void sort_registrations(std::vector<DispatchRegistration> & registrations) {
    std::stable_sort(
        registrations.begin(), registrations.end(),
        [](const DispatchRegistration & lhs, const DispatchRegistration & rhs) { return lhs.priority > rhs.priority; });
}

static bool qwen_dispatch_disabled_from_environment() {
    const char * value = std::getenv("GGML_HRX_DISABLE_QWEN_DISPATCH");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

static bool qwen30b_attention_needs_native_weight(const Graph & graph, const Value & value) {
    if (qwen_dispatch_disabled_from_environment()) {
        return false;
    }
    const ggml_tensor * tensor = value.tensor;
    const bool has_tensor = tensor != nullptr;
    const bool supported_type = value.type == GGML_TYPE_Q4_K || value.type == GGML_TYPE_Q6_K;
    const bool matrix = value.ne[2] == 1 && value.ne[3] == 1;
    const bool block_weight = has_tensor && std::strncmp(tensor->name, "blk.", 4) == 0;
    const bool candidate = has_tensor && supported_type && matrix && block_weight;
    if (!candidate) {
        return false;
    }

    // These four projections have native consumers in the measured Qwen3-30B
    // prefill/decode graph. Packing them also materializes a second native copy.
    const char * layer_start = tensor->name + 4;
    const bool starts_with_digit = *layer_start >= '0' && *layer_start <= '9';
    if (!starts_with_digit) {
        return false;
    }
    char * layer_end = nullptr;
    const long layer = std::strtol(layer_start, &layer_end, 10);
    const bool valid_layer = layer >= 0;
    const bool valid_separator = layer_end != nullptr && *layer_end == '.';
    const bool named_layer = valid_layer && valid_separator;
    if (!named_layer) {
        return false;
    }

    const char * role = layer_end + 1;
    const int64_t hidden_size = kQwen30BMoeDispatchProfile.hidden_size;
    const bool q4 = value.type == GGML_TYPE_Q4_K;
    const bool query_role = std::strcmp(role, "attn_q.weight") == 0;
    const bool key_role = std::strcmp(role, "attn_k.weight") == 0;
    const bool value_role = std::strcmp(role, "attn_v.weight") == 0;
    const bool output_role = std::strcmp(role, "attn_output.weight") == 0;
    const bool query_shape = value.ne[0] == hidden_size && value.ne[1] == 4096;
    const bool key_value_shape = value.ne[0] == hidden_size && value.ne[1] == 512;
    const bool output_shape = value.ne[0] == 4096 && value.ne[1] == hidden_size;
    const bool query = q4 && query_role && query_shape;
    const bool key = q4 && key_role && key_value_shape;
    const bool value_projection = value_role && key_value_shape;
    const bool output = q4 && output_role && output_shape;
    const bool attention_projection = query || key || value_projection || output;
    if (!attention_projection) {
        return false;
    }

    // K/V dimensions alone also occur in other models. Require this layer's
    // complete attention quartet and its Qwen30B MoE expert profile.
    const std::string prefix(tensor->name, static_cast<size_t>(layer_end - tensor->name + 1));
    bool has_query = false;
    bool has_key = false;
    bool has_value = false;
    bool has_output = false;
    bool has_experts = false;
    for (const Value & peer : graph.values().values()) {
        const ggml_tensor * peer_tensor = peer.tensor;
        const bool has_peer_tensor = peer_tensor != nullptr;
        const bool same_layer = has_peer_tensor &&
                                std::strncmp(peer_tensor->name, prefix.c_str(), prefix.size()) == 0;
        if (!same_layer) {
            continue;
        }
        const char * peer_role = peer_tensor->name + prefix.size();
        const bool peer_matrix = peer.ne[2] == 1 && peer.ne[3] == 1;
        const bool peer_q4 = peer.type == GGML_TYPE_Q4_K;
        const bool peer_q6 = peer.type == GGML_TYPE_Q6_K;
        const bool peer_hidden = peer.ne[0] == hidden_size;
        const bool peer_kv = peer_matrix && peer_hidden && peer.ne[1] == 512;
        const bool peer_query = peer_q4 && peer_matrix && peer_hidden && peer.ne[1] == 4096;
        const bool peer_output = peer_q4 && peer_matrix && peer.ne[0] == 4096 && peer.ne[1] == hidden_size;
        const bool peer_experts = peer_q4 && peer.ne[0] == hidden_size &&
                                  peer.ne[1] == kQwen30BMoeDispatchProfile.expert_hidden_size &&
                                  peer.ne[2] == kQwen30BMoeDispatchProfile.expert_count && peer.ne[3] == 1;
        has_query = has_query || (peer_query && std::strcmp(peer_role, "attn_q.weight") == 0);
        has_key = has_key || (peer_q4 && peer_kv && std::strcmp(peer_role, "attn_k.weight") == 0);
        has_value = has_value || ((peer_q4 || peer_q6) && peer_kv &&
                                  std::strcmp(peer_role, "attn_v.weight") == 0);
        has_output = has_output || (peer_output && std::strcmp(peer_role, "attn_output.weight") == 0);
        has_experts = has_experts || (peer_experts && std::strcmp(peer_role, "ffn_gate_exps.weight") == 0);
    }
    return has_query && has_key && has_value && has_output && has_experts;
}

static bool qwen17b_tied_token_embedding_needs_native_weight(const Value & value) {
    const ggml_tensor * tensor = value.tensor;
    const bool named_embedding = tensor != nullptr && std::strcmp(tensor->name, "token_embd.weight") == 0;
    const bool q6_weight = value.type == GGML_TYPE_Q6_K;
    const bool tied_source_shape = value.ne[0] == 2048 && value.ne[1] == kQwen3VocabularyCount &&
                                   value.ne[2] == 1 && value.ne[3] == 1;
    // The measured Qwen3-1.7B GGUF omits output.weight: this exact source
    // feeds both embedding lookup and the native output projection.
    return named_embedding && q6_weight && tied_source_shape;
}

static bool dispatch_has_eligible_weight_layouts(const DispatchMatchContext & context, const DispatchMatch & match) {
    auto eligible = [&](const std::vector<Dispatch> & dispatches) {
        for (const Dispatch & dispatch : dispatches) {
            for (const DispatchBinding & binding : dispatch.bindings) {
                if (binding.layout == kNativeWeightLayout) {
                    continue;
                }
                const Value * value = context.graph.values().find(binding.value);
                const bool has_graph_value = value != nullptr;
                const bool layout_is_eligible =
                    !has_graph_value ||
                    dispatch_weight_layout_eligible_for_graph(context.graph, *value, context.row64_weight_policy);
                const bool has_ineligible_weight = has_graph_value && !layout_is_eligible;
                if (has_ineligible_weight) {
                    return false;
                }
            }
        }
        return true;
    };
    return eligible(match.initialization_dispatches) && eligible(match.dispatches);
}

static DispatchRegistry build_registry(bool include_qwen) {
    DispatchRegistryBuilder builder;
    register_common_dispatches(builder);
    register_llm_attention_qkv_dispatches(builder);
    register_llm_gated_delta_net_dispatch(builder);
    register_llm_ssm_conv_dispatch(builder);
    if (include_qwen) {
        register_qwen_dispatches(builder);
    }
    return builder.build();
}

}  // namespace

bool dispatch_weight_layout_eligible(const Value & value) {
    return ggml_backend_hrx_weight_layout_eligible(value.tensor);
}

bool dispatch_weight_layout_eligible_for_graph(const Graph & graph, const Value & value, bool row64_weight_policy) {
    if (!row64_weight_policy) {
        return dispatch_weight_layout_eligible(value);
    }
    const bool needs_native_embedding = qwen17b_tied_token_embedding_needs_native_weight(value);
    const bool needs_native_attention = qwen30b_attention_needs_native_weight(graph, value);
    return !needs_native_embedding && !needs_native_attention && dispatch_weight_layout_eligible(value);
}

bool dispatch_weight_is_qwen30b_native_attention(const Graph & graph, const Value & value) {
    return qwen30b_attention_needs_native_weight(graph, value);
}

bool DispatchRegistry::match(const DispatchMatchContext & context, DispatchMatch & match) const {
    return this->match(context, match, nullptr);
}

bool DispatchRegistry::match(const DispatchMatchContext & context,
                             DispatchMatch &              match,
                             DispatchMatchDiagnostics *   diagnostics) const {
    if (context.root_node == nullptr || !valid_root_op(context.root_node->op) || registrations_by_root_.empty()) {
        return false;
    }
    if (diagnostics != nullptr) {
        diagnostics->root_op = context.root_node->op;
        diagnostics->attempts.clear();
    }
    const std::vector<DispatchRegistration> & registrations =
        registrations_by_root_[static_cast<size_t>(context.root_node->op)].ordered;
    for (const DispatchRegistration & registration : registrations) {
        DispatchMatch candidate;
        const bool matched = registration.matcher != nullptr && registration.matcher(context, candidate);
        const bool eligible = matched && dispatch_has_eligible_weight_layouts(context, candidate);
        if (eligible) {
            if (diagnostics != nullptr) {
                diagnostics->attempts.push_back({
                    registration.name != nullptr ? registration.name : "",
                    registration.root_op,
                    registration.kind,
                    registration.priority,
                    registration.source,
                    true,
                    candidate.covered_nodes,
                    candidate.status.errors(),
                });
            }
            match = std::move(candidate);
            return true;
        }
        if (diagnostics != nullptr) {
            diagnostics->attempts.push_back({
                registration.name != nullptr ? registration.name : "",
                registration.root_op,
                registration.kind,
                registration.priority,
                registration.source,
                false,
                candidate.covered_nodes,
                candidate.status.errors(),
            });
        }
        match.status.append(candidate.status);
    }
    return false;
}

const std::vector<DispatchRegistration> & DispatchRegistry::registrations_for_root(ggml_op root_op) const {
    if (!valid_root_op(root_op) || registrations_by_root_.empty()) {
        return kEmptyRegistrations;
    }
    return registrations_by_root_[static_cast<size_t>(root_op)].ordered;
}

const std::vector<DispatchActivationConsumerRegistration> &
DispatchRegistry::activation_consumers_for_root(ggml_op root_op) const {
    if (!valid_root_op(root_op) || activation_consumers_by_root_.empty()) {
        return kEmptyActivationConsumers;
    }
    return activation_consumers_by_root_[static_cast<size_t>(root_op)];
}

void DispatchRegistryBuilder::add(DispatchRegistration registration) {
    if (!valid_root_op(registration.root_op) || registration.matcher == nullptr) {
        return;
    }
    if (registry_.registrations_by_root_.empty()) {
        registry_.registrations_by_root_.resize(kOpCount);
    }
    DispatchRegistry::RegistrationGroup & group =
        registry_.registrations_by_root_[static_cast<size_t>(registration.root_op)];
    if (registration.kind == DispatchMatchKind::Fused) {
        group.fused.push_back(std::move(registration));
    } else {
        group.single_op.push_back(registration);
        registry_.single_op_registrations_.push_back(std::move(registration));
    }
}

void DispatchRegistryBuilder::add_activation_consumer(DispatchActivationConsumerRegistration registration) {
    if (!valid_root_op(registration.root_op) || registration.accepts == nullptr ||
        registration.accepted_formats == DispatchActivationInputNone) {
        return;
    }
    if (registry_.activation_consumers_by_root_.empty()) {
        registry_.activation_consumers_by_root_.resize(kOpCount);
    }
    registry_.activation_consumers_by_root_[static_cast<size_t>(registration.root_op)].push_back(
        std::move(registration));
}

DispatchRegistry DispatchRegistryBuilder::build() {
    if (registry_.registrations_by_root_.empty()) {
        registry_.registrations_by_root_.resize(kOpCount);
    }
    if (registry_.activation_consumers_by_root_.empty()) {
        registry_.activation_consumers_by_root_.resize(kOpCount);
    }
    for (DispatchRegistry::RegistrationGroup & group : registry_.registrations_by_root_) {
        sort_registrations(group.fused);
        sort_registrations(group.single_op);
        group.ordered = group.fused;
        group.ordered.insert(group.ordered.end(), group.single_op.begin(), group.single_op.end());
    }
    sort_registrations(registry_.single_op_registrations_);
    return std::move(registry_);
}

const DispatchRegistry * find_dispatch_registry(const DispatchTarget & target) {
    static const DispatchRegistry gfx1100_registry              = build_registry(true);
    static const DispatchRegistry gfx1100_generic_only_registry = build_registry(false);
    static const DispatchRegistry gfx1151_registry              = build_registry(true);
    static const DispatchRegistry gfx1151_generic_only_registry = build_registry(false);

    const bool qwen_disabled = qwen_dispatch_disabled_from_environment();

    if (target.architecture == "gfx1100") {
        return qwen_disabled ? &gfx1100_generic_only_registry : &gfx1100_registry;
    }
    if (target.architecture == "gfx1151") {
        return qwen_disabled ? &gfx1151_generic_only_registry : &gfx1151_registry;
    }
    return nullptr;
}

}  // namespace ggml::hrx
