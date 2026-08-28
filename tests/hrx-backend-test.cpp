#include "backend-buffer-binding.h"
#include "backend-context.h"
#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "dispatch/dispatch-scheduler.h"
#include "dispatch_registration/dispatch-registry.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph/graph-diagnostics.h"
#include "graph/graph-matcher.h"
#include "graph/graph-traversal.h"
#include "graph/graph.h"
#include "hrx-interop-utils.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/command-program-executor.h"
#include "runtime/graph-executor.h"
#include "runtime/graph-program-cache.h"
#include "runtime/graph-replay.h"
#include "runtime/loom-kernel-jit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static hrx_buffer_t dummy_hrx_buffer(uintptr_t value) {
    return reinterpret_cast<hrx_buffer_t>(value);
}

static hrx_stream_t dummy_hrx_stream(uintptr_t value) {
    return reinterpret_cast<hrx_stream_t>(value);
}

static hrx_graph_exec_t dummy_hrx_graph_exec(uintptr_t value) {
    return reinterpret_cast<hrx_graph_exec_t>(value);
}

static bool contains_value_id(const std::vector<ggml::hrx::ValueId> & ids, ggml::hrx::ValueId id) {
    for (const ggml::hrx::ValueId candidate : ids) {
        if (candidate == id) {
            return true;
        }
    }
    return false;
}

static bool command_program_verifies(const ggml::hrx::CommandProgram & program) {
    return ggml::hrx::verify_command_program(program, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151").valid();
}

static bool async_jit_expected_from_environment() {
    const char * value = std::getenv("GGML_HRX_ASYNC_JIT");
    return value == nullptr ||
           (std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
            std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0);
}

static ggml::hrx::CommandProgram copy_command_program_shape(const ggml::hrx::CommandProgram & program) {
    ggml::hrx::CommandProgram copy;
    copy.initialization_commands  = program.initialization_commands;
    copy.commands                 = program.commands;
    copy.transients               = program.transients;
    copy.completion_counters      = program.completion_counters;
    copy.constant_initializations = program.constant_initializations;
    return copy;
}

static bool status_contains(const ggml::hrx::Status & status, const char * text) {
    for (const std::string & message : status.errors()) {
        if (message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool string_contains(const std::string & value, const char * text) {
    return value.find(text) != std::string::npos;
}

static std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

static std::vector<std::filesystem::path> list_directories(const std::filesystem::path & path) {
    std::vector<std::filesystem::path> directories;
    if (!std::filesystem::exists(path)) {
        return directories;
    }
    for (const std::filesystem::directory_entry & entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_directory()) {
            directories.push_back(entry.path());
        }
    }
    std::sort(directories.begin(), directories.end());
    return directories;
}

struct EnvironmentVariableGuard {
    const char * name;
    bool         had_value = false;
    std::string  value;

    explicit EnvironmentVariableGuard(const char * name) : name(name) {
        const char * current = std::getenv(name);
        if (current != nullptr) {
            had_value = true;
            value     = current;
        }
    }

    ~EnvironmentVariableGuard() {
        if (had_value) {
            setenv(name, value.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }

    void set(const std::filesystem::path & path) const { setenv(name, path.string().c_str(), 1); }

    void unset() const { unsetenv(name); }
};

static std::filesystem::path fresh_test_directory(const char * name) {
    static uint64_t       sequence = 0;
    std::filesystem::path path     = std::filesystem::temp_directory_path() /
                                 ("hrx-backend-test-" + std::string(name) + "-" + std::to_string(sequence++));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

static void require_hrx_status(hrx_status_t status) {
    if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
        std::fprintf(stderr, "HRX status failed: %s\n", error->c_str());
        std::abort();
    }
}

static ggml::hrx::DispatchTarget test_dispatch_target() {
    return { "gfx1151" };
}

static const ggml::hrx::DispatchRegistry & test_dispatch_registry() {
    const ggml::hrx::DispatchRegistry * registry = ggml::hrx::find_dispatch_registry(test_dispatch_target());
    REQUIRE(registry != nullptr);
    return *registry;
}

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

static void require_compile_parameter(const ggml::hrx::Dispatch & dispatch,
                                      const char *                name,
                                      const std::string &         value) {
    const auto found = dispatch.kernel.compile_parameters.find(name);
    REQUIRE(found != dispatch.kernel.compile_parameters.end());
    if (found->second != value) {
        std::fprintf(stderr, "compile parameter %s expected '%s' got '%s'\n", name, value.c_str(),
                     found->second.c_str());
        std::abort();
    }
}

static constexpr int64_t kQwenFlashHeadSize       = 128;
static constexpr int64_t kQwenRouterExpertCount   = 128;
static constexpr int64_t kQwenRouterRouteCount    = 8;
static constexpr int64_t kQwenMoeHiddenSize       = 2048;
static constexpr int64_t kQwenMoeIntermediateSize = 768;

static size_t qwen_expert_table_size(int64_t token_count, int64_t expert_count = kQwenRouterExpertCount) {
    return static_cast<size_t>(expert_count + expert_count * token_count) * sizeof(int32_t);
}

static size_t qwen_partition_table_size(int64_t token_count,
                                        int64_t route_count  = kQwenRouterRouteCount,
                                        int64_t expert_count = kQwenRouterExpertCount) {
    const int64_t assignment_count           = token_count * route_count;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + expert_count) * sizeof(int32_t);
}

static size_t qwen_q8_1_x4_size(int64_t token_count, int64_t hidden_size) {
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static int64_t matmul_weight_format_config(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return 4;
        case GGML_TYPE_Q6_K:
            return 6;
        case GGML_TYPE_Q8_0:
            return 80;
        case GGML_TYPE_Q8_1:
            return 81;
        case GGML_TYPE_F16:
            return 16;
        case GGML_TYPE_F32:
            return 32;
        default:
            REQUIRE(false);
            return 0;
    }
}

static std::vector<int32_t> make_qwen_route_ids_iota(int64_t token_count,
                                                     int64_t route_count,
                                                     int64_t route_stride,
                                                     int64_t expert_count) {
    std::vector<int32_t> route_ids(static_cast<size_t>(token_count * route_stride), -1);
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t route = 0; route < route_count; ++route) {
            route_ids[static_cast<size_t>(token * route_stride + route)] =
                static_cast<int32_t>((token * route_count + route) % expert_count);
        }
    }
    return route_ids;
}

static std::vector<int32_t> make_qwen_expert_table_reference(const std::vector<int32_t> & route_ids,
                                                             int64_t                      token_count,
                                                             int64_t                      route_count,
                                                             int64_t                      route_stride,
                                                             int64_t                      expert_count) {
    std::vector<int32_t> expert_table(qwen_expert_table_size(token_count, expert_count) / sizeof(int32_t), -1);
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        expert_table[static_cast<size_t>(expert)] = 0;
    }
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t route = 0; route < route_count; ++route) {
            const int32_t expert = route_ids[static_cast<size_t>(token * route_stride + route)];
            REQUIRE(expert >= 0);
            REQUIRE(expert < expert_count);
            int32_t &    count = expert_table[static_cast<size_t>(expert)];
            const size_t assignment_offset =
                static_cast<size_t>(expert_count + static_cast<int64_t>(expert) * token_count + count);
            expert_table[assignment_offset] = static_cast<int32_t>(token * route_count + route);
            ++count;
        }
    }
    return expert_table;
}

static std::vector<int32_t> make_qwen_partition_table_reference(const std::vector<int32_t> & expert_table,
                                                                int64_t                      token_count,
                                                                int64_t                      route_count,
                                                                int64_t                      expert_count) {
    std::vector<int32_t> partition_table(
        qwen_partition_table_size(token_count, route_count, expert_count) / sizeof(int32_t), -1);
    int32_t partition_count = 0;
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        const int32_t expert_assignment_count = expert_table[static_cast<size_t>(expert)];
        REQUIRE(expert_assignment_count >= 0);
        const int32_t expert_partition_count = (expert_assignment_count + 31) / 32;
        for (int32_t partition = 0; partition < expert_partition_count; ++partition) {
            int32_t row_count = expert_assignment_count - partition * 32;
            if (row_count > 32) {
                row_count = 32;
            }
            const int32_t descriptor = static_cast<int32_t>(expert) | (partition << 7) | ((row_count - 1) << 13);
            partition_table[static_cast<size_t>(1 + partition_count)] = descriptor;
            ++partition_count;
        }
    }
    partition_table[0] = partition_count;
    return partition_table;
}

static void require_qwen_expert_table_matches(const std::vector<int32_t> & actual,
                                              const std::vector<int32_t> & expected,
                                              int64_t                      token_count,
                                              int64_t                      expert_count) {
    REQUIRE(actual.size() == expected.size());
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        const size_t count_index = static_cast<size_t>(expert);
        REQUIRE(actual[count_index] == expected[count_index]);
        for (int32_t ordinal = 0; ordinal < expected[count_index]; ++ordinal) {
            const size_t assignment_index =
                static_cast<size_t>(expert_count + expert * token_count + static_cast<int64_t>(ordinal));
            REQUIRE(actual[assignment_index] == expected[assignment_index]);
        }
    }
}

static void require_qwen_partition_table_matches(const std::vector<int32_t> & actual,
                                                 const std::vector<int32_t> & expected) {
    REQUIRE(actual.size() == expected.size());
    REQUIRE(actual[0] == expected[0]);
    for (int32_t i = 0; i < actual[0]; ++i) {
        REQUIRE(actual[static_cast<size_t>(1 + i)] == expected[static_cast<size_t>(1 + i)]);
    }
}

static size_t qwen_routed_gate_up_f16_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenRouterRouteCount * kQwenMoeIntermediateSize) * sizeof(ggml_fp16_t);
}

static size_t qwen_routed_down_f16_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenRouterRouteCount * kQwenMoeHiddenSize) * sizeof(ggml_fp16_t);
}

static void set_qwen_flash_query_layout(ggml_tensor * tensor, int64_t head_count, int64_t head_size) {
    REQUIRE(tensor != nullptr);
    tensor->nb[0] = sizeof(float);
    tensor->nb[1] = static_cast<size_t>(head_count * head_size) * sizeof(float);
    tensor->nb[2] = static_cast<size_t>(head_size) * sizeof(float);
}

static void set_qwen_flash_key_value_layout(ggml_tensor * tensor, int64_t head_count, int64_t head_size) {
    REQUIRE(tensor != nullptr);
    tensor->nb[0] = sizeof(ggml_fp16_t);
    tensor->nb[1] = static_cast<size_t>(head_count * head_size) * sizeof(ggml_fp16_t);
    tensor->nb[2] = static_cast<size_t>(head_size) * sizeof(ggml_fp16_t);
}

static ggml_tensor * build_qwen_flash_attention_graph(ggml_context * ctx,
                                                      int64_t        query_token_count,
                                                      int64_t        key_value_token_count,
                                                      int64_t        query_head_count,
                                                      int64_t        key_value_head_count,
                                                      ggml_type      query_type     = GGML_TYPE_F32,
                                                      ggml_type      key_value_type = GGML_TYPE_F16,
                                                      bool           include_mask   = true,
                                                      bool           include_sinks  = false,
                                                      int64_t        head_size      = kQwenFlashHeadSize,
                                                      float          scale          = 1.0f / std::sqrt(128.0f),
                                                      float          max_bias       = 0.0f,
                                                      float          logit_softcap  = 0.0f) {
    ggml_tensor * query = ggml_new_tensor_3d(ctx, query_type, head_size, query_token_count, query_head_count);
    ggml_tensor * key = ggml_new_tensor_3d(ctx, key_value_type, head_size, key_value_token_count, key_value_head_count);
    ggml_tensor * value =
        ggml_new_tensor_3d(ctx, key_value_type, head_size, key_value_token_count, key_value_head_count);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    if (query_type == GGML_TYPE_F32) {
        set_qwen_flash_query_layout(query, query_head_count, head_size);
    }
    if (key_value_type == GGML_TYPE_F16) {
        set_qwen_flash_key_value_layout(key, key_value_head_count, head_size);
        set_qwen_flash_key_value_layout(value, key_value_head_count, head_size);
    }
    ggml_tensor * mask = nullptr;
    if (include_mask) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_token_count, query_token_count);
        REQUIRE(mask != nullptr);
    }
    ggml_tensor * output = ggml_flash_attn_ext(ctx, query, key, value, mask, scale, max_bias, logit_softcap);
    REQUIRE(output != nullptr);
    if (include_sinks) {
        ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, query_head_count);
        REQUIRE(sinks != nullptr);
        ggml_flash_attn_ext_add_sinks(output, sinks);
    }
    return output;
}

static ggml_tensor * build_qwen_router_top8_graph(ggml_context *  ctx,
                                                  ggml_tensor *   logits,
                                                  ggml_tensor **  route_ids   = nullptr,
                                                  ggml_sort_order order       = GGML_SORT_ORDER_DESC,
                                                  int64_t         route_count = kQwenRouterRouteCount,
                                                  float           clamp_min   = 1.0e-7f) {
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    REQUIRE(probs != nullptr);
    ggml_tensor * probs_reshaped = ggml_reshape_3d(ctx, probs, 1, logits->ne[0], logits->ne[1]);
    REQUIRE(probs_reshaped != nullptr);
    ggml_tensor * argsort = ggml_argsort(ctx, probs, order);
    REQUIRE(argsort != nullptr);
    ggml_tensor * topk = ggml_view_2d(ctx, argsort, route_count, logits->ne[1], argsort->nb[1], 0);
    REQUIRE(topk != nullptr);
    if (route_ids != nullptr) {
        *route_ids = topk;
    }
    ggml_tensor * selected = ggml_get_rows(ctx, probs_reshaped, topk);
    REQUIRE(selected != nullptr);
    ggml_tensor * selected_reshaped = ggml_reshape_2d(ctx, selected, route_count, logits->ne[1]);
    REQUIRE(selected_reshaped != nullptr);
    ggml_tensor * sum = ggml_sum_rows(ctx, selected_reshaped);
    REQUIRE(sum != nullptr);
    ggml_tensor * clamped_sum = ggml_clamp(ctx, sum, clamp_min, std::numeric_limits<float>::infinity());
    REQUIRE(clamped_sum != nullptr);
    ggml_tensor * normalized = ggml_div(ctx, selected_reshaped, clamped_sum);
    REQUIRE(normalized != nullptr);
    ggml_tensor * output = ggml_reshape_3d(ctx, normalized, 1, route_count, logits->ne[1]);
    REQUIRE(output != nullptr);
    return output;
}

static std::vector<size_t> traversal_indices(const ggml::hrx::Graph & graph) {
    const ggml::hrx::GraphTraversalOrder order = ggml::hrx::GraphTraversalOrder::build(graph);
    std::vector<size_t>                  indices;
    indices.reserve(order.nodes().size());
    for (const ggml::hrx::GraphNode * node : order.nodes()) {
        size_t index = 0;
        REQUIRE(node != nullptr);
        REQUIRE(graph.index().node_index(node, index));
        indices.push_back(index);
    }
    return indices;
}

static size_t find_position(const std::vector<size_t> & indices, size_t node_index) {
    const std::vector<size_t>::const_iterator it = std::find(indices.begin(), indices.end(), node_index);
    REQUIRE(it != indices.end());
    return static_cast<size_t>(it - indices.begin());
}

static size_t producer_index_for_tensor(const ggml::hrx::Graph & graph, const ggml_tensor * tensor) {
    const ggml::hrx::Value * value = graph.values().find_tensor(tensor);
    REQUIRE(value != nullptr);
    const ggml::hrx::GraphNode * producer = graph.index().producer(value->id);
    REQUIRE(producer != nullptr);
    size_t index = 0;
    REQUIRE(graph.index().node_index(producer, index));
    return index;
}

static bool match_dispatch_at_index(const ggml::hrx::Graph &       graph,
                                    const ggml::hrx::CommandPlan & plan,
                                    const std::vector<bool> &      covered_nodes,
                                    size_t                         node_index,
                                    ggml::hrx::DispatchMatch &     match);

static void append_match_to_plan(ggml::hrx::CommandPlan &   plan,
                                 ggml::hrx::DispatchMatch & match,
                                 std::vector<bool> &        covered_nodes,
                                 ggml::hrx::Graph *         graph = nullptr);

static void run_status_checks() {
    ggml::hrx::Status status;
    REQUIRE(status.success());
    REQUIRE(status.errors().empty());

    status.log("first");
    REQUIRE(!status.success());
    REQUIRE(!status.errors().empty());
    REQUIRE(status.errors().size() == 1);
    REQUIRE(status.errors()[0] == "first");

    status.log("value %d", 7);
    REQUIRE(status.errors().size() == 2);
    REQUIRE(status.errors()[1] == "value 7");

    ggml::hrx::Status other;
    other.log("third");
    status.append(other);
    REQUIRE(status.errors().size() == 3);
    REQUIRE(status.errors()[2] == "third");
}

static void run_command_plan_metadata_checks() {
    const ggml::hrx::MoeRoutingResourceMetadata routing = {
        4,
        8,
        128,
        128,
    };
    const ggml::hrx::CommandPlanResourceMetadata metadata = ggml::hrx::make_command_plan_resource_metadata(routing);

    REQUIRE(metadata.kind == ggml::hrx::CommandPlanResourceMetadataKind::MoeRoutingResource);
    ggml::hrx::MoeRoutingResourceMetadata decoded;
    REQUIRE(metadata.read(decoded));
    REQUIRE(decoded.token_count == routing.token_count);
    REQUIRE(decoded.route_count == routing.route_count);
    REQUIRE(decoded.route_stride == routing.route_stride);
    REQUIRE(decoded.expert_count == routing.expert_count);

    const ggml::hrx::CommandPlanResourceMetadata empty;
    REQUIRE(!empty.read(decoded));

    ggml::hrx::CommandPlanMetadata metadata_plan;
    ggml::hrx::Status              status;
    REQUIRE(metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(2), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(2), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(metadata_plan.alternate_values().size() == 1);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F16, 16) != nullptr);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F32, 16) == nullptr);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F16, 32) == nullptr);
    REQUIRE(!metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(3), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(!status.success());

    ggml::hrx::CommandPlanMetadata               bundle_plan;
    ggml::hrx::Status                            bundle_status;
    const ggml::hrx::CommandPlanMoeRoutingBundle bundle = {
        ggml::hrx::ValueId(10),
        ggml::hrx::ValueId(11),
        ggml::hrx::ValueId(12),
        ggml::hrx::ValueId(13),
        128,
        64,
        4,
        8,
        128,
        128,
    };
    REQUIRE(bundle_plan.append_moe_routing_bundle(bundle, bundle_status));
    REQUIRE(bundle_plan.append_moe_routing_bundle(bundle, bundle_status));
    REQUIRE(bundle_plan.moe_routing_bundles().size() == 1);
    const ggml::hrx::CommandPlanMoeRoutingBundle * found_bundle =
        bundle_plan.find_moe_routing_bundle(ggml::hrx::ValueId(10));
    REQUIRE(found_bundle != nullptr);
    REQUIRE(found_bundle->route_weights == ggml::hrx::ValueId(11));
    REQUIRE(found_bundle->expert_table == ggml::hrx::ValueId(12));
    REQUIRE(found_bundle->partition_table == ggml::hrx::ValueId(13));
    ggml::hrx::CommandPlanMoeRoutingBundle conflicting_bundle = bundle;
    conflicting_bundle.route_weights                          = ggml::hrx::ValueId(14);
    REQUIRE(!bundle_plan.append_moe_routing_bundle(conflicting_bundle, bundle_status));
    REQUIRE(!bundle_status.success());
}

static bool has_dispatch_registration(const std::vector<ggml::hrx::DispatchRegistration> & registrations,
                                      const char *                                         name) {
    for (const ggml::hrx::DispatchRegistration & registration : registrations) {
        if (std::string(registration.name) == name) {
            return true;
        }
    }
    return false;
}

static bool has_dispatch_registration_kind(const std::vector<ggml::hrx::DispatchRegistration> & registrations,
                                           const char *                                         name,
                                           ggml::hrx::DispatchMatchKind                         kind) {
    for (const ggml::hrx::DispatchRegistration & registration : registrations) {
        if (std::string(registration.name) == name && registration.kind == kind) {
            return true;
        }
    }
    return false;
}

static void restore_environment_value(const char * name, bool had_value, const std::string & value) {
    if (had_value) {
        REQUIRE(setenv(name, value.c_str(), 1) == 0);
    } else {
        REQUIRE(unsetenv(name) == 0);
    }
}

static void add_binary_f32_exact_dispatch_params(ggml::hrx::Dispatch & dispatch, int64_t element_count) {
    dispatch.kernel.integer_parameters.emplace("element_count", element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.op", "0");
}

static bool match_test_single_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                       ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 1);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_test_fused_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                      ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 2);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_test_wrong_root_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                           ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 3);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static void run_dispatch_registry_checks() {
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);

    const ggml::hrx::DispatchRegistry & registry = test_dispatch_registry();
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx1100" }) != nullptr);
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx1151" }) != nullptr);
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx0000" }) == nullptr);

    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_ADD), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_SUB), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_DIV), "common.binary_f32"));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_swiglu.f32_f32_wmma", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "llm.attention_qkv_matmul_postprocess.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.f32_f32_wmma", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.f32_f32_decode", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_postops.f32_f32_wmma", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT), "qwen.matmul.q6k_q8_1_x4"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                      "llm.moe_router.projection_f32_four_row_wave32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM),
                                      "qwen.rmsnorm_f32_quantize_q8_1_x4"));
    REQUIRE(
        has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_binary_q8_1_x4"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_f32_f16_wmma"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_decode_split_next_q8"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RESHAPE),
                                      "qwen.attention_postprocess_f32_f16"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32_next"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_GET_ROWS), "common.gather_add_f32"));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_ROPE), "common.rope_set_rows.f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_ROPE), "common.rope.f32",
                                           ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_SET_ROWS), "common.set_rows"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_SOFT_MAX), "llm.moe_router.top8_f32"));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_swiglu.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_postops.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id.f32_f32_wmma", ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "llm.routed_ffn.gate_up_swiglu_q4k_f16_wmma"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "llm.routed_ffn.down_q4k_f16_wmma_grouped"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "llm.routed_ffn.down_q6k_f16_wmma_grouped"));
    REQUIRE(registry.single_op_registrations().size() >= 5);

    REQUIRE(setenv(kDisableQwenDispatchEnv, "0", 1) == 0);
    const ggml::hrx::DispatchRegistry & false_env_registry = test_dispatch_registry();
    REQUIRE(
        has_dispatch_registration(false_env_registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32"));

    REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
    const ggml::hrx::DispatchRegistry & generic_registry = test_dispatch_registry();
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_ADD), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM),
                                      "common.rmsnorm_binary_f32"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM),
                                      "common.rmsnorm_binary_q8_1_x4"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_f32"));
    REQUIRE(
        has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS), "common.gather_add_f32"));
    REQUIRE(
        has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS),
                                      "common.get_rows.f32_next"));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_ROPE),
                                           "common.rope_set_rows.f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_ROPE), "common.rope.f32",
                                           ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_SET_ROWS), "common.set_rows"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM),
                                       "qwen.rmsnorm_f32_quantize_q8_1_x4"));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_swiglu.f32_f32_wmma", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "llm.attention_qkv_matmul_postprocess.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.f32_f32_wmma", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.f32_f32_decode", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_postops.f32_f32_wmma", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                       "qwen.matmul.q6k_q8_1_x4"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_f32_f16_wmma"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_decode_split_next_q8"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RESHAPE),
                                       "qwen.attention_postprocess_f32_f16"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS),
                                       "qwen.preamble.token_embedding_q4k"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_SOFT_MAX),
                                       "llm.moe_router.top8_f32"));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_swiglu.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_postops.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id.f32_f32_wmma", ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                       "llm.routed_ffn.gate_up_swiglu_q4k_f16_wmma"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                       "llm.routed_ffn.down_q4k_f16_wmma_grouped"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                       "llm.routed_ffn.down_q6k_f16_wmma_grouped"));
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);

    ggml::hrx::DispatchRegistryBuilder builder;
    builder.add({
        "test.single_add",
        GGML_OP_ADD,
        ggml::hrx::DispatchMatchKind::SingleOp,
        1000,
        ggml::hrx::DispatchSource::Common,
        match_test_single_dispatch,
    });
    builder.add({
        "test.fused_add",
        GGML_OP_ADD,
        ggml::hrx::DispatchMatchKind::Fused,
        0,
        ggml::hrx::DispatchSource::Common,
        match_test_fused_dispatch,
    });
    builder.add({
        "test.wrong_root",
        GGML_OP_MUL_MAT,
        ggml::hrx::DispatchMatchKind::Fused,
        2000,
        ggml::hrx::DispatchSource::Common,
        match_test_wrong_root_dispatch,
    });
    const ggml::hrx::DispatchRegistry ordering_registry = builder.build();

    ggml::hrx::Graph graph;
    graph.add_node(GGML_OP_ADD, ggml::hrx::ValueId(0), {});
    REQUIRE(graph.build_index().success());

    const std::vector<bool>               covered_nodes(graph.nodes().size(), false);
    const ggml::hrx::CommandPlan          plan;
    const ggml::hrx::DispatchMatchContext context = {
        graph, &graph.nodes().front(),
        0,     covered_nodes,
        plan,  ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size())),
    };
    ggml::hrx::DispatchMatch match;
    REQUIRE(ordering_registry.match(context, match));
    REQUIRE(match.dispatches.size() == 1);
    REQUIRE(match.dispatches.front().kernel.integer_parameters.at("route") == 2);
}

static void run_graph_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    const ggml::hrx::GraphNode * node = &imported.graph.nodes().front();
    REQUIRE(node->op == GGML_OP_ADD);
    REQUIRE(node->inputs.size() == 2);
    REQUIRE(node->inputs[0] != node->inputs[1]);
    REQUIRE(ggml::hrx::DispatchScheduler::supports_node(imported.graph, node, test_dispatch_target()));

    const ggml::hrx::Value * a_value   = imported.graph.values().find_tensor(a);
    const ggml::hrx::Value * b_value   = imported.graph.values().find_tensor(b);
    const ggml::hrx::Value * out_value = imported.graph.values().find_tensor(out);
    REQUIRE(a_value != nullptr);
    REQUIRE(b_value != nullptr);
    REQUIRE(out_value != nullptr);
    REQUIRE(a_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(b_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(out_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(!a_value->buffer.has_value());
    REQUIRE(!b_value->buffer.has_value());
    REQUIRE(!out_value->buffer.has_value());

    const std::vector<ggml::hrx::ValueId> external_ids = imported.graph.values().external_value_ids();
    REQUIRE(external_ids.size() == 3);
    REQUIRE(contains_value_id(external_ids, a_value->id));
    REQUIRE(contains_value_id(external_ids, b_value->id));
    REQUIRE(contains_value_id(external_ids, out_value->id));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    REQUIRE(scheduler.plan().dispatches.front().bindings.size() == 3);
    REQUIRE(scheduler.plan().dispatches.front().bindings[0].length == a_value->byte_count);

    ggml::hrx::CommandProgramBindings missing_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!missing_bindings.valid());

    REQUIRE(imported.graph.values().bind_buffer(a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count }));
    ggml::hrx::CommandProgramBindings partial_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!partial_bindings.valid());

    REQUIRE(imported.graph.values().bind_buffer(b_value->id, { dummy_hrx_buffer(0x1800), 0, b_value->byte_count }));
    ggml::hrx::CommandProgramBindings missing_output_binding =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!missing_output_binding.valid());

    REQUIRE(imported.graph.values().bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, 0 }));
    ggml::hrx::CommandProgramBindings empty_runtime_binding =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!empty_runtime_binding.valid());

    REQUIRE(imported.graph.values().bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    ggml::hrx::CommandProgramBindings runtime_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(runtime_bindings.valid());
    REQUIRE(runtime_bindings.bindings().size() == 3);
    const ggml::hrx::CommandProgramBinding * a_binding = runtime_bindings.find(a_value->id);
    REQUIRE(a_binding != nullptr);
    REQUIRE(a_binding->buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(a_binding->offset == 0);
    REQUIRE(a_binding->length == a_value->byte_count);
    const ggml::hrx::CommandProgramBinding * b_binding = runtime_bindings.find(b_value->id);
    REQUIRE(b_binding != nullptr);
    REQUIRE(b_binding->buffer == dummy_hrx_buffer(0x1800));
    REQUIRE(b_binding->offset == 0);
    REQUIRE(b_binding->length == b_value->byte_count);
    REQUIRE(runtime_bindings.find(ggml::hrx::ValueId(123456)) == nullptr);

    const ggml::hrx::CommandProgramBindingsFingerprint runtime_fingerprint =
        ggml::hrx::command_program_bindings_fingerprint(runtime_bindings);
    REQUIRE(!runtime_fingerprint.value.empty());

    ggml::hrx::ValueMap changed_identity = imported.graph.values();
    REQUIRE(changed_identity.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 1, 0, a_value->byte_count }));
    const ggml::hrx::CommandProgramBindings changed_identity_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_identity);
    REQUIRE(changed_identity_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_identity_bindings).value !=
            runtime_fingerprint.value);

    ggml::hrx::ValueMap changed_generation = imported.graph.values();
    REQUIRE(changed_generation.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 0, 1, a_value->byte_count }));
    const ggml::hrx::CommandProgramBindings changed_generation_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_generation);
    REQUIRE(changed_generation_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_generation_bindings).value !=
            runtime_fingerprint.value);

    ggml::hrx::ValueMap changed_capacity = imported.graph.values();
    REQUIRE(changed_capacity.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 0, 0, a_value->byte_count + 256 }));
    const ggml::hrx::CommandProgramBindings changed_capacity_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_capacity);
    REQUIRE(changed_capacity_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_capacity_bindings).value !=
            runtime_fingerprint.value);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    const ggml::hrx::Command & command = commands.commands.front();
    REQUIRE(command.ordinal == 0);
    REQUIRE(command.kind == ggml::hrx::CommandKind::Kernel);
    REQUIRE(command.kernel.kernel_id != ggml::hrx::kUncatalogedKernelId);
    REQUIRE(command.bindings.size() == 3);
    REQUIRE(command.bindings[0].name == "lhs");
    REQUIRE(command.bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[0].access == ggml::hrx::ResourceAccess::Read);
    REQUIRE(command.bindings[1].name == "rhs");
    REQUIRE(command.bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[1].access == ggml::hrx::ResourceAccess::Read);
    REQUIRE(command.bindings[2].name == "output");
    REQUIRE(command.bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[2].access == ggml::hrx::ResourceAccess::ReadWrite);
    REQUIRE(command_program_verifies(commands));

    const ggml::hrx::PreparedCommand default_prepared_command;
    REQUIRE(default_prepared_command.kind == ggml::hrx::CommandKind::Invalid);
    REQUIRE(ggml::hrx::command_kind_name(ggml::hrx::CommandKind::Invalid) == "Invalid");
    REQUIRE(ggml::hrx::command_kind_name(ggml::hrx::CommandKind::Kernel) == "Kernel");
    REQUIRE(ggml::hrx::command_kind_name(static_cast<ggml::hrx::CommandKind>(255)) == "Unknown(255)");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::GraphValue) == "GraphValue");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::Transient) == "Transient");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::ProgramConstant) ==
            "ProgramConstant");
    REQUIRE(ggml::hrx::command_binding_origin_name(static_cast<ggml::hrx::CommandBindingOrigin>(255)) ==
            "Unknown(255)");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::Read) == "Read");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::Write) == "Write");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::ReadWrite) == "ReadWrite");
    REQUIRE(ggml::hrx::resource_access_name(static_cast<ggml::hrx::ResourceAccess>(255)) == "Unknown(255)");

    const std::string binding_text = ggml::hrx::format_command_binding(command.bindings[0]);
    REQUIRE(string_contains(binding_text, "binding lhs"));
    REQUIRE(string_contains(binding_text, "value="));
    REQUIRE(string_contains(binding_text, "origin=GraphValue"));
    REQUIRE(string_contains(binding_text, "access=Read"));
    REQUIRE(std::string(ggml::hrx::hrx_graph_replay_event_name(ggml::hrx::HrxGraphReplayEvent::Disabled)) ==
            "disabled");
    REQUIRE(std::string(ggml::hrx::hrx_graph_replay_event_name(ggml::hrx::HrxGraphReplayEvent::Hit)) == "hit");
    REQUIRE(string_contains(binding_text, "range=[0, "));
    REQUIRE(string_contains(binding_text, std::to_string(a_value->byte_count).c_str()));

    const std::string command_text = ggml::hrx::format_command(command);
    REQUIRE(string_contains(command_text, "command 0"));
    REQUIRE(string_contains(command_text, "kind=Kernel"));
    REQUIRE(string_contains(command_text, "kernel_id="));
    REQUIRE(string_contains(command_text, "bindings=3"));
    REQUIRE(string_contains(command_text, "deps=0"));

    const std::string program_text = ggml::hrx::format_command_program(commands);
    REQUIRE(string_contains(program_text, "command_program commands=1"));
    REQUIRE(string_contains(program_text, "command 0"));
    REQUIRE(string_contains(program_text, "binding lhs"));
    REQUIRE(string_contains(program_text, "binding rhs"));
    REQUIRE(string_contains(program_text, "binding output"));

    ggml::hrx::ResolvedCommandProgram resolved =
        ggml::hrx::resolve_command_program_bindings(commands, runtime_bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.size() == 1);
    REQUIRE(resolved.commands.front().ordinal == command.ordinal);
    REQUIRE(resolved.commands.front().kind == command.kind);
    REQUIRE(resolved.commands.front().kernel.kernel_id == command.kernel.kernel_id);
    REQUIRE(resolved.commands.front().bindings.size() == 3);
    REQUIRE(resolved.commands.front().bindings[0].binding.name == "lhs");
    REQUIRE(resolved.commands.front().bindings[0].ref.buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(resolved.commands.front().bindings[0].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[0].ref.length == a_value->byte_count);
    REQUIRE(resolved.commands.front().bindings[1].binding.name == "rhs");
    REQUIRE(resolved.commands.front().bindings[1].ref.buffer == dummy_hrx_buffer(0x1800));
    REQUIRE(resolved.commands.front().bindings[1].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[1].ref.length == b_value->byte_count);
    REQUIRE(resolved.commands.front().bindings[2].binding.name == "output");
    REQUIRE(resolved.commands.front().bindings[2].ref.buffer == dummy_hrx_buffer(0x2000));
    REQUIRE(resolved.commands.front().bindings[2].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[2].ref.length == out_value->byte_count);

    ggml::hrx::CommandProgram offset_command           = copy_command_program_shape(commands);
    offset_command.commands.front().bindings[0].offset = 4;
    offset_command.commands.front().bindings[0].length = 8;
    ggml::hrx::ValueMap offset_values                  = imported.graph.values();
    REQUIRE(offset_values.bind_buffer(a_value->id, { dummy_hrx_buffer(0x3000), 16, a_value->byte_count }));
    REQUIRE(offset_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x4000), 32, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings offset_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(offset_values);
    REQUIRE(offset_bindings.valid());
    resolved = ggml::hrx::resolve_command_program_bindings(offset_command, offset_bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.front().bindings[0].ref.buffer == dummy_hrx_buffer(0x3000));
    REQUIRE(resolved.commands.front().bindings[0].ref.offset == 20);
    REQUIRE(resolved.commands.front().bindings[0].ref.length == 8);

    std::vector<uint8_t>          host_storage(a_value->byte_count + 64);
    ggml::hrx::ValueBufferBinding host_value_binding;
    host_value_binding.host_data    = host_storage.data();
    host_value_binding.offset       = 16;
    host_value_binding.length       = a_value->byte_count;
    host_value_binding.identity     = 42;
    host_value_binding.generation   = 1;
    host_value_binding.capacity     = host_storage.size();
    host_value_binding.weight       = true;
    ggml::hrx::ValueMap host_values = imported.graph.values();
    REQUIRE(host_values.bind_buffer(a_value->id, host_value_binding));
    REQUIRE(host_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x5000), 0, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings host_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(host_values);
    REQUIRE(host_bindings.valid());
    const ggml::hrx::CommandProgramBinding * host_binding = host_bindings.find(a_value->id);
    REQUIRE(host_binding != nullptr);
    REQUIRE(host_binding->buffer == nullptr);
    REQUIRE(host_binding->host_data == host_storage.data());
    REQUIRE(host_binding->offset == 16);
    REQUIRE(host_binding->length == a_value->byte_count);
    REQUIRE(host_binding->weight);
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(host_bindings).value != runtime_fingerprint.value);

    resolved = ggml::hrx::resolve_command_program_bindings(commands, missing_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "is not bound"));
    REQUIRE(status_contains(resolved.status, "binding output"));
    REQUIRE(status_contains(resolved.status, "value="));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, partial_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "is not bound"));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, empty_runtime_binding);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "empty binding"));

    ggml::hrx::ValueMap null_values = imported.graph.values();
    REQUIRE(null_values.bind_buffer(a_value->id, { nullptr, 0, a_value->byte_count }));
    REQUIRE(null_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings null_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(null_values);
    REQUIRE(!null_bindings.valid());
    resolved = ggml::hrx::resolve_command_program_bindings(commands, null_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "null buffer"));
    REQUIRE(status_contains(resolved.status, "binding lhs"));

    ggml::hrx::CommandProgram empty_resolve_binding           = copy_command_program_shape(commands);
    empty_resolve_binding.commands.front().bindings[0].length = 0;
    resolved = ggml::hrx::resolve_command_program_bindings(empty_resolve_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "empty range"));
    REQUIRE(status_contains(resolved.status, "range=[0, 0)"));

    ggml::hrx::CommandProgram out_of_range_binding           = copy_command_program_shape(commands);
    out_of_range_binding.commands.front().bindings[0].offset = a_value->byte_count;
    out_of_range_binding.commands.front().bindings[0].length = 4;
    resolved = ggml::hrx::resolve_command_program_bindings(out_of_range_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "outside runtime binding length"));
    REQUIRE(status_contains(resolved.status, "binding lhs"));

    ggml::hrx::CommandProgram unsupported_origin           = copy_command_program_shape(commands);
    unsupported_origin.commands.front().bindings[0].origin = static_cast<ggml::hrx::CommandBindingOrigin>(255);
    resolved = ggml::hrx::resolve_command_program_bindings(unsupported_origin, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "unsupported binding origin"));
    REQUIRE(status_contains(resolved.status, "origin=Unknown(255)"));
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(unsupported_origin, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "unsupported binding origin"));
    REQUIRE(status_contains(verification.status, "origin=Unknown(255)"));

    ggml::hrx::CommandProgram invalid_kernel         = copy_command_program_shape(commands);
    invalid_kernel.commands.front().kernel.kernel_id = ggml::hrx::kUncatalogedKernelId;
    verification = ggml::hrx::verify_command_program(invalid_kernel, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "command 0"));
    REQUIRE(status_contains(verification.status, "kernel_id="));

    ggml::hrx::CommandProgram empty_bindings = copy_command_program_shape(commands);
    empty_bindings.commands.front().bindings.clear();
    verification = ggml::hrx::verify_command_program(empty_bindings, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "bindings=0"));

    ggml::hrx::CommandProgram empty_binding           = copy_command_program_shape(commands);
    empty_binding.commands.front().bindings[0].length = 0;
    verification = ggml::hrx::verify_command_program(empty_binding, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "binding lhs"));
    REQUIRE(status_contains(verification.status, "range=[0, 0)"));

    ggml::hrx::CommandProgram invalid_value          = copy_command_program_shape(commands);
    invalid_value.commands.front().bindings[0].value = ggml::hrx::ValueId(-1);
    verification = ggml::hrx::verify_command_program(invalid_value, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "value=-1"));

    ggml::hrx::CommandProgram forward_dependency = copy_command_program_shape(commands);
    forward_dependency.commands.front().dependencies.push_back(0);
    verification =
        ggml::hrx::verify_command_program(forward_dependency, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "forward dependency 0"));

    ggml::hrx::CommandProgram wrong_binding_name         = copy_command_program_shape(commands);
    wrong_binding_name.commands.front().bindings[0].name = "wrong";
    verification =
        ggml::hrx::verify_command_program(wrong_binding_name, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "binding wrong"));

    ggml::hrx::CommandProgram wrong_binding_access           = copy_command_program_shape(commands);
    wrong_binding_access.commands.front().bindings[0].access = ggml::hrx::ResourceAccess::Write;
    verification =
        ggml::hrx::verify_command_program(wrong_binding_access, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "access=Write"));

    const ggml::hrx::KernelCorpus &                 corpus          = ggml::hrx::get_qwen_kernel_corpus();
    const ggml::hrx::CommandProgramExecutionContext prepare_context = {
        nullptr, nullptr, "gfx1151", &corpus, nullptr, nullptr, nullptr, nullptr,
    };

    ggml::hrx::PreparedCommandProgram prepared =
        ggml::hrx::prepare_command_program(prepare_context, invalid_kernel, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "kernel_id="));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, missing_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "is not bound"));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "missing HRX device"));

    const ggml::hrx::CommandProgramExecutionContext missing_kernel_cache_context = {
        reinterpret_cast<hrx_device_t>(uintptr_t(1)), nullptr, "gfx1151", &corpus, nullptr, nullptr, nullptr, nullptr,
    };
    prepared = ggml::hrx::prepare_command_program(missing_kernel_cache_context, commands, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "missing HRX kernel executable cache"));

    ggml_free(ctx);
}

static ggml::hrx::Dispatch schedule_single_dispatch_for_tensor(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    return scheduler.plan().dispatches.front();
}

static bool can_schedule_tensor(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

static void require_compile_param(const ggml::hrx::Dispatch & dispatch, const char * name, const char * value) {
    REQUIRE(dispatch.kernel.compile_parameters.at(name) == value);
}

static void run_binary_f32_broadcast_dispatch_checks() {
    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * lhs    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
        ggml_tensor * rhs    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
        ggml_tensor * output = ggml_add(ctx, lhs, rhs);
        REQUIRE(lhs != nullptr);
        REQUIRE(rhs != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 48);
        REQUIRE(dispatch.kernel.integer_parameters.count("ne0") == 0);
        REQUIRE(dispatch.kernel.integer_parameters.count("src0_element_count") == 0);
        require_compile_param(dispatch, "ggml.binary_f32.op", "0");
        REQUIRE(dispatch.kernel.compile_parameters.count("ggml.binary_bc_f32.src0_broadcast_dim0") == 0);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_tensor * output = ggml_mul(ctx, input, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_bc_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 48);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne0") == 16);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne1") == 3);
        REQUIRE(dispatch.kernel.integer_parameters.at("src0_element_count") == 48);
        REQUIRE(dispatch.kernel.integer_parameters.at("src1_element_count") == 16);
        require_compile_param(dispatch, "ggml.binary_bc_f32.op", "2");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src0_broadcast_dim0", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src0_broadcast_dim1", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim0", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim1", "1");

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * numerator   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor * denominator = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 4);
        ggml_tensor * output      = ggml_div(ctx, numerator, denominator);
        REQUIRE(numerator != nullptr);
        REQUIRE(denominator != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_bc_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 32);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne0") == 8);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne1") == 4);
        REQUIRE(dispatch.kernel.integer_parameters.at("src0_element_count") == 32);
        REQUIRE(dispatch.kernel.integer_parameters.at("src1_element_count") == 4);
        require_compile_param(dispatch, "ggml.binary_bc_f32.op", "3");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src0_broadcast_dim0", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim0", "1");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim1", "0");

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 6, 7);
        ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 6, 7);
        ggml_tensor * output = ggml_mul(ctx, input, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_bc_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 210);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne0") == 5);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne1") == 6);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne2") == 7);
        REQUIRE(dispatch.kernel.integer_parameters.at("src0_element_count") == 210);
        REQUIRE(dispatch.kernel.integer_parameters.at("src1_element_count") == 42);
        require_compile_param(dispatch, "ggml.binary_bc_f32.op", "2");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim0", "1");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim1", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim2", "0");

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
        ggml_tensor * output = ggml_mul(ctx, input, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(output != nullptr);
        REQUIRE(!can_schedule_tensor(ctx, output));

        ggml_free(ctx);
    }
}

static void run_graph_snapshot_diagnostics_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    const std::string snapshot_json = ggml::hrx::serialize_graph_snapshot_json(imported.graph, "gfx1151", 42);
    REQUIRE(string_contains(snapshot_json, "ggml-hrx-graph-snapshot-v1"));
    REQUIRE(string_contains(snapshot_json, "ADD"));

    ggml::hrx::GraphSnapshotLoadResult loaded = ggml::hrx::load_graph_snapshot_json(snapshot_json);
    REQUIRE(loaded.valid());
    REQUIRE(loaded.uid == 42);
    REQUIRE(loaded.target == "gfx1151");
    REQUIRE(loaded.graph.nodes().size() == imported.graph.nodes().size());
    REQUIRE(loaded.graph.values().size() == imported.graph.values().size());

    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(loaded.graph, { loaded.target }, &diagnostics));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        loaded.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), loaded.target);
    REQUIRE(commands.valid());
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_unmatched_graph_diagnostics_checks() {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, 512);
    ggml_tensor * rows    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 2);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    REQUIRE(cache != nullptr);
    REQUIRE(rows != nullptr);
    REQUIRE(indices != nullptr);
    ggml_tensor * out = ggml_set_rows(ctx, cache, rows, indices);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes().front().op == GGML_OP_SET_ROWS);

    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(!scheduler.schedule_graph(imported.graph, test_dispatch_target(), &diagnostics));
    REQUIRE(diagnostics.unsupported_node != nullptr);
    REQUIRE(diagnostics.unsupported_node_index == 0);
    REQUIRE(diagnostics.unsupported_node->op == GGML_OP_SET_ROWS);
    REQUIRE(!diagnostics.match.attempts.empty());
    REQUIRE(status_contains(scheduler.plan().status, "unsupported HRX node 0: SET_ROWS"));

    const std::string diagnostics_text =
        ggml::hrx::format_schedule_diagnostics_text(imported.graph, scheduler.plan(), diagnostics);
    REQUIRE(string_contains(diagnostics_text, "unsupported_node=0:SET_ROWS"));
    REQUIRE(string_contains(diagnostics_text, "matcher_attempts="));
    const std::string diagnostics_json =
        ggml::hrx::serialize_schedule_diagnostics_json(imported.graph, scheduler.plan(), diagnostics);
    REQUIRE(string_contains(diagnostics_json, "SET_ROWS"));
    REQUIRE(string_contains(diagnostics_json, "matcher_attempts"));

    ggml_free(ctx);
}

static void run_completion_counter_plan_checks() {
    ggml::hrx::Graph         graph;
    ggml::hrx::CommandPlan   plan;
    const ggml::hrx::ValueId first_counter(1000);
    const ggml::hrx::ValueId second_counter(1001);
    plan.completion_counter_requests.push_back({ first_counter, "first_completion_counter", 1 });
    plan.completion_counter_requests.push_back({ second_counter, "second_completion_counters", 2 });

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.empty());
    REQUIRE(commands.constant_initializations.empty());
    REQUIRE(commands.completion_counters.count == 3);
    REQUIRE(commands.completion_counters.arena_offset == 0);
    REQUIRE(commands.completion_counters.byte_count == 24);
    REQUIRE(commands.transients.allocations.size() == 2);
    REQUIRE(commands.transients.arena_size == 256);
    REQUIRE(command_program_verifies(commands));

    const ggml::hrx::TransientAllocation * first_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, first_counter);
    const ggml::hrx::TransientAllocation * second_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, second_counter);
    REQUIRE(first_allocation != nullptr);
    REQUIRE(second_allocation != nullptr);
    REQUIRE(first_allocation->size == sizeof(int32_t));
    REQUIRE(first_allocation->alignment == 16);
    REQUIRE(first_allocation->arena_offset == commands.completion_counters.arena_offset);
    REQUIRE(second_allocation->size == 2 * sizeof(int32_t));
    REQUIRE(second_allocation->alignment == 16);
    REQUIRE(second_allocation->arena_offset == 16);

    ggml::hrx::CommandPlan bound_plan;
    bound_plan.completion_counter_requests.push_back({ first_counter, "first_bound_completion_counter", 1 });
    bound_plan.completion_counter_requests.push_back({ second_counter, "second_bound_completion_counter", 1 });
    ggml::hrx::Dispatch first_dispatch;
    first_dispatch.kernel =
        ggml::hrx::make_kernel_specialization(ggml::hrx::kernel_catalog_ref("loom_libs", "ggml_binary_f32"));
    add_binary_f32_exact_dispatch_params(first_dispatch, 1);
    first_dispatch.bindings.push_back({ first_counter, 0, sizeof(int32_t) });
    first_dispatch.bindings.push_back({ second_counter, 0, sizeof(int32_t) });
    first_dispatch.bindings.push_back({ first_counter, 0, sizeof(int32_t) });
    bound_plan.dispatches.push_back(std::move(first_dispatch));
    ggml::hrx::Dispatch second_dispatch;
    second_dispatch.kernel =
        ggml::hrx::make_kernel_specialization(ggml::hrx::kernel_catalog_ref("loom_libs", "ggml_binary_f32"));
    add_binary_f32_exact_dispatch_params(second_dispatch, 1);
    second_dispatch.bindings.push_back({ second_counter, 0, sizeof(int32_t) });
    second_dispatch.bindings.push_back({ first_counter, 0, sizeof(int32_t) });
    second_dispatch.bindings.push_back({ second_counter, 0, sizeof(int32_t) });
    bound_plan.dispatches.push_back(std::move(second_dispatch));

    const ggml::hrx::CommandProgram bound_commands =
        ggml::hrx::build_command_program(graph, bound_plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(bound_commands.valid());
    REQUIRE(bound_commands.commands.size() == 2);
    REQUIRE(bound_commands.completion_counters.count == 2);
    REQUIRE(bound_commands.completion_counters.arena_offset == 0);
    REQUIRE(bound_commands.completion_counters.byte_count == 20);
    REQUIRE(command_program_verifies(bound_commands));

    const ggml::hrx::TransientArenaAllocationRef transient_arena = {
        dummy_hrx_buffer(0x8000),
        bound_commands.transients.arena_size,
        7,
    };
    ggml::hrx::PreparedCommandProgram prepared_shape;
    for (const ggml::hrx::Command & prepared_source : bound_commands.commands) {
        ggml::hrx::PreparedCommand prepared_command;
        prepared_command.ordinal               = prepared_source.ordinal;
        prepared_command.kind                  = prepared_source.kind;
        prepared_command.kernel.specialization = prepared_source.kernel;
        for (const ggml::hrx::CommandBinding & binding : prepared_source.bindings) {
            prepared_command.kernel.bindings.push_back({
                binding, { dummy_hrx_buffer(0x4000), 123, binding.length }
            });
        }
        prepared_shape.commands.push_back(std::move(prepared_command));
    }
    prepared_shape.bound_transient_arena_allocation_id = 1;

    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(bound_commands, transient_arena, prepared_shape));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[0].ref.offset == 0);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[1].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[1].ref.offset == 16);
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.offset == 16);
    REQUIRE(prepared_shape.commands[1].kernel.bindings[1].ref.offset == 0);
}

static void run_graph_index_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * rms = ggml_rms_norm(ctx, input, 0.000001f);
    REQUIRE(rms != nullptr);
    ggml_tensor * out = ggml_mul(ctx, rms, weight);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.has_index());
    REQUIRE(imported.graph.nodes().size() == 2);
    const ggml::hrx::GraphNode * rms_node = &imported.graph.nodes()[0];
    const ggml::hrx::GraphNode * mul_node = &imported.graph.nodes()[1];
    REQUIRE(rms_node->op == GGML_OP_RMS_NORM);
    REQUIRE(mul_node->op == GGML_OP_MUL);
    const ggml::hrx::RmsNormParams * rms_params = ggml::hrx::op_params_as<ggml::hrx::RmsNormParams>(rms_node->params);
    REQUIRE(rms_params != nullptr);
    REQUIRE(rms_params->eps == 0.000001f);
    REQUIRE(imported.graph.index().producer(rms_node->output) == rms_node);
    REQUIRE(imported.graph.index().producer(mul_node->output) == mul_node);
    REQUIRE(imported.graph.index().has_single_consumer(rms_node->output));
    const std::vector<const ggml::hrx::GraphNode *> & consumers = imported.graph.index().consumers(rms_node->output);
    REQUIRE(consumers.size() == 1);
    REQUIRE(consumers.front() == mul_node);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    ggml_free(ctx);

    params.mem_size = 256 * 1024;
    params.no_alloc = true;
    ctx             = ggml_init(params);
    REQUIRE(ctx != nullptr);

    input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
    REQUIRE(input != nullptr);
    rms = ggml_rms_norm(ctx, input, 0.00001f);
    REQUIRE(rms != nullptr);

    graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, rms);

    imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);

    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 4);
    require_compile_parameter(dispatch, "ggml.rmsnorm_f32.hidden_size", "256");
    REQUIRE(dispatch.kernel.compile_parameters.count("ggml.rmsnorm_f32.rms_epsilon") == 1);
    REQUIRE(dispatch.bindings.size() == 2);

    ggml_free(ctx);
}

static void run_rope_set_rows_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t head_size   = 8;
    constexpr int64_t head_count  = 2;
    constexpr int64_t token_count = 3;
    constexpr int64_t cache_rows  = 8;
    constexpr int64_t hidden_size = head_size * head_count;

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * output = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f,
                                             0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);

        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        require_compile_parameter(dispatch, "ggml.rope_f32.head_size", "8");
        require_compile_parameter(dispatch, "ggml.rope_f32.head_count", "2");
        require_compile_parameter(dispatch, "ggml.rope_f32.token_capacity", "3");
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "2");
        REQUIRE(dispatch.bindings.size() == 5);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * output = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f,
                                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);

        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        require_compile_parameter(dispatch, "ggml.rope_f32.head_size", "8");
        require_compile_parameter(dispatch, "ggml.rope_f32.head_count", "2");
        require_compile_parameter(dispatch, "ggml.rope_f32.token_capacity", "3");
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "0");
        REQUIRE(dispatch.bindings.size() == 5);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * output = ggml_rope(ctx, input, pos, head_size, GGML_ROPE_TYPE_NEOX);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        REQUIRE(scheduler.plan().transients.size() == 2);
        REQUIRE(scheduler.plan().constant_initializations.size() == 2);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "2");
    }

    {
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * rows    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(cache != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_SET_ROWS);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);

        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_set_rows");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_rows);
        REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == hidden_size);
        require_compile_parameter(dispatch, "ggml.set_rows.input_format", "32");
        require_compile_parameter(dispatch, "ggml.set_rows.output_format", "16");
        REQUIRE(dispatch.bindings.size() == 3);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * rows    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, token_count);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(cache != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_set_rows");
        require_compile_parameter(dispatch, "ggml.set_rows.input_format", "16");
        require_compile_parameter(dispatch, "ggml.set_rows.output_format", "16");
    }

    {
        ggml_tensor * input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * rope    = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f,
                                              0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * rows    = ggml_reshape_2d(ctx, rope, hidden_size, token_count);
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(cache != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 3);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_RESHAPE);
        REQUIRE(imported.graph.nodes()[2].op == GGML_OP_SET_ROWS);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_set_rows_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_rows);
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.output_format", "16");
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.mode", "2");
        REQUIRE(dispatch.bindings.size() == 6);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * rope  = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f,
                                            0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * rows  = ggml_reshape_2d(ctx, rope, hidden_size, token_count);
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(cache != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 3);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_RESHAPE);
        REQUIRE(imported.graph.nodes()[2].op == GGML_OP_SET_ROWS);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_set_rows_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_rows);
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.output_format", "16");
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.mode", "0");
        REQUIRE(dispatch.bindings.size() == 6);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    ggml_free(ctx);
}

static void run_graph_traversal_checks() {
    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * out0 = ggml_add(ctx, a, b);
        ggml_tensor * out1 = ggml_add(ctx, c, d);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(c != nullptr);
        REQUIRE(d != nullptr);
        REQUIRE(out0 != nullptr);
        REQUIRE(out1 != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, out0);
        ggml_build_forward_expand(graph, out1);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].output == imported.graph.values().find_tensor(out0)->id);
        REQUIRE(imported.graph.nodes()[1].output == imported.graph.values().find_tensor(out1)->id);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 0);
        REQUIRE(order[1] == 1);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * add_out = ggml_add(ctx, a, b);
        ggml_tensor * weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
        ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        ggml_tensor * matmul  = ggml_mul_mat(ctx, weight, input);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(add_out != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(matmul != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, add_out);
        ggml_build_forward_expand(graph, matmul);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL_MAT);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 1);
        REQUIRE(order[1] == 0);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * rms    = ggml_rms_norm(ctx, input, 0.000001f);
        ggml_tensor * out    = ggml_mul(ctx, rms, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(rms != nullptr);
        REQUIRE(out != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, out);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_RMS_NORM);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 0);
        REQUIRE(order[1] == 1);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * c     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * left  = ggml_add(ctx, a, b);
        ggml_tensor * right = ggml_mul(ctx, a, c);
        ggml_tensor * join  = ggml_add(ctx, left, right);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(c != nullptr);
        REQUIRE(left != nullptr);
        REQUIRE(right != nullptr);
        REQUIRE(join != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, join);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 3);

        const size_t left_index  = producer_index_for_tensor(imported.graph, left);
        const size_t right_index = producer_index_for_tensor(imported.graph, right);
        const size_t join_index  = producer_index_for_tensor(imported.graph, join);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 3);
        REQUIRE(find_position(order, join_index) > find_position(order, left_index));
        REQUIRE(find_position(order, join_index) > find_position(order, right_index));

        ggml_free(ctx);
    }
}

static void schedule_single_matmul_command(ggml_context * ctx,
                                           ggml_tensor *  output,
                                           const char *   expected_kernel_name,
                                           int64_t        expected_token_count,
                                           int64_t        expected_input_size,
                                           int64_t        expected_output_size) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == 3);
    if (string_contains(kernel_name, "_decode")) {
        REQUIRE(dispatch.kernel.integer_parameters.at("input_size") == expected_input_size);
        REQUIRE(dispatch.kernel.integer_parameters.at("output_size") == expected_output_size);
        require_compile_parameter(dispatch, "ggml.mul_mat_f32_f32_decode.token_capacity",
                                  std::to_string(expected_token_count));
        require_compile_parameter(dispatch, "ggml.mul_mat_f32_f32_decode.output_capacity",
                                  std::to_string(expected_output_size));
        const ggml::hrx::Value * weight = imported.graph.values().find(dispatch.bindings[1].value);
        REQUIRE(weight != nullptr);
        require_compile_parameter(dispatch, "ggml.mul_mat_f32_f32_decode.weight_format",
                                  std::to_string(matmul_weight_format_config(weight->type)));
    } else if (string_contains(kernel_name, "ggml_mul_mat")) {
        require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
        require_compile_parameter(dispatch, "ggml.mul_mat.input_size", std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "ggml.mul_mat.output_size", std::to_string(expected_output_size));
        require_compile_parameter(dispatch, "ggml.mul_mat.output_accumulation", "0");
        require_compile_parameter(dispatch, "ggml.mul_mat.output_unary_op",
                                  std::to_string(ggml::hrx::unary_kind_config_value(ggml::hrx::UnaryKind::Identity)));
        const ggml::hrx::Value * weight = imported.graph.values().find(dispatch.bindings[1].value);
        REQUIRE(weight != nullptr);
        require_compile_parameter(dispatch, "ggml.mul_mat.weight_format",
                                  std::to_string(matmul_weight_format_config(weight->type)));
    } else if (string_contains(kernel_name, "dense_linear")) {
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(expected_token_count));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.input_size",
                                  std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.output_size",
                                  std::to_string(expected_output_size));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "0");
    } else {
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(expected_token_count));
        require_compile_parameter(dispatch, "qwen3_moe.model.hidden_size", std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", std::to_string(expected_output_size));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 3);
    REQUIRE(commands.commands.front().bindings[0].name == "input");
    REQUIRE(commands.commands.front().bindings[1].name == "weight");
    REQUIRE(commands.commands.front().bindings[2].name == "output");
}

static void schedule_fused_matmul_unary_command(ggml_context *       ctx,
                                                ggml_tensor *        output,
                                                const char *         expected_kernel_name,
                                                ggml::hrx::UnaryKind expected_unary_op,
                                                int64_t              expected_token_count,
                                                int64_t              expected_input_size,
                                                int64_t              expected_output_size) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == 3);
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "ggml.mul_mat.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "ggml.mul_mat.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "ggml.mul_mat.output_accumulation", "0");
    require_compile_parameter(dispatch, "ggml.mul_mat.output_unary_op",
                              std::to_string(ggml::hrx::unary_kind_config_value(expected_unary_op)));
    const ggml::hrx::Value * weight = imported.graph.values().find(dispatch.bindings[1].value);
    REQUIRE(weight != nullptr);
    require_compile_parameter(dispatch, "ggml.mul_mat.weight_format",
                              std::to_string(matmul_weight_format_config(weight->type)));
    REQUIRE(dispatch.bindings[2].value == imported.graph.nodes()[1].output);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 3);
    REQUIRE(commands.commands.front().bindings[0].name == "input");
    REQUIRE(commands.commands.front().bindings[1].name == "weight");
    REQUIRE(commands.commands.front().bindings[2].name == "output");
}

static void schedule_fused_matmul_swiglu_command(ggml_context * ctx,
                                                 ggml_tensor *  output,
                                                 ggml_type      expected_gate_type,
                                                 ggml_type      expected_up_type,
                                                 int64_t        expected_token_count,
                                                 int64_t        expected_input_size,
                                                 int64_t        expected_output_size) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL_MAT);
    REQUIRE(imported.graph.nodes()[2].op == GGML_OP_GLU);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "loom_libs:ggml_mul_mat_swiglu_f32_f32_wmma");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == 4);
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.gate_weight_format",
                              std::to_string(matmul_weight_format_config(expected_gate_type)));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.up_weight_format",
                              std::to_string(matmul_weight_format_config(expected_up_type)));
    REQUIRE(dispatch.bindings[3].value == imported.graph.nodes()[2].output);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 4);
    REQUIRE(commands.commands.front().bindings[0].name == "input");
    REQUIRE(commands.commands.front().bindings[1].name == "gate_weight");
    REQUIRE(commands.commands.front().bindings[2].name == "up_weight");
    REQUIRE(commands.commands.front().bindings[3].name == "output");
}

static void schedule_fused_matmul_postops_command(ggml_context *                  ctx,
                                                  ggml_tensor *                   output,
                                                  const char *                    expected_kernel_name,
                                                  ggml_type                       expected_weight_type,
                                                  int64_t                         expected_token_count,
                                                  int64_t                         expected_input_size,
                                                  int64_t                         expected_output_size,
                                                  size_t                          expected_node_count,
                                                  const std::vector<ggml_op> &    expected_ops,
                                                  const std::vector<const char *> expected_binding_names,
                                                  bool                            expected_rmsnorm) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == expected_node_count);
    REQUIRE(expected_ops.size() == expected_node_count);
    for (size_t i = 0; i < expected_ops.size(); ++i) {
        REQUIRE(imported.graph.nodes()[i].op == expected_ops[i]);
    }

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == (expected_rmsnorm ? 2 : 1));
    REQUIRE(scheduler.plan().completion_counter_requests.empty());

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == expected_binding_names.size());
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "ggml.mul_mat_postops.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_postops.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_postops.weight_format",
                              std::to_string(matmul_weight_format_config(expected_weight_type)));
    REQUIRE(dispatch.kernel.compile_parameters.count("ggml.mul_mat_postops.rms_epsilon") == 0);
    if (expected_rmsnorm) {
        const ggml::hrx::Dispatch & rmsnorm_dispatch = scheduler.plan().dispatches[1];
        REQUIRE(kernel_name_for_id(rmsnorm_dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_binary_f32");
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == (expected_rmsnorm ? 2 : 1));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == expected_binding_names.size());
    for (size_t i = 0; i < expected_binding_names.size(); ++i) {
        REQUIRE(commands.commands.front().bindings[i].name == expected_binding_names[i]);
    }
}

static void schedule_fused_llama_attention_matmul_command(ggml_context *                  ctx,
                                                          ggml_tensor *                   output,
                                                          const char *                    expected_kernel_name,
                                                          ggml_type                       expected_weight_type,
                                                          int64_t                         expected_token_count,
                                                          int64_t                         expected_input_size,
                                                          int64_t                         expected_output_size,
                                                          int64_t                         expected_head_size,
                                                          int64_t                         expected_head_count,
                                                          int64_t                         expected_cache_row_count,
                                                          int64_t                         expected_cache_format,
                                                          size_t                          expected_node_count,
                                                          const std::vector<ggml_op> &    expected_ops,
                                                          const std::vector<const char *> expected_binding_names) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == expected_node_count);
    REQUIRE(expected_ops.size() == expected_node_count);
    for (size_t i = 0; i < expected_ops.size(); ++i) {
        REQUIRE(imported.graph.nodes()[i].op == expected_ops[i]);
    }

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == expected_binding_names.size());
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "llm.attention_qkv.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "llm.attention_qkv.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "llm.attention_qkv.weight_format",
                              std::to_string(matmul_weight_format_config(expected_weight_type)));
    if (expected_head_size > 0) {
        require_compile_parameter(dispatch, "llm.attention_qkv.head_size", std::to_string(expected_head_size));
        require_compile_parameter(dispatch, "llm.attention_qkv.head_count", std::to_string(expected_head_count));
    }
    if (expected_cache_row_count > 0) {
        require_compile_parameter(dispatch, "llm.attention_qkv.cache_row_count",
                                  std::to_string(expected_cache_row_count));
        require_compile_parameter(dispatch, "llm.attention_qkv.cache_output_format",
                                  std::to_string(expected_cache_format));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == expected_binding_names.size());
    for (size_t i = 0; i < expected_binding_names.size(); ++i) {
        REQUIRE(commands.commands.front().bindings[i].name == expected_binding_names[i]);
    }
}

static void require_matmul_swiglu_falls_back(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() > 1);
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) != "loom_libs:ggml_mul_mat_swiglu_f32_f32_wmma");
    }
}

static void require_matmul_root_matches_identity_single(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(!imported.graph.nodes().empty());
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);

    const std::vector<bool>               covered_nodes(imported.graph.nodes().size(), false);
    const ggml::hrx::CommandPlan          plan;
    const ggml::hrx::DispatchMatchContext context = {
        imported.graph,
        &imported.graph.nodes()[0],
        0,
        covered_nodes,
        plan,
        ggml::hrx::ValueId(static_cast<int32_t>(imported.graph.values().size())),
    };
    ggml::hrx::DispatchMatch match;
    REQUIRE(test_dispatch_registry().match(context, match));
    REQUIRE(match.covered_nodes.size() == 1);
    REQUIRE(match.covered_nodes.front() == 0);
    REQUIRE(match.dispatches.size() == 1);
    require_compile_parameter(match.dispatches.front(), "ggml.mul_mat.output_unary_op",
                              std::to_string(ggml::hrx::unary_kind_config_value(ggml::hrx::UnaryKind::Identity)));
}

static bool matmul_graph_is_supported(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

static void schedule_qwen_terminal_q6k_q8_command(int64_t token_count) {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, token_count);
    ggml_tensor * norm_weight  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    ggml_tensor * vocab_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 151936);
    REQUIRE(input != nullptr);
    REQUIRE(norm_weight != nullptr);
    REQUIRE(vocab_weight != nullptr);
    ggml_tensor * rms        = ggml_rms_norm(ctx, input, 0.000001f);
    ggml_tensor * normalized = ggml_mul(ctx, rms, norm_weight);
    ggml_tensor * logits     = ggml_mul_mat(ctx, vocab_weight, normalized);
    REQUIRE(rms != nullptr);
    REQUIRE(normalized != nullptr);
    REQUIRE(logits != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, logits);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);
    REQUIRE(scheduler.plan().transients.size() == 1);

    const ggml::hrx::Dispatch & rms_dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(rms_dispatch.kernel.kernel_id) == "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4");
    REQUIRE(rms_dispatch.kernel.integer_parameters.at("token_count") == token_count);
    require_compile_parameter(rms_dispatch, "qwen3_moe.model.hidden_size", "2048");
    require_compile_parameter(rms_dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));
    REQUIRE(rms_dispatch.bindings.size() == 4);

    const ggml::hrx::CommandPlanTransient & q8_transient = scheduler.plan().transients.front();
    REQUIRE(q8_transient.size == qwen_q8_1_x4_size(token_count, 2048));
    REQUIRE(rms_dispatch.bindings[3].value == q8_transient.value);
    REQUIRE(rms_dispatch.bindings[3].length == q8_transient.size);

    const ggml::hrx::Dispatch & vocab_dispatch = scheduler.plan().dispatches[1];
    REQUIRE(kernel_name_for_id(vocab_dispatch.kernel.kernel_id) == "qwen3_moe:ggml_linear_q6k_q8_1_x4");
    REQUIRE(vocab_dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(vocab_dispatch.kernel.integer_parameters.at("input_size") == 2048);
    REQUIRE(vocab_dispatch.kernel.integer_parameters.at("output_size") == 151936);
    require_compile_parameter(vocab_dispatch, "ggml.linear_q6k_q8_1_x4.token_capacity", std::to_string(token_count));
    require_compile_parameter(vocab_dispatch, "ggml.linear_q6k_q8_1_x4.output_capacity", "151936");
    REQUIRE(vocab_dispatch.bindings.size() == 3);
    REQUIRE(vocab_dispatch.bindings[0].value == q8_transient.value);
    REQUIRE(vocab_dispatch.bindings[0].length == q8_transient.size);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings.size() == 4);
    REQUIRE(commands.commands[0].bindings[3].name == "q8_output");
    REQUIRE(commands.commands[0].bindings[3].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings.size() == 3);
    REQUIRE(commands.commands[1].bindings[0].name == "q8_input");
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[0].value == q8_transient.value);

    ggml_free(ctx);
}

static void schedule_get_rows_q8_1_alternate_command() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * embedding_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 151936);
    ggml_tensor * token_ids        = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * vocab_weight     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 151936);
    REQUIRE(embedding_weight != nullptr);
    REQUIRE(token_ids != nullptr);
    REQUIRE(vocab_weight != nullptr);

    ggml_tensor * embedding = ggml_get_rows(ctx, embedding_weight, token_ids);
    ggml_tensor * logits    = ggml_mul_mat(ctx, vocab_weight, embedding);
    REQUIRE(embedding != nullptr);
    REQUIRE(logits != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, logits);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_GET_ROWS);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL_MAT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);
    REQUIRE(scheduler.plan().transients.size() == 1);

    const size_t q8_byte_count = qwen_q8_1_x4_size(1, 2048);

    const ggml::hrx::Dispatch & get_rows_dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(get_rows_dispatch.kernel.kernel_id) == "loom_libs:ggml_get_rows_f32_next");
    REQUIRE(get_rows_dispatch.kernel.integer_parameters.at("token_count") == 1);
    REQUIRE(get_rows_dispatch.kernel.integer_parameters.at("row_count") == 151936);
    REQUIRE(get_rows_dispatch.kernel.integer_parameters.at("hidden_size") == 2048);
    require_compile_parameter(get_rows_dispatch, "ggml.get_rows_f32.weight_format", "4");
    require_compile_parameter(get_rows_dispatch, "ggml.get_rows_f32.next_format", "81");
    REQUIRE(get_rows_dispatch.bindings.size() == 4);

    const ggml::hrx::CommandPlanTransient & q8_transient = scheduler.plan().transients.front();
    REQUIRE(q8_transient.size == q8_byte_count);
    REQUIRE(get_rows_dispatch.bindings[3].value == q8_transient.value);
    REQUIRE(get_rows_dispatch.bindings[3].length == q8_transient.size);

    const ggml::hrx::CommandPlanAlternateValue * alternate =
        scheduler.plan().metadata.find_alternate_value(imported.graph.nodes()[0].output, GGML_TYPE_Q8_1, q8_byte_count);
    REQUIRE(alternate != nullptr);
    REQUIRE(alternate->alternate_value == q8_transient.value);

    const ggml::hrx::Dispatch & vocab_dispatch = scheduler.plan().dispatches[1];
    REQUIRE(kernel_name_for_id(vocab_dispatch.kernel.kernel_id) == "qwen3_moe:ggml_linear_q6k_q8_1_x4");
    REQUIRE(vocab_dispatch.bindings.size() == 3);
    REQUIRE(vocab_dispatch.bindings[0].value == alternate->alternate_value);
    REQUIRE(vocab_dispatch.bindings[0].length == alternate->byte_count);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings.size() == 4);
    REQUIRE(commands.commands[0].bindings[3].name == "next_output");
    REQUIRE(commands.commands[0].bindings[3].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings.size() == 3);
    REQUIRE(commands.commands[1].bindings[0].name == "q8_input");

    ggml_free(ctx);
}

static bool graph_is_supported(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

struct ManualQwenRouterTop8Graph {
    ggml::hrx::Graph   graph;
    ggml::hrx::ValueId route_ids;
};

static ggml::hrx::ValueId add_manual_graph_tensor(ggml::hrx::Graph &   graph,
                                                  ggml_tensor *        tensor,
                                                  ggml::hrx::ValueKind kind) {
    REQUIRE(tensor != nullptr);
    return graph.values().get_or_add_tensor_value(tensor, kind);
}

static ManualQwenRouterTop8Graph build_manual_qwen_router_top8_graph(ggml_context * ctx,
                                                                     int64_t        expert_count,
                                                                     int64_t        route_count,
                                                                     int64_t        token_count) {
    ManualQwenRouterTop8Graph manual;
    ggml::hrx::Graph &        graph = manual.graph;

    const ggml::hrx::ValueId logits = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, expert_count, token_count), ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId probs = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, expert_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId probs_reshaped = add_manual_graph_tensor(
        graph, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, expert_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId argsort = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, expert_count, token_count), ggml::hrx::ValueKind::Transient);
    manual.route_ids = add_manual_graph_tensor(graph, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, token_count),
                                               ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId selected = add_manual_graph_tensor(
        graph, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, route_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId weights_flat = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, route_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId sum = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId clamped = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId normalized = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, route_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId route_weights = add_manual_graph_tensor(
        graph, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, route_count, token_count), ggml::hrx::ValueKind::External);

    ggml::hrx::GraphNode & softmax = graph.add_node(GGML_OP_SOFT_MAX, probs, { logits });
    softmax.params                 = ggml::hrx::SoftMaxParams{ 1.0f, 0.0f };
    graph.add_node(GGML_OP_RESHAPE, probs_reshaped, { probs });
    ggml::hrx::GraphNode & argsort_node = graph.add_node(GGML_OP_ARGSORT, argsort, { probs });
    argsort_node.params                 = ggml::hrx::ArgsortParams{ GGML_SORT_ORDER_DESC };
    graph.add_node(GGML_OP_VIEW, manual.route_ids, { argsort });
    graph.add_node(GGML_OP_GET_ROWS, selected, { probs_reshaped, manual.route_ids });
    graph.add_node(GGML_OP_RESHAPE, weights_flat, { selected });
    graph.add_node(GGML_OP_SUM_ROWS, sum, { weights_flat });
    ggml::hrx::GraphNode & clamp = graph.add_node(GGML_OP_CLAMP, clamped, { sum });
    clamp.params                 = ggml::hrx::ClampParams{ 0.00006103515625f, std::numeric_limits<float>::infinity() };
    graph.add_node(GGML_OP_DIV, normalized, { weights_flat, clamped });
    graph.add_node(GGML_OP_RESHAPE, route_weights, { normalized });
    REQUIRE(graph.build_index().success());
    return manual;
}

static ggml::hrx::Graph build_manual_token_embedding_graph(ggml_tensor * weight,
                                                           ggml_tensor * token_ids,
                                                           ggml_tensor * output) {
    ggml::hrx::Graph   graph;
    ggml::hrx::ValueId weight_value = graph.values().get_or_add_tensor_value(weight, ggml::hrx::ValueKind::External);
    ggml::hrx::ValueId token_ids_value =
        graph.values().get_or_add_tensor_value(token_ids, ggml::hrx::ValueKind::External);
    ggml::hrx::ValueId output_value = graph.values().get_or_add_tensor_value(output, ggml::hrx::ValueKind::External);
    graph.add_node(GGML_OP_GET_ROWS, output_value, { weight_value, token_ids_value });
    REQUIRE(graph.build_index().success());
    return graph;
}

static bool manual_token_embedding_graph_is_supported(ggml_context * ctx,
                                                      ggml_type      weight_type,
                                                      ggml_type      token_ids_type,
                                                      ggml_type      output_type,
                                                      int64_t        hidden_size,
                                                      int64_t        vocabulary_count,
                                                      int64_t        token_count,
                                                      int64_t        output_hidden_size = -1,
                                                      int64_t        output_token_count = -1) {
    if (output_hidden_size < 0) {
        output_hidden_size = hidden_size;
    }
    if (output_token_count < 0) {
        output_token_count = token_count;
    }
    ggml_tensor * weight    = ggml_new_tensor_2d(ctx, weight_type, hidden_size, vocabulary_count);
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, token_ids_type, token_count);
    ggml_tensor * output    = ggml_new_tensor_2d(ctx, output_type, output_hidden_size, output_token_count);
    REQUIRE(weight != nullptr);
    REQUIRE(token_ids != nullptr);
    REQUIRE(output != nullptr);

    ggml::hrx::Graph graph = build_manual_token_embedding_graph(weight, token_ids, output);
    return ggml::hrx::DispatchScheduler::can_schedule_graph(graph, test_dispatch_target());
}

static void schedule_qwen_token_embedding_command(ggml_context * ctx,
                                                  ggml_tensor *  output,
                                                  int64_t        expected_token_count,
                                                  int64_t        expected_vocabulary_count) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_GET_ROWS);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "loom_libs:ggml_get_rows_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("row_count") == expected_vocabulary_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == 2048);
    require_compile_parameter(dispatch, "ggml.get_rows_f32.weight_format", "4");
    REQUIRE(dispatch.bindings.size() == 3);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 3);
    REQUIRE(commands.commands.front().bindings[0].name == "token_ids");
    REQUIRE(commands.commands.front().bindings[1].name == "weight");
    REQUIRE(commands.commands.front().bindings[2].name == "output");
}

static void run_qwen_token_embedding_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 151936);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_qwen_token_embedding_command(ctx, output, 1, 151936);
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 151936);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 13);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_qwen_token_embedding_command(ctx, output, 13, 151936);
    }

    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q6_K, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q8_0, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q8_1, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        !manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q4_K, GGML_TYPE_I64, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        !manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q4_K, GGML_TYPE_I32, GGML_TYPE_F16, 2048, 151936, 1));
    REQUIRE(
        !manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F32, 128, 151936, 1));
    REQUIRE(
        !manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F32, 33024, 151936, 1));
    REQUIRE(!manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q4_K, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936,
                                                       1, 2048, 2));

    ggml_free(ctx);
}

static ggml::hrx::Graph build_manual_gather_add_graph(ggml_context * ctx,
                                                      int64_t        hidden_size,
                                                      int64_t        source_token_count,
                                                      int64_t        output_token_count,
                                                      bool           shared_row_ids,
                                                      int64_t        second_source_hidden_size = -1) {
    if (second_source_hidden_size < 0) {
        second_source_hidden_size = hidden_size;
    }

    ggml::hrx::Graph graph;

    ggml_tensor * attention = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor * residual  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, second_source_hidden_size, source_token_count);
    ggml_tensor * row_ids0  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor * row_ids1  = shared_row_ids ? row_ids0 : ggml_new_tensor_1d(ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor * selected0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, output_token_count);
    ggml_tensor * selected1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, output_token_count);
    ggml_tensor * output    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, output_token_count);
    REQUIRE(attention != nullptr);
    REQUIRE(residual != nullptr);
    REQUIRE(row_ids0 != nullptr);
    REQUIRE(row_ids1 != nullptr);
    REQUIRE(selected0 != nullptr);
    REQUIRE(selected1 != nullptr);
    REQUIRE(output != nullptr);

    const ggml::hrx::ValueId attention_value =
        graph.values().get_or_add_tensor_value(attention, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId residual_value =
        graph.values().get_or_add_tensor_value(residual, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId row_ids0_value =
        graph.values().get_or_add_tensor_value(row_ids0, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId row_ids1_value =
        graph.values().get_or_add_tensor_value(row_ids1, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId selected0_value =
        graph.values().get_or_add_tensor_value(selected0, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId selected1_value =
        graph.values().get_or_add_tensor_value(selected1, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId output_value =
        graph.values().get_or_add_tensor_value(output, ggml::hrx::ValueKind::External);

    graph.add_node(GGML_OP_GET_ROWS, selected0_value, { attention_value, row_ids0_value });
    graph.add_node(GGML_OP_GET_ROWS, selected1_value, { residual_value, row_ids1_value });
    graph.add_node(GGML_OP_ADD, output_value, { selected0_value, selected1_value });
    REQUIRE(graph.build_index().success());
    return graph;
}

static bool manual_gather_add_graph_is_supported(ggml_context * ctx,
                                                 int64_t        hidden_size,
                                                 int64_t        source_token_count,
                                                 int64_t        output_token_count,
                                                 bool           shared_row_ids,
                                                 int64_t        second_source_hidden_size = -1) {
    ggml::hrx::Graph graph = build_manual_gather_add_graph(ctx, hidden_size, source_token_count, output_token_count,
                                                           shared_row_ids, second_source_hidden_size);
    return ggml::hrx::DispatchScheduler::can_schedule_graph(graph, test_dispatch_target());
}

static bool partial_gather_add_graph_is_supported(ggml_context * ctx) {
    ggml::hrx::Graph graph;

    ggml_tensor * attention  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 13);
    ggml_tensor * row_ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * selected   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * add_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * add_output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    REQUIRE(attention != nullptr);
    REQUIRE(row_ids != nullptr);
    REQUIRE(selected != nullptr);
    REQUIRE(add_input != nullptr);
    REQUIRE(add_output != nullptr);

    const ggml::hrx::ValueId attention_value =
        graph.values().get_or_add_tensor_value(attention, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId row_ids_value =
        graph.values().get_or_add_tensor_value(row_ids, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId selected_value =
        graph.values().get_or_add_tensor_value(selected, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId add_input_value =
        graph.values().get_or_add_tensor_value(add_input, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId add_output_value =
        graph.values().get_or_add_tensor_value(add_output, ggml::hrx::ValueKind::External);

    graph.add_node(GGML_OP_GET_ROWS, selected_value, { attention_value, row_ids_value });
    graph.add_node(GGML_OP_ADD, add_output_value, { selected_value, add_input_value });
    REQUIRE(graph.build_index().success());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(graph, test_dispatch_target());
}

static void schedule_gather_add_command(ggml::hrx::Graph & graph,
                                        int64_t            expected_hidden_size,
                                        int64_t            expected_source_token_count,
                                        int64_t            expected_output_token_count) {
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx:ggml_gather_add_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == expected_hidden_size);
    REQUIRE(dispatch.kernel.integer_parameters.at("source_token_count") == expected_source_token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("output_token_count") == expected_output_token_count);
    REQUIRE(dispatch.bindings.size() == 4);

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 4);
    REQUIRE(commands.commands.front().bindings[0].name == "attention");
    REQUIRE(commands.commands.front().bindings[1].name == "residual");
    REQUIRE(commands.commands.front().bindings[2].name == "output_ids");
    REQUIRE(commands.commands.front().bindings[3].name == "output");
}

static void run_gather_add_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml::hrx::Graph graph = build_manual_gather_add_graph(ctx, 2048, 13, 1, true);
        schedule_gather_add_command(graph, 2048, 13, 1);
    }
    {
        ggml::hrx::Graph graph = build_manual_gather_add_graph(ctx, 2048, 128, 8, true);
        schedule_gather_add_command(graph, 2048, 128, 8);
    }

    REQUIRE(!manual_gather_add_graph_is_supported(ctx, 2048, 13, 1, false));
    REQUIRE(!manual_gather_add_graph_is_supported(ctx, 2048, 13, 1, true, 1024));
    REQUIRE(!manual_gather_add_graph_is_supported(ctx, 96, 13, 1, true));
    REQUIRE(!partial_gather_add_graph_is_supported(ctx));

    ggml_free(ctx);
}

static void schedule_flash_attention_command(ggml_context * ctx,
                                             ggml_tensor *  output,
                                             const char *   expected_kernel_name,
                                             const char *   config_prefix,
                                             int64_t        query_token_count,
                                             int64_t        key_value_token_count,
                                             int64_t        query_head_count,
                                             int64_t        key_value_head_count,
                                             int64_t        head_size = kQwenFlashHeadSize) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_FLASH_ATTN_EXT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("query_token_count") == query_token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == key_value_token_count);
    REQUIRE(dispatch.bindings.size() == 5);
    require_compile_parameter(dispatch, (std::string(config_prefix) + "query_head_count").c_str(),
                              std::to_string(query_head_count));
    require_compile_parameter(dispatch, (std::string(config_prefix) + "key_value_head_count").c_str(),
                              std::to_string(key_value_head_count));
    if (std::strcmp(config_prefix, "ggml.flash_attention.") == 0) {
        require_compile_parameter(dispatch, "ggml.flash_attention.head_size", std::to_string(head_size));
    } else {
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(query_token_count));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 5);
    REQUIRE(commands.commands.front().bindings[0].name == "query");
    REQUIRE(commands.commands.front().bindings[1].name == "key");
    REQUIRE(commands.commands.front().bindings[2].name == "value");
    REQUIRE(commands.commands.front().bindings[3].name == "mask");
    REQUIRE(commands.commands.front().bindings[4].name == "output");
}

static void schedule_qwen_flash_attention_command(ggml_context * ctx, ggml_tensor * output) {
    schedule_flash_attention_command(ctx, output, "loom_libs:ggml_flash_attention_f32_f16_wmma",
                                     "ggml.flash_attention.", 4, 8, 4, 2);
}

static void schedule_common_flash_attention_command(ggml_context * ctx,
                                                    ggml_tensor *  output,
                                                    int64_t        query_token_count,
                                                    int64_t        key_value_token_count,
                                                    int64_t        query_head_count,
                                                    int64_t        key_value_head_count,
                                                    int64_t        head_size = kQwenFlashHeadSize) {
    schedule_flash_attention_command(ctx, output, "loom_libs:ggml_flash_attention_f32_f16_wmma",
                                     "ggml.flash_attention.", query_token_count, key_value_token_count,
                                     query_head_count, key_value_head_count, head_size);
}

static void schedule_common_flash_attention_decode_split_command(ggml_context * ctx,
                                                                 ggml_tensor *  output,
                                                                 int64_t        query_token_count,
                                                                 int64_t        key_value_token_count,
                                                                 int64_t        query_head_count,
                                                                 int64_t        key_value_head_count) {
    const int64_t key_value_capacity = ((key_value_token_count + 63) / 64) * 64;
    ggml_cgraph * graph              = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_FLASH_ATTN_EXT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == static_cast<size_t>(query_token_count));

    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        const std::string kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
        REQUIRE(kernel_name == "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8");
        REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == key_value_token_count);
        REQUIRE(dispatch.bindings.size() == 10);
        require_compile_parameter(dispatch, "ggml.flash_attention.query_head_count", std::to_string(query_head_count));
        require_compile_parameter(dispatch, "ggml.flash_attention.key_value_head_count",
                                  std::to_string(key_value_head_count));
        require_compile_parameter(dispatch, "ggml.flash_attention.decode.key_value_token_capacity",
                                  std::to_string(key_value_capacity));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == static_cast<size_t>(query_token_count));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 10);
    REQUIRE(commands.commands.front().bindings[0].name == "query");
    REQUIRE(commands.commands.front().bindings[1].name == "key");
    REQUIRE(commands.commands.front().bindings[2].name == "value");
    REQUIRE(commands.commands.front().bindings[3].name == "mask");
    REQUIRE(commands.commands.front().bindings[4].name == "partial_max");
    REQUIRE(commands.commands.front().bindings[5].name == "partial_sum");
    REQUIRE(commands.commands.front().bindings[6].name == "partial_output");
    REQUIRE(commands.commands.front().bindings[7].name == "completion_counter");
    REQUIRE(commands.commands.front().bindings[8].name == "output");
    REQUIRE(commands.commands.front().bindings[9].name == "next_q8_output");
}

static void run_qwen_flash_attention_dispatch_checks() {
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2);
        schedule_qwen_flash_attention_command(ctx, output);
    }
    {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 16, 16, 4, 2);
        schedule_common_flash_attention_command(ctx, output, 16, 16, 4, 2);
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 1, 8, 4, 2);
        schedule_common_flash_attention_decode_split_command(ctx, output, 1, 8, 4, 2);
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F32);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, false);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output =
            build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true, true);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true,
                                                                false, kQwenFlashHeadSize, 1.0f);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output =
            build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true, false,
                                             kQwenFlashHeadSize, 1.0f / std::sqrt(128.0f), 1.0f);
        REQUIRE(!graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);
}

struct QwenAttentionPostprocessTensors {
    ggml_tensor * query_raw           = nullptr;
    ggml_tensor * key_raw             = nullptr;
    ggml_tensor * value_raw           = nullptr;
    ggml_tensor * query_reshape       = nullptr;
    ggml_tensor * key_reshape         = nullptr;
    ggml_tensor * value_reshape       = nullptr;
    ggml_tensor * query_output        = nullptr;
    ggml_tensor * key_cache           = nullptr;
    ggml_tensor * value_cache         = nullptr;
    ggml_tensor * key_output          = nullptr;
    ggml_tensor * value_output        = nullptr;
    ggml_tensor * positions           = nullptr;
    ggml_tensor * key_cache_indices   = nullptr;
    ggml_tensor * value_cache_indices = nullptr;
    ggml_tensor * mask                = nullptr;
    ggml_tensor * flash_output        = nullptr;
};

static QwenAttentionPostprocessTensors build_qwen_attention_postprocess_graph(ggml_context * ctx,
                                                                              int64_t        token_count,
                                                                              int64_t        query_head_count,
                                                                              int64_t        key_value_head_count,
                                                                              int64_t        cache_row_count,
                                                                              float          rms_epsilon = 0.000001f,
                                                                              bool include_inverse_frequencies = true) {
    QwenAttentionPostprocessTensors tensors;
    const int64_t                   query_size     = query_head_count * kQwenFlashHeadSize;
    const int64_t                   key_value_size = key_value_head_count * kQwenFlashHeadSize;

    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, token_count);
    ggml_tensor * query_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenMoeHiddenSize, query_size);
    ggml_tensor * key_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenMoeHiddenSize, key_value_size);
    ggml_tensor * value_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, kQwenMoeHiddenSize, key_value_size);
    REQUIRE(input != nullptr);
    REQUIRE(query_weight != nullptr);
    REQUIRE(key_weight != nullptr);
    REQUIRE(value_weight != nullptr);

    tensors.query_raw = ggml_mul_mat(ctx, query_weight, input);
    tensors.key_raw   = ggml_mul_mat(ctx, key_weight, input);
    tensors.value_raw = ggml_mul_mat(ctx, value_weight, input);
    REQUIRE(tensors.query_raw != nullptr);
    REQUIRE(tensors.key_raw != nullptr);
    REQUIRE(tensors.value_raw != nullptr);

    tensors.query_reshape = ggml_reshape_3d(ctx, tensors.query_raw, kQwenFlashHeadSize, query_head_count, token_count);
    tensors.key_reshape = ggml_reshape_3d(ctx, tensors.key_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    tensors.value_reshape =
        ggml_reshape_3d(ctx, tensors.value_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    REQUIRE(tensors.query_reshape != nullptr);
    REQUIRE(tensors.key_reshape != nullptr);
    REQUIRE(tensors.value_reshape != nullptr);

    ggml_tensor * query_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    ggml_tensor * key_norm_weight   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    tensors.positions               = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    ggml_tensor * inverse_frequencies =
        include_inverse_frequencies ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize / 2) : nullptr;
    REQUIRE(query_norm_weight != nullptr);
    REQUIRE(key_norm_weight != nullptr);
    REQUIRE(tensors.positions != nullptr);
    REQUIRE(include_inverse_frequencies == (inverse_frequencies != nullptr));

    ggml_tensor * query_norm = ggml_rms_norm(ctx, tensors.query_reshape, rms_epsilon);
    ggml_tensor * query_mul  = ggml_mul(ctx, query_norm, query_norm_weight);
    REQUIRE(query_norm != nullptr);
    REQUIRE(query_mul != nullptr);
    tensors.query_output = include_inverse_frequencies ?
                               ggml_rope_ext(ctx, query_mul, tensors.positions, inverse_frequencies, kQwenFlashHeadSize,
                                             GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) :
                               ggml_rope(ctx, query_mul, tensors.positions, kQwenFlashHeadSize, GGML_ROPE_TYPE_NEOX);
    REQUIRE(tensors.query_output != nullptr);

    ggml_tensor * key_norm = ggml_rms_norm(ctx, tensors.key_reshape, rms_epsilon);
    ggml_tensor * key_mul  = ggml_mul(ctx, key_norm, key_norm_weight);
    REQUIRE(key_norm != nullptr);
    REQUIRE(key_mul != nullptr);
    ggml_tensor * key_rope = include_inverse_frequencies ?
                                 ggml_rope_ext(ctx, key_mul, tensors.positions, inverse_frequencies, kQwenFlashHeadSize,
                                               GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) :
                                 ggml_rope(ctx, key_mul, tensors.positions, kQwenFlashHeadSize, GGML_ROPE_TYPE_NEOX);
    REQUIRE(key_rope != nullptr);

    ggml_tensor * key_cache_rows =
        ggml_reshape_2d(ctx, key_rope, kQwenFlashHeadSize * key_value_head_count, token_count);
    ggml_tensor * value_cache_rows =
        ggml_reshape_2d(ctx, tensors.value_reshape, kQwenFlashHeadSize * key_value_head_count, token_count);
    REQUIRE(key_cache_rows != nullptr);
    REQUIRE(value_cache_rows != nullptr);

    tensors.key_cache           = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    tensors.value_cache         = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    tensors.key_cache_indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    tensors.value_cache_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    REQUIRE(tensors.key_cache != nullptr);
    REQUIRE(tensors.value_cache != nullptr);
    REQUIRE(tensors.key_cache_indices != nullptr);
    REQUIRE(tensors.value_cache_indices != nullptr);

    tensors.key_output   = ggml_set_rows(ctx, tensors.key_cache, key_cache_rows, tensors.key_cache_indices);
    tensors.value_output = ggml_set_rows(ctx, tensors.value_cache, value_cache_rows, tensors.value_cache_indices);
    REQUIRE(tensors.key_output != nullptr);
    REQUIRE(tensors.value_output != nullptr);
    return tensors;
}

static ggml::hrx::GraphImportResult import_qwen_attention_postprocess_graph(
    ggml_context *                          ctx,
    const QwenAttentionPostprocessTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.query_output);
    ggml_build_forward_expand(graph, tensors.key_output);
    ggml_build_forward_expand(graph, tensors.value_output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static ggml_tensor * append_qwen_flash_attention_consumer(ggml_context *                    ctx,
                                                          QwenAttentionPostprocessTensors & tensors,
                                                          int64_t                           token_count,
                                                          int64_t                           query_head_count,
                                                          int64_t                           key_value_head_count,
                                                          int64_t                           cache_row_count) {
    ggml_tensor * query_layout =
        ggml_reshape_3d(ctx, tensors.query_output, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * query_permute = ggml_permute(ctx, query_layout, 0, 2, 1, 3);
    ggml_tensor * key_cache_layout =
        ggml_reshape_3d(ctx, tensors.key_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * key_permute = ggml_permute(ctx, key_cache_layout, 0, 2, 1, 3);
    ggml_tensor * value_cache_layout =
        ggml_reshape_3d(ctx, tensors.value_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * value_permute = ggml_permute(ctx, value_cache_layout, 0, 2, 1, 3);
    tensors.mask                = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cache_row_count, token_count);
    REQUIRE(query_layout != nullptr);
    REQUIRE(query_permute != nullptr);
    REQUIRE(key_cache_layout != nullptr);
    REQUIRE(key_permute != nullptr);
    REQUIRE(value_cache_layout != nullptr);
    REQUIRE(value_permute != nullptr);
    REQUIRE(tensors.mask != nullptr);

    tensors.flash_output = ggml_flash_attn_ext(ctx, query_permute, key_permute, value_permute, tensors.mask,
                                               1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize)), 0.0f, 0.0f);
    REQUIRE(tensors.flash_output != nullptr);
    return tensors.flash_output;
}

static void schedule_qwen_attention_postprocess_command(ggml_context *                          ctx,
                                                        const QwenAttentionPostprocessTensors & tensors,
                                                        int64_t                                 token_count,
                                                        int64_t                                 query_head_count,
                                                        int64_t                                 key_value_head_count,
                                                        int64_t                                 cache_row_count,
                                                        bool expect_synthetic_inverse_frequencies = false) {
    ggml::hrx::GraphImportResult imported = import_qwen_attention_postprocess_graph(ctx, tensors);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().initialization_dispatches.empty());
    REQUIRE(scheduler.plan().dispatches.size() == 4);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.back();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "qwen3_moe:qwen3_moe_attention_postprocess_f32_f16");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_row_count);
    REQUIRE(dispatch.bindings.size() == 12);
    require_compile_parameter(dispatch, "qwen3_moe.model.rms_epsilon", "0.000001");
    require_compile_parameter(dispatch, "qwen3_moe.attention.head_size", std::to_string(kQwenFlashHeadSize));
    require_compile_parameter(dispatch, "qwen3_moe.attention.query_size",
                              std::to_string(query_head_count * kQwenFlashHeadSize));
    require_compile_parameter(dispatch, "qwen3_moe.attention.key_value_size",
                              std::to_string(key_value_head_count * kQwenFlashHeadSize));
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

    const ggml::hrx::Value * positions_value     = imported.graph.values().find_tensor(tensors.positions);
    const ggml::hrx::Value * key_indices_value   = imported.graph.values().find_tensor(tensors.key_cache_indices);
    const ggml::hrx::Value * value_indices_value = imported.graph.values().find_tensor(tensors.value_cache_indices);
    const ggml::hrx::Value * query_raw_value     = imported.graph.values().find_tensor(tensors.query_raw);
    const ggml::hrx::Value * key_raw_value       = imported.graph.values().find_tensor(tensors.key_raw);
    const ggml::hrx::Value * value_raw_value     = imported.graph.values().find_tensor(tensors.value_raw);
    const ggml::hrx::Value * query_output_value  = imported.graph.values().find_tensor(tensors.query_output);
    const ggml::hrx::Value * key_cache_value     = imported.graph.values().find_tensor(tensors.key_cache);
    const ggml::hrx::Value * value_cache_value   = imported.graph.values().find_tensor(tensors.value_cache);
    REQUIRE(positions_value != nullptr);
    REQUIRE(key_indices_value != nullptr);
    REQUIRE(value_indices_value != nullptr);
    REQUIRE(query_raw_value != nullptr);
    REQUIRE(key_raw_value != nullptr);
    REQUIRE(value_raw_value != nullptr);
    REQUIRE(query_output_value != nullptr);
    REQUIRE(key_cache_value != nullptr);
    REQUIRE(value_cache_value != nullptr);
    REQUIRE(dispatch.bindings[0].value == positions_value->id);
    REQUIRE(dispatch.bindings[1].value == key_indices_value->id);
    REQUIRE(dispatch.bindings[2].value == value_indices_value->id);
    REQUIRE(dispatch.bindings[3].value == query_raw_value->id);
    REQUIRE(dispatch.bindings[4].value == key_raw_value->id);
    REQUIRE(dispatch.bindings[5].value == value_raw_value->id);
    REQUIRE(dispatch.bindings[9].value == query_output_value->id);
    REQUIRE(dispatch.bindings[10].value == key_cache_value->id);
    REQUIRE(dispatch.bindings[11].value == value_cache_value->id);
    if (expect_synthetic_inverse_frequencies) {
        REQUIRE(scheduler.plan().transients.size() == 1);
        REQUIRE(scheduler.plan().constant_initializations.size() == 1);
        REQUIRE(dispatch.bindings[8].value == scheduler.plan().transients[0].value);
        REQUIRE(scheduler.plan().constant_initializations[0].value == dispatch.bindings[8].value);
        REQUIRE(scheduler.plan().constant_initializations[0].data.size() ==
                static_cast<size_t>(kQwenFlashHeadSize / 2) * sizeof(float));
    } else {
        REQUIRE(scheduler.plan().transients.empty());
        REQUIRE(scheduler.plan().constant_initializations.empty());
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.initialization_commands.empty());
    REQUIRE(commands.commands.size() == 4);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.back().bindings.size() == 12);
    REQUIRE(commands.commands.back().bindings[0].name == "positions");
    REQUIRE(commands.commands.back().bindings[1].name == "key_cache_indices");
    REQUIRE(commands.commands.back().bindings[2].name == "value_cache_indices");
    REQUIRE(commands.commands.back().bindings[3].name == "query_input");
    REQUIRE(commands.commands.back().bindings[4].name == "key_input");
    REQUIRE(commands.commands.back().bindings[5].name == "value_input");
    REQUIRE(commands.commands.back().bindings[6].name == "query_norm_weight");
    REQUIRE(commands.commands.back().bindings[7].name == "key_norm_weight");
    REQUIRE(commands.commands.back().bindings[8].name == "inverse_frequencies");
    REQUIRE(commands.commands.back().bindings[9].name == "query_output");
    REQUIRE(commands.commands.back().bindings[10].name == "key_cache");
    REQUIRE(commands.commands.back().bindings[11].name == "value_cache");
    REQUIRE(commands.constant_initializations.size() == (expect_synthetic_inverse_frequencies ? 1 : 0));
}

static void run_qwen_attention_postprocess_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 8 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        constexpr int64_t                     token_count          = 4;
        constexpr int64_t                     query_head_count     = 4;
        constexpr int64_t                     key_value_head_count = 2;
        constexpr int64_t                     cache_row_count      = 16;
        const QwenAttentionPostprocessTensors tensors              = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count);
        schedule_qwen_attention_postprocess_command(ctx, tensors, token_count, query_head_count, key_value_head_count,
                                                    cache_row_count);
    }

    {
        const QwenAttentionPostprocessTensors tensors =
            build_qwen_attention_postprocess_graph(ctx, 4, 4, 2, 16, 0.00001f);
        ggml::hrx::GraphImportResult imported = import_qwen_attention_postprocess_graph(ctx, tensors);
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
            REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) !=
                    "qwen3_moe:qwen3_moe_attention_postprocess_f32_f16");
        }
    }

    {
        const QwenAttentionPostprocessTensors tensors =
            build_qwen_attention_postprocess_graph(ctx, 4, 4, 2, 16, 0.000001f, false);
        schedule_qwen_attention_postprocess_command(ctx, tensors, 4, 4, 2, 16, true);
    }

    {
        constexpr int64_t                     token_count          = 13;
        constexpr int64_t                     query_head_count     = 32;
        constexpr int64_t                     key_value_head_count = 4;
        constexpr int64_t                     cache_row_count      = 512;
        const QwenAttentionPostprocessTensors tensors              = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count, 0.000001f, false);
        schedule_qwen_attention_postprocess_command(ctx, tensors, token_count, query_head_count, key_value_head_count,
                                                    cache_row_count, true);
    }

    {
        constexpr int64_t token_count          = 13;
        constexpr int64_t query_head_count     = 32;
        constexpr int64_t key_value_head_count = 4;
        constexpr int64_t cache_row_count      = 512;

        const QwenAttentionPostprocessTensors first = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count, 0.000001f, false);
        const QwenAttentionPostprocessTensors second = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count, 0.000001f, false);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, first.query_output);
        ggml_build_forward_expand(graph, first.key_output);
        ggml_build_forward_expand(graph, first.value_output);
        ggml_build_forward_expand(graph, second.query_output);
        ggml_build_forward_expand(graph, second.key_output);
        ggml_build_forward_expand(graph, second.value_output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 8);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 8);
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t               token_count          = 4;
        constexpr int64_t               query_head_count     = 4;
        constexpr int64_t               key_value_head_count = 2;
        constexpr int64_t               cache_row_count      = 16;
        QwenAttentionPostprocessTensors tensors              = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count);
        ggml_tensor * flash_output = append_qwen_flash_attention_consumer(ctx, tensors, token_count, query_head_count,
                                                                          key_value_head_count, cache_row_count);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, tensors.key_output);
        ggml_build_forward_expand(graph, tensors.value_output);
        ggml_build_forward_expand(graph, flash_output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        std::vector<bool>        covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan   plan;
        ggml::hrx::DispatchMatch match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes,
                                        producer_index_for_tensor(imported.graph, tensors.query_reshape), match));
        append_match_to_plan(plan, match, covered_nodes);
        REQUIRE(plan.valid());
        REQUIRE(plan.initialization_dispatches.size() == 2);

        const ggml::hrx::Dispatch & context_capture = plan.initialization_dispatches[0];
        const ggml::hrx::Dispatch & metadata        = plan.initialization_dispatches[1];
        REQUIRE(kernel_name_for_id(context_capture.kernel.kernel_id) ==
                "qwen_owned:qwen_attention_context_base_capture");
        REQUIRE(kernel_name_for_id(metadata.kernel.kernel_id) == "qwen3_moe:qwen_attention_metadata");
        REQUIRE(context_capture.bindings.size() == 2);
        REQUIRE(metadata.bindings.size() == 5);
        REQUIRE(context_capture.bindings[1].value == metadata.bindings[0].value);
        REQUIRE(metadata.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(metadata.kernel.integer_parameters.at("context_capacity") == cache_row_count);

        const ggml::hrx::Value * positions_value     = imported.graph.values().find_tensor(tensors.positions);
        const ggml::hrx::Value * key_indices_value   = imported.graph.values().find_tensor(tensors.key_cache_indices);
        const ggml::hrx::Value * value_indices_value = imported.graph.values().find_tensor(tensors.value_cache_indices);
        const ggml::hrx::Value * mask_value          = imported.graph.values().find_tensor(tensors.mask);
        REQUIRE(positions_value != nullptr);
        REQUIRE(key_indices_value != nullptr);
        REQUIRE(value_indices_value != nullptr);
        REQUIRE(mask_value != nullptr);
        REQUIRE(context_capture.bindings[0].value == positions_value->id);
        REQUIRE(metadata.bindings[1].value == positions_value->id);
        REQUIRE(metadata.bindings[2].value == key_indices_value->id);
        REQUIRE(metadata.bindings[3].value == value_indices_value->id);
        REQUIRE(metadata.bindings[4].value == mask_value->id);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.initialization_commands.size() == 2);
        REQUIRE(commands.initialization_commands[0].bindings[0].name == "positions");
        REQUIRE(commands.initialization_commands[0].bindings[1].name == "control");
        REQUIRE(commands.initialization_commands[1].bindings[0].name == "control");
        REQUIRE(commands.initialization_commands[1].bindings[4].name == "attention_mask");
        REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, context_capture.bindings[1].value) !=
                nullptr);
    }

    ggml_free(ctx);
}

static void run_qwen_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 8 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 2, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 256);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 256);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(bias != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * output = ggml_add(ctx, projection, bias);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(ctx, output, "loom_libs:ggml_mul_mat_bias_f32_f32_wmma", GGML_TYPE_F16,
                                              33, 2048, 256, 2, { GGML_OP_MUL_MAT, GGML_OP_ADD },
                                              { "input", "weight", "bias", "output" }, false);
    }

    {
        ggml_tensor * weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(residual != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * output = ggml_add(ctx, projection, residual);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(ctx, output, "loom_libs:ggml_mul_mat_add_f32_f32_wmma", GGML_TYPE_F16, 33,
                                              2048, 256, 2, { GGML_OP_MUL_MAT, GGML_OP_ADD },
                                              { "input", "weight", "residual_input", "residual_output" }, false);
    }

    {
        ggml_tensor * weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * bias     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(bias != nullptr);
        REQUIRE(residual != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * biased = ggml_add(ctx, projection, bias);
        REQUIRE(biased != nullptr);
        ggml_tensor * output = ggml_add(ctx, biased, residual);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_bias_add_f32_f32_wmma", GGML_TYPE_F16, 33, 2048, 256, 3,
            { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD },
            { "input", "weight", "bias", "residual_input", "residual_output" }, false);
    }

    {
        ggml_tensor * weight      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * residual    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(residual != nullptr);
        REQUIRE(norm_weight != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * added = ggml_add(ctx, projection, residual);
        REQUIRE(added != nullptr);
        ggml_tensor * rms = ggml_rms_norm(ctx, added, 0.000001f);
        REQUIRE(rms != nullptr);
        ggml_tensor * output = ggml_mul(ctx, rms, norm_weight);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(ctx, output, "loom_libs:ggml_mul_mat_add_f32_f32_wmma", GGML_TYPE_F16, 33,
                                              2048, 256, 4,
                                              { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL },
                                              { "input", "weight", "residual_input", "residual_output" }, true);
    }

    {
        ggml_tensor * weight      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * bias        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * residual    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(bias != nullptr);
        REQUIRE(residual != nullptr);
        REQUIRE(norm_weight != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * biased = ggml_add(ctx, projection, bias);
        REQUIRE(biased != nullptr);
        ggml_tensor * added = ggml_add(ctx, biased, residual);
        REQUIRE(added != nullptr);
        ggml_tensor * rms = ggml_rms_norm(ctx, added, 0.000001f);
        REQUIRE(rms != nullptr);
        ggml_tensor * output = ggml_mul(ctx, rms, norm_weight);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_bias_add_f32_f32_wmma", GGML_TYPE_F16, 33, 2048, 256, 5,
            { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL },
            { "input", "weight", "bias", "residual_input", "residual_output" }, true);
    }

    {
        struct FormatCase {
            ggml_type type;
            int64_t   output_size;
        };

        const FormatCase cases[] = {
            { GGML_TYPE_Q4_K, 128 },
            { GGML_TYPE_Q6_K, 128 },
            { GGML_TYPE_Q8_0, 128 },
            { GGML_TYPE_Q8_1, 128 },
            { GGML_TYPE_F16,  128 },
            { GGML_TYPE_F32,  256 },
        };

        for (const FormatCase & c : cases) {
            ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, c.type, 2048, c.output_size);
            ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, c.type, 2048, c.output_size);
            ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
            REQUIRE(gate_weight != nullptr);
            REQUIRE(up_weight != nullptr);
            REQUIRE(input != nullptr);
            ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
            ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
            REQUIRE(gate != nullptr);
            REQUIRE(up != nullptr);
            ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
            REQUIRE(output != nullptr);
            schedule_fused_matmul_swiglu_command(ctx, output, c.type, c.type, 4, 2048, c.output_size);
        }
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_Q4_K, GGML_TYPE_F16, 4, 2048, 128);
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_F32, GGML_TYPE_Q8_1, 4, 2048, 128);
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * gate_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        ggml_tensor * up_input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(gate_input != nullptr);
        REQUIRE(up_input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, gate_input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, up_input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        require_matmul_swiglu_falls_back(ctx, output);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_silu(ctx, matmul);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_unary_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma",
                                            ggml::hrx::UnaryKind::Silu, 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_relu(ctx, matmul);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_unary_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma",
                                            ggml::hrx::UnaryKind::Relu, 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_gelu(ctx, matmul);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_unary_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma",
                                            ggml::hrx::UnaryKind::Gelu, 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * relu = ggml_relu(ctx, matmul);
        ggml_tensor * sqr  = ggml_sqr(ctx, matmul);
        REQUIRE(relu != nullptr);
        REQUIRE(sqr != nullptr);
        ggml_tensor * output = ggml_add(ctx, relu, sqr);
        REQUIRE(output != nullptr);
        require_matmul_root_matches_identity_single(ctx, output);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_softplus(ctx, matmul);
        REQUIRE(output != nullptr);
        require_matmul_root_matches_identity_single(ctx, output);
    }

    {
        static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
        const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
        const bool                    had_original_env        = original_env != nullptr;
        const std::string             original_env_value      = had_original_env ? original_env : "";
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);

        ggml_tensor * q4_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * q4_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(q4_weight != nullptr);
        REQUIRE(q4_input != nullptr);
        ggml_tensor * q4_output = ggml_mul_mat(ctx, q4_weight, q4_input);
        REQUIRE(q4_output != nullptr);
        schedule_single_matmul_command(ctx, q4_output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);

        ggml_tensor * q4_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * q4_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q4_decode_weight != nullptr);
        REQUIRE(q4_decode_input != nullptr);
        ggml_tensor * q4_decode_output = ggml_mul_mat(ctx, q4_decode_weight, q4_decode_input);
        REQUIRE(q4_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q4_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * q6_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * q6_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2);
        REQUIRE(q6_weight != nullptr);
        REQUIRE(q6_input != nullptr);
        ggml_tensor * q6_output = ggml_mul_mat(ctx, q6_weight, q6_input);
        REQUIRE(q6_output != nullptr);
        schedule_single_matmul_command(ctx, q6_output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 2, 2048, 128);

        ggml_tensor * q6_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * q6_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q6_decode_weight != nullptr);
        REQUIRE(q6_decode_input != nullptr);
        ggml_tensor * q6_decode_output = ggml_mul_mat(ctx, q6_decode_weight, q6_decode_input);
        REQUIRE(q6_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q6_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * q8_0_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * q8_0_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(q8_0_weight != nullptr);
        REQUIRE(q8_0_input != nullptr);
        ggml_tensor * q8_0_output = ggml_mul_mat(ctx, q8_0_weight, q8_0_input);
        REQUIRE(q8_0_output != nullptr);
        schedule_single_matmul_command(ctx, q8_0_output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);

        ggml_tensor * q8_0_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * q8_0_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q8_0_decode_weight != nullptr);
        REQUIRE(q8_0_decode_input != nullptr);
        ggml_tensor * q8_0_decode_output = ggml_mul_mat(ctx, q8_0_decode_weight, q8_0_decode_input);
        REQUIRE(q8_0_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q8_0_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * q8_1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * q8_1_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(q8_1_weight != nullptr);
        REQUIRE(q8_1_input != nullptr);
        ggml_tensor * q8_1_output = ggml_mul_mat(ctx, q8_1_weight, q8_1_input);
        REQUIRE(q8_1_output != nullptr);
        schedule_single_matmul_command(ctx, q8_1_output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);

        ggml_tensor * q8_1_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * q8_1_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q8_1_decode_weight != nullptr);
        REQUIRE(q8_1_decode_input != nullptr);
        ggml_tensor * q8_1_decode_output = ggml_mul_mat(ctx, q8_1_decode_weight, q8_1_decode_input);
        REQUIRE(q8_1_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q8_1_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * f16_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * f16_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(f16_weight != nullptr);
        REQUIRE(f16_input != nullptr);
        ggml_tensor * f16_output = ggml_mul_mat(ctx, f16_weight, f16_input);
        REQUIRE(f16_output != nullptr);
        schedule_single_matmul_command(ctx, f16_output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);

        ggml_tensor * f16_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * f16_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(f16_decode_weight != nullptr);
        REQUIRE(f16_decode_input != nullptr);
        ggml_tensor * f16_decode_output = ggml_mul_mat(ctx, f16_decode_weight, f16_decode_input);
        REQUIRE(f16_decode_output != nullptr);
        schedule_single_matmul_command(ctx, f16_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * f32_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * f32_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(f32_weight != nullptr);
        REQUIRE(f32_input != nullptr);
        ggml_tensor * f32_output = ggml_mul_mat(ctx, f32_weight, f32_input);
        REQUIRE(f32_output != nullptr);
        schedule_single_matmul_command(ctx, f32_output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);

        ggml_tensor * f32_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * f32_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(f32_decode_weight != nullptr);
        REQUIRE(f32_decode_input != nullptr);
        ggml_tensor * f32_decode_output = ggml_mul_mat(ctx, f32_decode_weight, f32_decode_input);
        REQUIRE(f32_decode_output != nullptr);
        schedule_single_matmul_command(ctx, f32_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 151936);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_dense_linear_q6k_f16_wmma", 1, 2048, 151936);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32", 4,
                                       2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32", 1,
                                       2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    {
        ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 128, 2);
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 4, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
}

static void run_llama_attention_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 16 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t input_size      = 2048;
    constexpr int64_t head_size       = 128;
    constexpr int64_t head_count      = 2;
    constexpr int64_t output_size     = head_size * head_count;
    constexpr int64_t token_count     = 7;
    constexpr int64_t cache_row_count = 16;

    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, output_size);
        ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(positions != nullptr);
        REQUIRE(freqs != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        ggml_tensor * query      = ggml_reshape_3d(ctx, projection, head_size, head_count, token_count);
        ggml_tensor * output     = ggml_rope_ext(ctx, query, positions, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0,
                                                 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(projection != nullptr);
        REQUIRE(query != nullptr);
        REQUIRE(output != nullptr);
        schedule_fused_llama_attention_matmul_command(
            ctx, output, "loom_libs:llm_attention_q_matmul_rope_f32_f32_wmma", GGML_TYPE_F16, token_count, input_size,
            output_size, head_size, head_count, 0, 0, 3, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ROPE },
            { "input", "weight", "positions", "theta", "freq_factors", "output" });
    }

    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, output_size);
        ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * cache     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, output_size, cache_row_count);
        ggml_tensor * indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(positions != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(cache != nullptr);
        REQUIRE(indices != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        ggml_tensor * key        = ggml_reshape_3d(ctx, projection, head_size, head_count, token_count);
        ggml_tensor * rope   = ggml_rope_ext(ctx, key, positions, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f,
                                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * rows   = ggml_reshape_2d(ctx, rope, output_size, token_count);
        ggml_tensor * output = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(projection != nullptr);
        REQUIRE(key != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(output != nullptr);
        schedule_fused_llama_attention_matmul_command(
            ctx, output, "loom_libs:llm_attention_k_matmul_rope_set_rows_f32_f32_wmma", GGML_TYPE_F16, token_count,
            input_size, output_size, head_size, head_count, cache_row_count, 16, 5,
            { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ROPE, GGML_OP_RESHAPE, GGML_OP_SET_ROWS },
            { "input", "weight", "positions", "indices", "theta", "freq_factors", "cache" });
    }

    {
        ggml_tensor * weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, output_size);
        ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, output_size, cache_row_count);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(cache != nullptr);
        REQUIRE(indices != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        ggml_tensor * value      = ggml_reshape_3d(ctx, projection, head_size, head_count, token_count);
        ggml_tensor * rows       = ggml_reshape_2d(ctx, value, output_size, token_count);
        ggml_tensor * output     = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(projection != nullptr);
        REQUIRE(value != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(output != nullptr);
        schedule_fused_llama_attention_matmul_command(
            ctx, output, "loom_libs:llm_attention_v_matmul_set_rows_f32_f32_wmma", GGML_TYPE_F16, token_count,
            input_size, output_size, 0, 0, cache_row_count, 16, 4,
            { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_SET_ROWS },
            { "input", "weight", "indices", "cache" });
    }

    ggml_free(ctx);
}

static void schedule_qwen_router_top8_command(ggml_context * ctx,
                                              ggml_tensor *  output,
                                              ggml_tensor *  route_ids,
                                              int64_t        expected_expert_count = kQwenRouterExpertCount,
                                              int64_t        expected_route_count  = kQwenRouterRouteCount) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 10);

    const ggml::hrx::Value * route_ids_value = imported.graph.values().find_tensor(route_ids);
    const ggml::hrx::Value * output_value    = imported.graph.values().find_tensor(output);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(output_value != nullptr);
    REQUIRE(route_ids_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(output_value->kind == ggml::hrx::ValueKind::External);

    const ggml::hrx::GraphNode * softmax_node  = nullptr;
    const ggml::hrx::GraphNode * get_rows_node = nullptr;
    for (const ggml::hrx::GraphNode & node : imported.graph.nodes()) {
        if (node.op == GGML_OP_SOFT_MAX) {
            softmax_node = &node;
        } else if (node.op == GGML_OP_GET_ROWS) {
            get_rows_node = &node;
        }
    }
    REQUIRE(softmax_node != nullptr);
    REQUIRE(get_rows_node != nullptr);
    const ggml::hrx::GraphNode * probs_reshape =
        ggml::hrx::find_single_layout_alias_consumer_with_op(imported.graph, softmax_node->output, GGML_OP_RESHAPE);
    REQUIRE(probs_reshape != nullptr);
    REQUIRE(ggml::hrx::is_layout_alias_node(imported.graph, *probs_reshape));
    REQUIRE(ggml::hrx::find_single_consumer_with_op_through_layout_aliases(imported.graph, softmax_node->output,
                                                                           GGML_OP_GET_ROWS) == get_rows_node);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());

    const int64_t token_count        = output->ne[2];
    const size_t  route_id_length    = static_cast<size_t>(token_count) * route_ids->nb[1];
    const size_t  expert_table_bytes = qwen_expert_table_size(token_count, expected_expert_count);
    const size_t  partition_table_bytes =
        qwen_partition_table_size(token_count, expected_route_count, expected_expert_count);
    const bool uses_fused_prefill_expert_table_partition =
        token_count == 512 && expected_route_count == 8 &&
        route_ids->nb[1] / sizeof(int32_t) == static_cast<size_t>(expected_route_count) && expected_expert_count == 128;
    REQUIRE(scheduler.plan().dispatches.size() == (uses_fused_prefill_expert_table_partition ? 2 : 3));
    REQUIRE(scheduler.plan().transients.size() == 2);
    REQUIRE(scheduler.plan().constant_initializations.empty());
    REQUIRE(scheduler.plan().completion_counter_requests.size() == (uses_fused_prefill_expert_table_partition ? 1 : 0));

    const ggml::hrx::CommandPlanTransient & expert_table_transient    = scheduler.plan().transients[0];
    const ggml::hrx::CommandPlanTransient & partition_table_transient = scheduler.plan().transients[1];
    REQUIRE(expert_table_transient.value.value == static_cast<int32_t>(imported.graph.values().size()));
    REQUIRE(expert_table_transient.name == "qwen.router.expert_table");
    REQUIRE(expert_table_transient.size == expert_table_bytes);
    REQUIRE(partition_table_transient.value.value == expert_table_transient.value.value + 1);
    REQUIRE(partition_table_transient.name == "qwen.router.partition_table");
    REQUIRE(partition_table_transient.size == partition_table_bytes);
    if (uses_fused_prefill_expert_table_partition) {
        const ggml::hrx::CommandPlanCompletionCounterRequest & completion_counter_request =
            scheduler.plan().completion_counter_requests[0];
        REQUIRE(completion_counter_request.value.value == partition_table_transient.value.value + 1);
        REQUIRE(completion_counter_request.name == "qwen.router.prefill_expert_table_partition_completion_counter");
        REQUIRE(completion_counter_request.count == 1);
    }

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches[0];
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "qwen3_moe:qwen3_moe_router_top8_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("route_id_stride") == route_ids->nb[1] / sizeof(int32_t));
    REQUIRE(dispatch.bindings.size() == 3);
    REQUIRE(dispatch.bindings[1].value == route_ids_value->id);
    REQUIRE(dispatch.bindings[1].length == route_id_length);
    REQUIRE(dispatch.bindings[2].value == output_value->id);
    require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", std::to_string(expected_expert_count));
    require_compile_parameter(dispatch, "qwen3_moe.router.route_count", std::to_string(expected_route_count));
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

    if (uses_fused_prefill_expert_table_partition) {
        const ggml::hrx::CommandPlanCompletionCounterRequest & completion_counter_request =
            scheduler.plan().completion_counter_requests[0];
        const ggml::hrx::Dispatch & expert_table_partition_dispatch = scheduler.plan().dispatches[1];
        REQUIRE(kernel_name_for_id(expert_table_partition_dispatch.kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_build_expert_table_partition_prefill_512");
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("route_count") == expected_route_count);
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("route_stride") ==
                route_ids->nb[1] / sizeof(int32_t));
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("expert_count") == expected_expert_count);
        REQUIRE(expert_table_partition_dispatch.bindings.size() == 4);
        REQUIRE(expert_table_partition_dispatch.bindings[0].value == route_ids_value->id);
        REQUIRE(expert_table_partition_dispatch.bindings[0].length == route_id_length);
        REQUIRE(expert_table_partition_dispatch.bindings[1].value == expert_table_transient.value);
        REQUIRE(expert_table_partition_dispatch.bindings[1].length == expert_table_bytes);
        REQUIRE(expert_table_partition_dispatch.bindings[2].value == partition_table_transient.value);
        REQUIRE(expert_table_partition_dispatch.bindings[2].length == partition_table_bytes);
        REQUIRE(expert_table_partition_dispatch.bindings[3].value == completion_counter_request.value);
        REQUIRE(expert_table_partition_dispatch.bindings[3].length == sizeof(int32_t));
    } else {
        const ggml::hrx::Dispatch & expert_table_dispatch = scheduler.plan().dispatches[1];
        REQUIRE(kernel_name_for_id(expert_table_dispatch.kernel.kernel_id) == "loom_libs:ggml_moe_build_expert_table");
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("route_count") == expected_route_count);
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("route_stride") ==
                route_ids->nb[1] / sizeof(int32_t));
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("expert_count") == expected_expert_count);
        REQUIRE(expert_table_dispatch.bindings.size() == 2);
        REQUIRE(expert_table_dispatch.bindings[0].value == route_ids_value->id);
        REQUIRE(expert_table_dispatch.bindings[0].length == route_id_length);
        REQUIRE(expert_table_dispatch.bindings[1].value == expert_table_transient.value);
        REQUIRE(expert_table_dispatch.bindings[1].length == expert_table_bytes);
        require_compile_parameter(expert_table_dispatch, "ggml.moe_routing.expert_count",
                                  std::to_string(expected_expert_count));
        require_compile_parameter(expert_table_dispatch, "ggml.moe_routing.route_count",
                                  std::to_string(expected_route_count));
        require_compile_parameter(expert_table_dispatch, "ggml.workload.token_capacity", std::to_string(token_count));

        const ggml::hrx::Dispatch & partition_table_dispatch = scheduler.plan().dispatches[2];
        REQUIRE(kernel_name_for_id(partition_table_dispatch.kernel.kernel_id) ==
                "loom_libs:ggml_moe_build_expert_partition_table");
        REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("route_count") == expected_route_count);
        REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("expert_count") == expected_expert_count);
        REQUIRE(partition_table_dispatch.bindings.size() == 2);
        REQUIRE(partition_table_dispatch.bindings[0].value == expert_table_transient.value);
        REQUIRE(partition_table_dispatch.bindings[0].length == expert_table_bytes);
        REQUIRE(partition_table_dispatch.bindings[1].value == partition_table_transient.value);
        REQUIRE(partition_table_dispatch.bindings[1].length == partition_table_bytes);
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.expert_count",
                                  std::to_string(expected_expert_count));
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.route_count",
                                  std::to_string(expected_route_count));
        require_compile_parameter(partition_table_dispatch, "ggml.workload.token_capacity",
                                  std::to_string(token_count));
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.descriptor_expert_mask", "127");
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.descriptor_partition_shift", "7");
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.descriptor_row_count_shift", "13");
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.partition_workgroup_size", "128");
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == (uses_fused_prefill_expert_table_partition ? 2 : 3));
    REQUIRE(commands.constant_initializations.empty());
    REQUIRE(commands.completion_counters.count == (uses_fused_prefill_expert_table_partition ? 1 : 0));
    REQUIRE(commands.completion_counters.byte_count ==
            (uses_fused_prefill_expert_table_partition ? sizeof(int32_t) : 0));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[0].name == "logits");
    REQUIRE(commands.commands[0].bindings[1].name == "route_ids");
    REQUIRE(commands.commands[0].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[0].bindings[1].value == route_ids_value->storage_root);
    REQUIRE(commands.commands[0].bindings[1].offset == route_ids_value->storage_offset);
    REQUIRE(commands.commands[0].bindings[1].length == dispatch.bindings[1].length);
    REQUIRE(commands.commands[0].bindings[2].name == "route_weights");
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(commands.commands[1].bindings.size() == (uses_fused_prefill_expert_table_partition ? 4 : 2));
    REQUIRE(commands.commands[1].bindings[0].name == "route_ids");
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[0].value == route_ids_value->storage_root);
    REQUIRE(commands.commands[1].bindings[0].offset == route_ids_value->storage_offset);
    REQUIRE(commands.commands[1].bindings[1].name == "expert_table");
    REQUIRE(commands.commands[1].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[1].length == expert_table_bytes);
    if (uses_fused_prefill_expert_table_partition) {
        REQUIRE(commands.commands[1].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[1].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[1].bindings[2].length == partition_table_bytes);
        REQUIRE(commands.commands[1].bindings[3].name == "completion_counter");
        REQUIRE(commands.commands[1].bindings[3].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[1].bindings[3].length == sizeof(int32_t));
    } else {
        REQUIRE(commands.commands[2].dependencies.size() == 1);
        REQUIRE(commands.commands[2].dependencies[0] == 1);
        REQUIRE(commands.commands[2].bindings.size() == 2);
        REQUIRE(commands.commands[2].bindings[0].name == "expert_table");
        REQUIRE(commands.commands[2].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[2].bindings[1].name == "partition_table");
        REQUIRE(commands.commands[2].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[2].bindings[1].length == partition_table_bytes);
    }
    const ggml::hrx::TransientAllocation * route_ids_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, route_ids_value->storage_root);
    REQUIRE(route_ids_allocation != nullptr);
    REQUIRE(route_ids_allocation->size == dispatch.bindings[1].length);
    const ggml::hrx::TransientAllocation * expert_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, expert_table_transient.value);
    REQUIRE(expert_table_allocation != nullptr);
    REQUIRE(expert_table_allocation->size == expert_table_bytes);
    const ggml::hrx::TransientAllocation * partition_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, partition_table_transient.value);
    REQUIRE(partition_table_allocation != nullptr);
    REQUIRE(partition_table_allocation->size == partition_table_bytes);
    if (uses_fused_prefill_expert_table_partition) {
        const ggml::hrx::CommandPlanCompletionCounterRequest & completion_counter_request =
            scheduler.plan().completion_counter_requests[0];
        const ggml::hrx::TransientAllocation * completion_counter_allocation =
            ggml::hrx::find_transient_allocation(commands.transients, completion_counter_request.value);
        REQUIRE(completion_counter_allocation != nullptr);
        REQUIRE(completion_counter_allocation->size == sizeof(int32_t));
        REQUIRE(completion_counter_allocation->arena_offset == commands.completion_counters.arena_offset);
    }
}

static void schedule_manual_qwen_router_top8_command(ggml::hrx::Graph & graph,
                                                     ggml::hrx::ValueId route_ids,
                                                     int64_t            token_count,
                                                     int64_t            expert_count,
                                                     int64_t            route_count) {
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 3);
    REQUIRE(scheduler.plan().transients.size() == 2);

    const size_t route_id_length       = static_cast<size_t>(token_count * route_count) * sizeof(int32_t);
    const size_t expert_table_bytes    = qwen_expert_table_size(token_count, expert_count);
    const size_t partition_table_bytes = qwen_partition_table_size(token_count, route_count, expert_count);

    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:qwen3_moe_router_top8_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("route_id_stride") == route_count);
    REQUIRE(dispatch.bindings.size() == 3);
    REQUIRE(dispatch.bindings[1].value == route_ids);
    REQUIRE(dispatch.bindings[1].length == route_id_length);
    require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", std::to_string(expert_count));
    require_compile_parameter(dispatch, "qwen3_moe.router.route_count", std::to_string(route_count));

    const ggml::hrx::CommandPlanTransient & expert_table_transient    = scheduler.plan().transients[0];
    const ggml::hrx::CommandPlanTransient & partition_table_transient = scheduler.plan().transients[1];
    REQUIRE(expert_table_transient.size == expert_table_bytes);
    REQUIRE(partition_table_transient.size == partition_table_bytes);

    const ggml::hrx::CommandPlanMoeRoutingBundle * bundle =
        scheduler.plan().metadata.find_moe_routing_bundle(route_ids);
    REQUIRE(bundle != nullptr);
    REQUIRE(bundle->token_count == token_count);
    REQUIRE(bundle->route_count == route_count);
    REQUIRE(bundle->route_stride == route_count);
    REQUIRE(bundle->expert_count == expert_count);
    REQUIRE(bundle->expert_table_byte_count == expert_table_bytes);
    REQUIRE(bundle->partition_table_byte_count == partition_table_bytes);

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 3);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings[1].value == route_ids);
    REQUIRE(commands.commands[0].bindings[1].length == route_id_length);
}

struct QwenRoutedGateUpTensors {
    ggml_tensor *              route_ids     = nullptr;
    ggml_tensor *              route_weights = nullptr;
    ggml_tensor *              input         = nullptr;
    ggml_tensor *              gate_weight   = nullptr;
    ggml_tensor *              up_weight     = nullptr;
    ggml_tensor *              down_weight   = nullptr;
    ggml_tensor *              gate          = nullptr;
    ggml_tensor *              up            = nullptr;
    ggml_tensor *              glu           = nullptr;
    ggml_tensor *              output        = nullptr;
    ggml_tensor *              weighted      = nullptr;
    ggml_tensor *              hidden_state  = nullptr;
    ggml_tensor *              residual      = nullptr;
    ggml_tensor *              next_rms      = nullptr;
    ggml_tensor *              next_output   = nullptr;
    ggml_tensor *              hidden_use    = nullptr;
    std::vector<ggml_tensor *> route_views;
};

static QwenRoutedGateUpTensors build_qwen_routed_gate_up_graph(ggml_context * ctx,
                                                               int64_t        token_count,
                                                               ggml_glu_op    glu_op           = GGML_GLU_OP_SWIGLU,
                                                               ggml_type      up_weight_type   = GGML_TYPE_Q4_K,
                                                               bool           include_down     = true,
                                                               ggml_type      down_weight_type = GGML_TYPE_Q6_K) {
    QwenRoutedGateUpTensors tensors;
    ggml_tensor *           logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
    REQUIRE(logits != nullptr);
    tensors.route_weights = build_qwen_router_top8_graph(ctx, logits, &tensors.route_ids);
    REQUIRE(tensors.route_weights != nullptr);
    REQUIRE(tensors.route_ids != nullptr);

    tensors.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, 1, token_count);
    tensors.gate_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenMoeHiddenSize, kQwenMoeIntermediateSize, kQwenRouterExpertCount);
    tensors.up_weight =
        ggml_new_tensor_3d(ctx, up_weight_type, kQwenMoeHiddenSize, kQwenMoeIntermediateSize, kQwenRouterExpertCount);
    REQUIRE(tensors.input != nullptr);
    REQUIRE(tensors.gate_weight != nullptr);
    REQUIRE(tensors.up_weight != nullptr);

    tensors.gate = ggml_mul_mat_id(ctx, tensors.gate_weight, tensors.input, tensors.route_ids);
    tensors.up   = ggml_mul_mat_id(ctx, tensors.up_weight, tensors.input, tensors.route_ids);
    REQUIRE(tensors.gate != nullptr);
    REQUIRE(tensors.up != nullptr);
    tensors.glu = ggml_glu_split(ctx, tensors.gate, tensors.up, glu_op);
    REQUIRE(tensors.glu != nullptr);

    if (include_down) {
        tensors.down_weight = ggml_new_tensor_3d(ctx, down_weight_type, kQwenMoeIntermediateSize, kQwenMoeHiddenSize,
                                                 kQwenRouterExpertCount);
        REQUIRE(tensors.down_weight != nullptr);
        tensors.output = ggml_mul_mat_id(ctx, tensors.down_weight, tensors.glu, tensors.route_ids);
        REQUIRE(tensors.output != nullptr);
    } else {
        tensors.output = tensors.glu;
    }
    return tensors;
}

static void append_qwen_weighted_reduce_tail(ggml_context *            ctx,
                                             QwenRoutedGateUpTensors & tensors,
                                             bool                      include_next_rmsnorm = false) {
    REQUIRE(tensors.output != nullptr);
    REQUIRE(tensors.route_weights != nullptr);
    tensors.weighted = ggml_mul(ctx, tensors.output, tensors.route_weights);
    REQUIRE(tensors.weighted != nullptr);
    tensors.route_views.clear();
    for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
        ggml_tensor * view =
            ggml_view_2d(ctx, tensors.weighted, kQwenMoeHiddenSize, tensors.weighted->ne[2], tensors.weighted->nb[2],
                         static_cast<size_t>(route) * tensors.weighted->nb[1]);
        REQUIRE(view != nullptr);
        tensors.route_views.push_back(view);
    }

    ggml_tensor * reduced = tensors.route_views.front();
    for (size_t i = 1; i < tensors.route_views.size(); ++i) {
        reduced = ggml_add(ctx, reduced, tensors.route_views[i]);
        REQUIRE(reduced != nullptr);
    }

    tensors.hidden_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, tensors.output->ne[2]);
    REQUIRE(tensors.hidden_state != nullptr);
    tensors.residual = ggml_add(ctx, tensors.hidden_state, reduced);
    REQUIRE(tensors.residual != nullptr);

    if (include_next_rmsnorm) {
        ggml_tensor * next_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize);
        REQUIRE(next_norm_weight != nullptr);
        tensors.next_rms = ggml_rms_norm(ctx, tensors.residual, 0.000001f);
        REQUIRE(tensors.next_rms != nullptr);
        tensors.next_output = ggml_mul(ctx, tensors.next_rms, next_norm_weight);
        REQUIRE(tensors.next_output != nullptr);
    }
}

static ggml::hrx::GraphImportResult import_qwen_routed_gate_up_graph(ggml_context *                  ctx,
                                                                     const QwenRoutedGateUpTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.route_weights);
    ggml_build_forward_expand(graph, tensors.output);
    if (tensors.residual != nullptr) {
        ggml_build_forward_expand(graph, tensors.residual);
    }
    if (tensors.next_output != nullptr) {
        ggml_build_forward_expand(graph, tensors.next_output);
    }
    if (tensors.hidden_use != nullptr) {
        ggml_build_forward_expand(graph, tensors.hidden_use);
    }

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static bool match_dispatch_at_index(const ggml::hrx::Graph &       graph,
                                    const ggml::hrx::CommandPlan & plan,
                                    const std::vector<bool> &      covered_nodes,
                                    size_t                         node_index,
                                    ggml::hrx::DispatchMatch &     match) {
    REQUIRE(node_index < graph.nodes().size());
    const ggml::hrx::DispatchMatchContext context = {
        graph,
        &graph.nodes()[node_index],
        node_index,
        covered_nodes,
        plan,
        ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size() + plan.transients.size() +
                                                plan.completion_counter_requests.size())),
    };
    return test_dispatch_registry().match(context, match);
}

static void append_match_to_plan(ggml::hrx::CommandPlan &   plan,
                                 ggml::hrx::DispatchMatch & match,
                                 std::vector<bool> &        covered_nodes,
                                 ggml::hrx::Graph *         graph) {
    if (graph != nullptr) {
        for (const ggml::hrx::DispatchValueAliasRequest & alias : match.value_aliases) {
            ggml::hrx::Status status = graph->values().alias_storage(alias.target_value, alias.source_value);
            REQUIRE(status.success());
        }
    }
    for (ggml::hrx::Dispatch & dispatch : match.initialization_dispatches) {
        plan.initialization_dispatches.push_back(std::move(dispatch));
    }
    for (ggml::hrx::Dispatch & dispatch : match.dispatches) {
        plan.dispatches.push_back(std::move(dispatch));
    }
    for (ggml::hrx::CommandPlanTransient & transient : match.transients) {
        plan.transients.push_back(std::move(transient));
    }
    for (ggml::hrx::CommandPlanConstantInitialization & initialization : match.constant_initializations) {
        plan.constant_initializations.push_back(std::move(initialization));
    }
    for (ggml::hrx::CommandPlanCompletionCounterRequest & request : match.completion_counter_requests) {
        plan.completion_counter_requests.push_back(std::move(request));
    }
    REQUIRE(plan.metadata.append(std::move(match.metadata), plan.status));
    for (const size_t covered_node : match.covered_nodes) {
        REQUIRE(covered_node < covered_nodes.size());
        REQUIRE(!covered_nodes[covered_node]);
        covered_nodes[covered_node] = true;
    }
}

static ggml::hrx::CommandPlan build_qwen_router_plan_for_graph(ggml::hrx::Graph &  graph,
                                                               std::vector<bool> & covered_nodes) {
    ggml::hrx::CommandPlan plan;
    size_t                 softmax_index = graph.nodes().size();
    for (size_t i = 0; i < graph.nodes().size(); ++i) {
        if (graph.nodes()[i].op == GGML_OP_SOFT_MAX) {
            softmax_index = i;
            break;
        }
    }
    REQUIRE(softmax_index < graph.nodes().size());
    ggml::hrx::DispatchMatch router_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, softmax_index, router_match));
    append_match_to_plan(plan, router_match, covered_nodes, &graph);
    return plan;
}

static void append_qwen_routed_gate_up_for_graph(ggml::hrx::Graph &              graph,
                                                 const QwenRoutedGateUpTensors & tensors,
                                                 std::vector<bool> &             covered_nodes,
                                                 ggml::hrx::CommandPlan &        plan) {
    const size_t             gate_index = producer_index_for_tensor(graph, tensors.gate);
    ggml::hrx::DispatchMatch gate_up_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, gate_index, gate_up_match));
    append_match_to_plan(plan, gate_up_match, covered_nodes, &graph);
}

static void append_qwen_routed_down_for_graph(ggml::hrx::Graph &              graph,
                                              const QwenRoutedGateUpTensors & tensors,
                                              std::vector<bool> &             covered_nodes,
                                              ggml::hrx::CommandPlan &        plan,
                                              const char *                    expected_kernel_name) {
    const size_t             down_index = producer_index_for_tensor(graph, tensors.output);
    ggml::hrx::DispatchMatch down_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, down_index, down_match));
    append_match_to_plan(plan, down_match, covered_nodes, &graph);

    REQUIRE(plan.dispatches.size() >= 1);
    REQUIRE(plan.transients.size() >= 1);
    const ggml::hrx::Value * route_ids_value = graph.values().find_tensor(tensors.route_ids);
    const ggml::hrx::Value * glu_value       = graph.values().find_tensor(tensors.glu);
    const ggml::hrx::Value * output_value    = graph.values().find_tensor(tensors.output);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(glu_value != nullptr);
    REQUIRE(output_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        plan.metadata.find_moe_routing_bundle(route_ids_value->id);
    const ggml::hrx::CommandPlanAlternateValue * gate_up_alternate = plan.metadata.find_alternate_value(
        glu_value->id, GGML_TYPE_F16, qwen_routed_gate_up_f16_output_size(tensors.output->ne[2]));
    const ggml::hrx::CommandPlanAlternateValue * routed_down_alternate = plan.metadata.find_alternate_value(
        output_value->id, GGML_TYPE_F16, qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(routing_bundle != nullptr);
    REQUIRE(gate_up_alternate != nullptr);
    REQUIRE(routed_down_alternate != nullptr);
    const ggml::hrx::Dispatch &             dispatch              = plan.dispatches.back();
    const ggml::hrx::CommandPlanTransient & routed_down_transient = plan.transients.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == 4);
    REQUIRE(routed_down_transient.name == "qwen.moe.routed_down_f16");
    REQUIRE(routed_down_transient.size == qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[0].value == gate_up_alternate->alternate_value);
    REQUIRE(dispatch.bindings[0].length == qwen_routed_gate_up_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[1].length == routing_bundle->expert_table_byte_count);
    REQUIRE(routed_down_alternate->alternate_value == routed_down_transient.value);
    REQUIRE(routed_down_alternate->byte_count == routed_down_transient.size);
    REQUIRE(dispatch.bindings[3].value == routed_down_transient.value);
    REQUIRE(dispatch.bindings[3].length == routed_down_transient.size);
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.input_size", "768");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.route_count", "8");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.expert_count", "128");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.output_size", "2048");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.weight_format",
                              tensors.down_weight->type == GGML_TYPE_Q4_K ? "4" : "6");
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(tensors.output->ne[2]));
}

static std::string common_mul_mat_weight_format(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return "4";
        case GGML_TYPE_Q6_K:
            return "6";
        case GGML_TYPE_Q8_0:
            return "80";
        case GGML_TYPE_Q8_1:
            return "81";
        case GGML_TYPE_F16:
            return "16";
        case GGML_TYPE_F32:
            return "32";
        default:
            return "0";
    }
}

static ggml::hrx::DispatchMatch require_common_mul_mat_id_for_graph(ggml::hrx::Graph &       graph,
                                                                    ggml::hrx::CommandPlan & plan,
                                                                    std::vector<bool> &      covered_nodes,
                                                                    ggml_tensor *            output,
                                                                    ggml_tensor *            input,
                                                                    ggml_tensor *            weight,
                                                                    ggml_tensor *            route_ids) {
    const ggml::hrx::Value * output_value    = graph.values().find_tensor(output);
    const ggml::hrx::Value * input_value     = graph.values().find_tensor(input);
    const ggml::hrx::Value * weight_value    = graph.values().find_tensor(weight);
    const ggml::hrx::Value * route_ids_value = graph.values().find_tensor(route_ids);
    REQUIRE(output_value != nullptr);
    REQUIRE(input_value != nullptr);
    REQUIRE(weight_value != nullptr);
    REQUIRE(route_ids_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        plan.metadata.find_moe_routing_bundle(route_ids_value->id);
    REQUIRE(routing_bundle != nullptr);

    const size_t             output_index = producer_index_for_tensor(graph, output);
    ggml::hrx::DispatchMatch match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, output_index, match));
    REQUIRE(match.dispatches.size() == 1);
    const ggml::hrx::Dispatch & dispatch = match.dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_mul_mat_id_f32_f32_wmma");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == output->ne[2]);
    REQUIRE(dispatch.bindings.size() == 5);
    REQUIRE(dispatch.bindings[0].value == input_value->id);
    REQUIRE(dispatch.bindings[0].length == input_value->byte_count);
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[1].length == qwen_expert_table_size(output->ne[2], weight->ne[2]));
    REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
    REQUIRE(dispatch.bindings[2].length == qwen_partition_table_size(output->ne[2], route_ids->ne[0], weight->ne[2]));
    REQUIRE(dispatch.bindings[3].value == weight_value->id);
    REQUIRE(dispatch.bindings[3].length == weight_value->byte_count);
    REQUIRE(dispatch.bindings[4].value == output_value->id);
    REQUIRE(dispatch.bindings[4].length == output_value->byte_count);
    require_compile_parameter(dispatch, "ggml.mul_mat_id.input_size", std::to_string(weight->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.output_size", std::to_string(weight->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.expert_count", std::to_string(weight->ne[2]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.route_count", std::to_string(route_ids->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.input_route_count", std::to_string(input->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.weight_format", common_mul_mat_weight_format(weight->type));
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(output->ne[2]));
    return match;
}

struct CommonMulMatIdSwiGLUTensors {
    ggml_tensor * input       = nullptr;
    ggml_tensor * route_ids   = nullptr;
    ggml_tensor * gate_weight = nullptr;
    ggml_tensor * up_weight   = nullptr;
    ggml_tensor * gate        = nullptr;
    ggml_tensor * up          = nullptr;
    ggml_tensor * output      = nullptr;
};

static CommonMulMatIdSwiGLUTensors build_common_mul_mat_id_swiglu_graph(ggml_context * ctx,
                                                                        ggml_type      gate_weight_type,
                                                                        ggml_type      up_weight_type,
                                                                        ggml_glu_op    glu_op = GGML_GLU_OP_SWIGLU) {
    CommonMulMatIdSwiGLUTensors tensors;
    constexpr int64_t           token_count       = 4;
    constexpr int64_t           route_count       = 8;
    constexpr int64_t           input_route_count = 1;
    constexpr int64_t           input_size        = 256;
    constexpr int64_t           output_size       = 128;
    constexpr int64_t           expert_count      = 16;
    tensors.route_ids                             = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, token_count);
    tensors.input       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_size, input_route_count, token_count);
    tensors.gate_weight = ggml_new_tensor_3d(ctx, gate_weight_type, input_size, output_size, expert_count);
    tensors.up_weight   = ggml_new_tensor_3d(ctx, up_weight_type, input_size, output_size, expert_count);
    REQUIRE(tensors.route_ids != nullptr);
    REQUIRE(tensors.input != nullptr);
    REQUIRE(tensors.gate_weight != nullptr);
    REQUIRE(tensors.up_weight != nullptr);

    tensors.gate = ggml_mul_mat_id(ctx, tensors.gate_weight, tensors.input, tensors.route_ids);
    tensors.up   = ggml_mul_mat_id(ctx, tensors.up_weight, tensors.input, tensors.route_ids);
    REQUIRE(tensors.gate != nullptr);
    REQUIRE(tensors.up != nullptr);
    tensors.output = ggml_glu_split(ctx, tensors.gate, tensors.up, glu_op);
    REQUIRE(tensors.output != nullptr);
    return tensors;
}

static ggml::hrx::GraphImportResult import_common_mul_mat_id_swiglu_graph(ggml_context *                      ctx,
                                                                          const CommonMulMatIdSwiGLUTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static void require_common_mul_mat_id_swiglu_match(ggml_context *                      ctx,
                                                   const CommonMulMatIdSwiGLUTensors & tensors,
                                                   ggml_tensor *                       root_projection) {
    ggml::hrx::GraphImportResult imported = import_common_mul_mat_id_swiglu_graph(ctx, tensors);
    std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
    ggml::hrx::CommandPlan       plan;
    const size_t                 root_index = producer_index_for_tensor(imported.graph, root_projection);
    ggml::hrx::DispatchMatch     match;
    REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, root_index, match));
    REQUIRE(match.dispatches.size() == 3);
    REQUIRE(match.transients.size() == 2);
    REQUIRE(match.metadata.generated_resources().size() == 2);
    REQUIRE(match.metadata.moe_routing_bundles().size() == 1);

    const ggml::hrx::Dispatch & dispatch = match.dispatches.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_mul_mat_id_swiglu_f32_f32_wmma");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == 6);
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.input_size",
                              std::to_string(tensors.gate_weight->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.output_size",
                              std::to_string(tensors.gate_weight->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.expert_count",
                              std::to_string(tensors.gate_weight->ne[2]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.route_count", std::to_string(tensors.route_ids->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.input_route_count",
                              std::to_string(tensors.input->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.gate_weight_format",
                              common_mul_mat_weight_format(tensors.gate_weight->type));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.up_weight_format",
                              common_mul_mat_weight_format(tensors.up_weight->type));
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(tensors.output->ne[2]));

    const ggml::hrx::Value * input_value       = imported.graph.values().find_tensor(tensors.input);
    const ggml::hrx::Value * route_ids_value   = imported.graph.values().find_tensor(tensors.route_ids);
    const ggml::hrx::Value * gate_weight_value = imported.graph.values().find_tensor(tensors.gate_weight);
    const ggml::hrx::Value * up_weight_value   = imported.graph.values().find_tensor(tensors.up_weight);
    const ggml::hrx::Value * output_value      = imported.graph.values().find_tensor(tensors.output);
    REQUIRE(input_value != nullptr);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(gate_weight_value != nullptr);
    REQUIRE(up_weight_value != nullptr);
    REQUIRE(output_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        match.metadata.find_moe_routing_bundle(route_ids_value->id);
    REQUIRE(routing_bundle != nullptr);

    REQUIRE(dispatch.bindings[0].value == input_value->id);
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
    REQUIRE(dispatch.bindings[3].value == gate_weight_value->id);
    REQUIRE(dispatch.bindings[4].value == up_weight_value->id);
    REQUIRE(dispatch.bindings[5].value == output_value->id);

    append_match_to_plan(plan, match, covered_nodes, &imported.graph);
    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 3);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[2].bindings.size() == 6);
    REQUIRE(commands.commands[2].bindings[0].name == "input");
    REQUIRE(commands.commands[2].bindings[1].name == "expert_table");
    REQUIRE(commands.commands[2].bindings[2].name == "partition_table");
    REQUIRE(commands.commands[2].bindings[3].name == "gate_weight");
    REQUIRE(commands.commands[2].bindings[4].name == "up_weight");
    REQUIRE(commands.commands[2].bindings[5].name == "output");
}

static void run_common_mul_mat_id_swiglu_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_Q4_K, GGML_TYPE_F16);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_F32, GGML_TYPE_Q8_1);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.up);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_GLU_OP_GEGLU);
        ggml::hrx::GraphImportResult imported = import_common_mul_mat_id_swiglu_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan;
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, match));
        REQUIRE(kernel_name_for_id(match.dispatches.back().kernel.kernel_id) ==
                "loom_libs:ggml_mul_mat_id_f32_f32_wmma");
    }

    ggml_free(ctx);
}

struct CommonMulMatIdPostOpsTensors {
    ggml_tensor * input             = nullptr;
    ggml_tensor * weight            = nullptr;
    ggml_tensor * route_ids         = nullptr;
    ggml_tensor * projection        = nullptr;
    ggml_tensor * bias              = nullptr;
    ggml_tensor * residual_input    = nullptr;
    ggml_tensor * residual_output   = nullptr;
    ggml_tensor * rms_output        = nullptr;
    ggml_tensor * norm_weight       = nullptr;
    ggml_tensor * normalized_output = nullptr;
};

static int64_t common_mul_mat_id_partition_descriptor_capacity(const CommonMulMatIdPostOpsTensors & tensors) {
    const int64_t token_count                = tensors.projection->ne[2];
    const int64_t route_count                = tensors.route_ids->ne[0];
    const int64_t expert_count               = tensors.weight->ne[2];
    const int64_t assignment_count           = token_count * route_count;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return assignment_partition_count + expert_count;
}

static CommonMulMatIdPostOpsTensors build_common_mul_mat_id_postops_graph(ggml_context * ctx,
                                                                          bool           include_bias,
                                                                          bool           include_residual,
                                                                          bool           include_rmsnorm) {
    CommonMulMatIdPostOpsTensors tensors;
    constexpr int64_t            token_count  = 4;
    constexpr int64_t            route_count  = 8;
    constexpr int64_t            input_size   = 256;
    constexpr int64_t            output_size  = 128;
    constexpr int64_t            expert_count = 16;
    tensors.route_ids                         = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, token_count);
    tensors.input                             = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_size, 1, token_count);
    tensors.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, input_size, output_size, expert_count);
    REQUIRE(tensors.route_ids != nullptr);
    REQUIRE(tensors.input != nullptr);
    REQUIRE(tensors.weight != nullptr);

    tensors.projection = ggml_mul_mat_id(ctx, tensors.weight, tensors.input, tensors.route_ids);
    REQUIRE(tensors.projection != nullptr);

    ggml_tensor * current = tensors.projection;
    if (include_bias) {
        tensors.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_size);
        REQUIRE(tensors.bias != nullptr);
        current = ggml_add(ctx, current, tensors.bias);
        REQUIRE(current != nullptr);
    }
    if (include_residual) {
        tensors.residual_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, output_size, route_count, token_count);
        REQUIRE(tensors.residual_input != nullptr);
        current = ggml_add(ctx, current, tensors.residual_input);
        REQUIRE(current != nullptr);
    }
    tensors.residual_output = current;

    if (include_rmsnorm) {
        REQUIRE(include_residual);
        tensors.norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_size);
        REQUIRE(tensors.norm_weight != nullptr);
        tensors.rms_output = ggml_rms_norm(ctx, tensors.residual_output, 0.000001f);
        REQUIRE(tensors.rms_output != nullptr);
        tensors.normalized_output = ggml_mul(ctx, tensors.rms_output, tensors.norm_weight);
        REQUIRE(tensors.normalized_output != nullptr);
    } else {
        tensors.normalized_output = tensors.residual_output;
    }
    return tensors;
}

static ggml::hrx::GraphImportResult import_common_mul_mat_id_postops_graph(
    ggml_context *                       ctx,
    const CommonMulMatIdPostOpsTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.normalized_output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static void require_common_mul_mat_id_postops_match(ggml_context *                       ctx,
                                                    const CommonMulMatIdPostOpsTensors & tensors,
                                                    const char *                         expected_kernel_name,
                                                    bool                                 expect_bias,
                                                    bool                                 expect_residual,
                                                    bool                                 expect_rmsnorm) {
    ggml::hrx::GraphImportResult imported = import_common_mul_mat_id_postops_graph(ctx, tensors);
    std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
    ggml::hrx::CommandPlan       plan;
    const size_t                 projection_index = producer_index_for_tensor(imported.graph, tensors.projection);
    ggml::hrx::DispatchMatch     match;
    REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, projection_index, match));
    REQUIRE(match.dispatches.size() == 3);
    REQUIRE(match.transients.size() == 2);

    const ggml::hrx::Dispatch & dispatch = match.dispatches.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.projection->ne[2]);
    REQUIRE(dispatch.bindings.size() == (expect_rmsnorm ? 10 : 7));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.input_size", std::to_string(tensors.weight->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.output_size", std::to_string(tensors.weight->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.expert_count", std::to_string(tensors.weight->ne[2]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.route_count",
                              std::to_string(tensors.route_ids->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.input_route_count",
                              std::to_string(tensors.input->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.weight_format",
                              common_mul_mat_weight_format(tensors.weight->type));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.has_bias", expect_bias ? "1" : "0");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.has_residual", expect_residual ? "1" : "0");

    const ggml::hrx::Value * input_value             = imported.graph.values().find_tensor(tensors.input);
    const ggml::hrx::Value * weight_value            = imported.graph.values().find_tensor(tensors.weight);
    const ggml::hrx::Value * route_ids_value         = imported.graph.values().find_tensor(tensors.route_ids);
    const ggml::hrx::Value * residual_output_value   = imported.graph.values().find_tensor(tensors.residual_output);
    const ggml::hrx::Value * normalized_output_value = imported.graph.values().find_tensor(tensors.normalized_output);
    REQUIRE(input_value != nullptr);
    REQUIRE(weight_value != nullptr);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(residual_output_value != nullptr);
    REQUIRE(normalized_output_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        match.metadata.find_moe_routing_bundle(route_ids_value->id);
    REQUIRE(routing_bundle != nullptr);

    REQUIRE(dispatch.bindings[0].value == input_value->id);
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
    REQUIRE(dispatch.bindings[3].value == weight_value->id);
    if (expect_bias) {
        const ggml::hrx::Value * bias_value = imported.graph.values().find_tensor(tensors.bias);
        REQUIRE(bias_value != nullptr);
        REQUIRE(dispatch.bindings[4].value == bias_value->id);
    }
    if (expect_residual) {
        const ggml::hrx::Value * residual_input_value = imported.graph.values().find_tensor(tensors.residual_input);
        REQUIRE(residual_input_value != nullptr);
        REQUIRE(dispatch.bindings[5].value == residual_input_value->id);
    }
    REQUIRE(dispatch.bindings[6].value == residual_output_value->id);

    if (expect_rmsnorm) {
        const ggml::hrx::Value * norm_weight_value = imported.graph.values().find_tensor(tensors.norm_weight);
        REQUIRE(norm_weight_value != nullptr);
        REQUIRE(dispatch.bindings[7].value == norm_weight_value->id);
        REQUIRE(dispatch.bindings[8].value == normalized_output_value->id);
        REQUIRE(match.completion_counter_requests.size() == 1);
        REQUIRE(match.completion_counter_requests.front().count ==
                common_mul_mat_id_partition_descriptor_capacity(tensors));
        REQUIRE(dispatch.bindings[9].value == match.completion_counter_requests.front().value);
        require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.rms_epsilon", "9.99999997e-07");
    } else {
        REQUIRE(match.completion_counter_requests.empty());
    }

    append_match_to_plan(plan, match, covered_nodes, &imported.graph);
    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 3);
    REQUIRE(command_program_verifies(commands));
}

static void run_common_mul_mat_id_postops_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, true, false, false);
        require_common_mul_mat_id_postops_match(ctx, tensors, "loom_libs:ggml_mul_mat_id_postops_f32_f32_wmma", true,
                                                false, false);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, false, true, false);
        require_common_mul_mat_id_postops_match(ctx, tensors, "loom_libs:ggml_mul_mat_id_postops_f32_f32_wmma", false,
                                                true, false);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, true, true, false);
        require_common_mul_mat_id_postops_match(ctx, tensors, "loom_libs:ggml_mul_mat_id_postops_f32_f32_wmma", true,
                                                true, false);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, false, true, true);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, "loom_libs:ggml_mul_mat_id_postops_next_rmsnorm_f32_f32_wmma", false, true, true);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, true, true, true);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, "loom_libs:ggml_mul_mat_id_postops_next_rmsnorm_f32_f32_wmma", true, true, true);
    }

    ggml_free(ctx);
}

static void append_qwen_weighted_reduce_for_graph(ggml::hrx::Graph &              graph,
                                                  const QwenRoutedGateUpTensors & tensors,
                                                  std::vector<bool> &             covered_nodes,
                                                  ggml::hrx::CommandPlan &        plan,
                                                  const char *                    expected_kernel_name) {
    REQUIRE(tensors.weighted != nullptr);
    REQUIRE(tensors.residual != nullptr);
    const size_t             weighted_index = producer_index_for_tensor(graph, tensors.weighted);
    ggml::hrx::DispatchMatch weighted_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, weighted_index, weighted_match));
    append_match_to_plan(plan, weighted_match, covered_nodes, &graph);

    const ggml::hrx::Value * route_weights_value = graph.values().find_tensor(tensors.route_weights);
    const ggml::hrx::Value * routed_output_value = graph.values().find_tensor(tensors.output);
    const ggml::hrx::Value * residual_value      = graph.values().find_tensor(tensors.residual);
    REQUIRE(route_weights_value != nullptr);
    REQUIRE(routed_output_value != nullptr);
    REQUIRE(residual_value != nullptr);

    const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == (tensors.next_output != nullptr ? 5 : 3));
    REQUIRE(dispatch.bindings[0].value == route_weights_value->id);
    REQUIRE(dispatch.bindings[0].length == route_weights_value->byte_count);
    REQUIRE(dispatch.bindings[1].value == plan.metadata.alternate_values().back().alternate_value);
    REQUIRE(dispatch.bindings[1].length == qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[2].value == residual_value->id);
    REQUIRE(dispatch.bindings[2].length == residual_value->byte_count);
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.route_count", "8");
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.output_size", "2048");
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(tensors.output->ne[2]));

    if (tensors.next_output != nullptr) {
        const ggml::hrx::Value * next_output_value = graph.values().find_tensor(tensors.next_output);
        REQUIRE(next_output_value != nullptr);
        REQUIRE(dispatch.bindings[4].value == next_output_value->id);
        REQUIRE(dispatch.bindings[4].length == next_output_value->byte_count);
        require_compile_parameter(dispatch, "qwen3_moe.model.hidden_size", "2048");
        require_compile_parameter(dispatch, "qwen3_moe.model.rms_epsilon", "0.000001");
    }
}

static void run_qwen_routed_gate_up_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        constexpr int64_t            token_count = 4;
        QwenRoutedGateUpTensors      tensors     = build_qwen_routed_gate_up_graph(ctx, token_count, GGML_GLU_OP_GEGLU);
        ggml::hrx::GraphImportResult imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        ggml::hrx::DispatchMatch     gate_match = require_common_mul_mat_id_for_graph(
            imported.graph, plan, covered_nodes, tensors.gate, tensors.input, tensors.gate_weight, tensors.route_ids);
        append_match_to_plan(plan, gate_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 2);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands[3].bindings.size() == 5);
        REQUIRE(commands.commands[3].bindings[0].name == "input");
        REQUIRE(commands.commands[3].bindings[1].name == "expert_table");
        REQUIRE(commands.commands[3].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[3].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[3].name == "weight");
        REQUIRE(commands.commands[3].bindings[4].name == "output");
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        ggml::hrx::DispatchMatch      down_match = require_common_mul_mat_id_for_graph(
            imported.graph, plan, covered_nodes, tensors.output, tensors.glu, tensors.down_weight, tensors.route_ids);
        append_match_to_plan(plan, down_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 2);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands[3].bindings[0].name == "input");
        REQUIRE(commands.commands[3].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[3].bindings[3].name == "weight");
        REQUIRE(commands.commands[3].bindings[4].name == "output");
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        REQUIRE(imported.graph.nodes().size() == 14);

        const ggml::hrx::Value * route_ids_value     = imported.graph.values().find_tensor(tensors.route_ids);
        const ggml::hrx::Value * route_weights_value = imported.graph.values().find_tensor(tensors.route_weights);
        const ggml::hrx::Value * glu_value           = imported.graph.values().find_tensor(tensors.glu);
        REQUIRE(route_ids_value != nullptr);
        REQUIRE(route_weights_value != nullptr);
        REQUIRE(glu_value != nullptr);
        REQUIRE(route_ids_value->kind == ggml::hrx::ValueKind::Transient);
        REQUIRE(route_weights_value->kind == ggml::hrx::ValueKind::External);
        REQUIRE(glu_value->kind == ggml::hrx::ValueKind::Transient);

        std::vector<bool>      covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const ggml::hrx::CommandPlanGeneratedResource * expert_table_resource = plan.metadata.find_generated_resource(
            route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * partition_table_resource =
            plan.metadata.find_generated_resource(route_ids_value->id,
                                                  ggml::hrx::GeneratedResourceRole::MoePartitionTable);
        const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
            plan.metadata.find_moe_routing_bundle(route_ids_value->id);
        REQUIRE(expert_table_resource != nullptr);
        REQUIRE(partition_table_resource != nullptr);
        REQUIRE(routing_bundle != nullptr);
        REQUIRE(routing_bundle->route_ids == route_ids_value->id);
        REQUIRE(routing_bundle->route_weights == route_weights_value->id);
        REQUIRE(routing_bundle->expert_table == expert_table_resource->generated_value);
        REQUIRE(routing_bundle->partition_table == partition_table_resource->generated_value);
        REQUIRE(routing_bundle->expert_table_byte_count == qwen_expert_table_size(token_count));
        REQUIRE(routing_bundle->partition_table_byte_count == qwen_partition_table_size(token_count));
        REQUIRE(routing_bundle->route_count == kQwenRouterRouteCount);
        REQUIRE(routing_bundle->expert_count == kQwenRouterExpertCount);
        ggml::hrx::MoeRoutingResourceMetadata expert_metadata;
        ggml::hrx::MoeRoutingResourceMetadata partition_metadata;
        REQUIRE(expert_table_resource->metadata.read(expert_metadata));
        REQUIRE(partition_table_resource->metadata.read(partition_metadata));
        REQUIRE(expert_metadata.token_count == token_count);
        REQUIRE(expert_metadata.route_count == kQwenRouterRouteCount);
        REQUIRE(expert_metadata.expert_count == kQwenRouterExpertCount);
        REQUIRE(partition_metadata.route_stride == expert_metadata.route_stride);
        REQUIRE(expert_table_resource->byte_count == qwen_expert_table_size(token_count));
        REQUIRE(partition_table_resource->byte_count == qwen_partition_table_size(token_count));

        const size_t             gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 3);
        REQUIRE(plan.metadata.alternate_values().size() == 1);
        const ggml::hrx::CommandPlanTransient & f16_output_transient = plan.transients.back();
        REQUIRE(f16_output_transient.name == "qwen.moe.gate_up_swiglu_f16");
        REQUIRE(f16_output_transient.size == qwen_routed_gate_up_f16_output_size(token_count));
        REQUIRE(plan.metadata.alternate_values().front().graph_value == glu_value->id);
        REQUIRE(plan.metadata.alternate_values().front().alternate_value == f16_output_transient.value);
        REQUIRE(plan.metadata.alternate_values().front().type == GGML_TYPE_F16);
        REQUIRE(plan.metadata.alternate_values().front().byte_count == f16_output_transient.size);

        const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.bindings.size() == 6);
        REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
        REQUIRE(dispatch.bindings[1].length == qwen_expert_table_size(token_count));
        REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
        REQUIRE(dispatch.bindings[2].length == qwen_partition_table_size(token_count));
        REQUIRE(dispatch.bindings[5].value == f16_output_transient.value);
        REQUIRE(dispatch.bindings[5].length == f16_output_transient.size);
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.input_size", "2048");
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.expert_count", "128");
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.route_count", "8");
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.output_size", "768");
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands[3].bindings.size() == 6);
        REQUIRE(commands.commands[3].bindings[0].name == "input");
        REQUIRE(commands.commands[3].bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[1].name == "expert_table");
        REQUIRE(commands.commands[3].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[3].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[3].name == "gate_weight");
        REQUIRE(commands.commands[3].bindings[3].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[4].name == "up_weight");
        REQUIRE(commands.commands[3].bindings[4].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[5].name == "output");
        REQUIRE(commands.commands[3].bindings[5].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, f16_output_transient.value) != nullptr);
    }

    {
        constexpr int64_t             token_count = 1;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 3);
        REQUIRE(kernel_name_for_id(plan.dispatches.back().kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma");
        REQUIRE(plan.dispatches.back().kernel.integer_parameters.at("token_count") == 1);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t token_count     = 4;
        ggml_tensor *     first_logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
        ggml_tensor *     first_route_ids = nullptr;
        REQUIRE(first_logits != nullptr);
        ggml_tensor * first_route_weights = build_qwen_router_top8_graph(ctx, first_logits, &first_route_ids);
        REQUIRE(first_route_ids != nullptr);
        REQUIRE(first_route_weights != nullptr);

        const QwenRoutedGateUpTensors tensors = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml_cgraph *                 graph   = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, first_route_weights);
        ggml_build_forward_expand(graph, tensors.route_weights);
        ggml_build_forward_expand(graph, tensors.output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        const ggml::hrx::Value * first_route_ids_value  = imported.graph.values().find_tensor(first_route_ids);
        const ggml::hrx::Value * second_route_ids_value = imported.graph.values().find_tensor(tensors.route_ids);
        REQUIRE(first_route_ids_value != nullptr);
        REQUIRE(second_route_ids_value != nullptr);
        REQUIRE(first_route_ids_value->id != second_route_ids_value->id);

        std::vector<bool>      covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan plan;
        size_t                 router_matches = 0;
        for (size_t i = 0; i < imported.graph.nodes().size(); ++i) {
            if (imported.graph.nodes()[i].op != GGML_OP_SOFT_MAX) {
                continue;
            }
            ggml::hrx::DispatchMatch router_match;
            REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, i, router_match));
            append_match_to_plan(plan, router_match, covered_nodes);
            ++router_matches;
        }
        REQUIRE(router_matches == 2);
        REQUIRE(plan.metadata.generated_resources().size() == 4);
        REQUIRE(plan.metadata.moe_routing_bundles().size() == 2);

        const ggml::hrx::CommandPlanGeneratedResource * first_expert_table = plan.metadata.find_generated_resource(
            first_route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * second_expert_table = plan.metadata.find_generated_resource(
            second_route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * second_partition_table = plan.metadata.find_generated_resource(
            second_route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoePartitionTable);
        const ggml::hrx::CommandPlanMoeRoutingBundle * second_routing_bundle =
            plan.metadata.find_moe_routing_bundle(second_route_ids_value->id);
        REQUIRE(first_expert_table != nullptr);
        REQUIRE(second_expert_table != nullptr);
        REQUIRE(second_partition_table != nullptr);
        REQUIRE(second_routing_bundle != nullptr);
        REQUIRE(second_routing_bundle->expert_table == second_expert_table->generated_value);
        REQUIRE(second_routing_bundle->partition_table == second_partition_table->generated_value);
        REQUIRE(first_expert_table->generated_value != second_expert_table->generated_value);

        const size_t             gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 7);
        REQUIRE(plan.transients.size() == 5);
        const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
        REQUIRE(dispatch.bindings.size() == 6);
        REQUIRE(dispatch.bindings[1].value == second_routing_bundle->expert_table);
        REQUIRE(dispatch.bindings[1].value != first_expert_table->generated_value);
        REQUIRE(dispatch.bindings[2].value == second_routing_bundle->partition_table);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 7);
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");

        REQUIRE(plan.dispatches.size() == 5);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 5);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, plan.transients.back().value) != nullptr);
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 6);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands.back().bindings.size() == 3);
        REQUIRE(commands.commands.back().bindings[0].name == "route_weights");
        REQUIRE(commands.commands.back().bindings[1].name == "routed_output");
        REQUIRE(commands.commands.back().bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands.back().bindings[2].name == "output");
        REQUIRE(commands.commands.back().bindings[2].access == ggml::hrx::ResourceAccess::ReadWrite);
    }

    {
        constexpr int64_t       token_count = 1;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.dispatches.back().kernel.integer_parameters.at("token_count") == 1);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors, true);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::Value * hidden_state_value = imported.graph.values().find_tensor(tensors.hidden_state);
        const ggml::hrx::Value * residual_value     = imported.graph.values().find_tensor(tensors.residual);
        REQUIRE(hidden_state_value != nullptr);
        REQUIRE(residual_value != nullptr);
        REQUIRE(residual_value->alias_source == hidden_state_value->id);
        REQUIRE(imported.graph.values().same_storage(hidden_state_value->id, residual_value->id));
        REQUIRE(residual_value->storage_root == hidden_state_value->storage_root);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 6);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands.back().bindings.size() == 5);
        REQUIRE(commands.commands.back().bindings[0].name == "route_weights");
        REQUIRE(commands.commands.back().bindings[1].name == "routed_output");
        REQUIRE(commands.commands.back().bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands.back().bindings[2].name == "hidden_state");
        REQUIRE(commands.commands.back().bindings[2].value == hidden_state_value->storage_root);
        REQUIRE(commands.commands.back().bindings[2].access == ggml::hrx::ResourceAccess::ReadWrite);
        REQUIRE(commands.commands.back().bindings[3].name == "next_norm_weight");
        REQUIRE(commands.commands.back().bindings[4].name == "next_projection_input");
        REQUIRE(commands.commands.back().bindings[4].access == ggml::hrx::ResourceAccess::Write);
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors, true);
        ggml_tensor * hidden_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, token_count);
        REQUIRE(hidden_bias != nullptr);
        tensors.hidden_use = ggml_add(ctx, tensors.hidden_state, hidden_bias);
        REQUIRE(tensors.hidden_use != nullptr);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");

        const size_t             weighted_index = producer_index_for_tensor(imported.graph, tensors.weighted);
        ggml::hrx::DispatchMatch weighted_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, weighted_index, weighted_match));
        append_match_to_plan(plan, weighted_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(kernel_name_for_id(plan.dispatches.back().kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");
        REQUIRE(weighted_match.value_aliases.empty());
        REQUIRE(plan.dispatches.back().bindings.size() == 3);
        const ggml::hrx::Value * hidden_state_value = imported.graph.values().find_tensor(tensors.hidden_state);
        const ggml::hrx::Value * residual_value     = imported.graph.values().find_tensor(tensors.residual);
        REQUIRE(hidden_state_value != nullptr);
        REQUIRE(residual_value != nullptr);
        REQUIRE(residual_value->alias_source != hidden_state_value->id);
        REQUIRE(!imported.graph.values().same_storage(hidden_state_value->id, residual_value->id));
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, token_count, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, true, GGML_TYPE_Q4_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");

        REQUIRE(plan.dispatches.size() == 5);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 5);
        REQUIRE(command_program_verifies(commands));
    }

    {
        const QwenRoutedGateUpTensors tensors    = build_qwen_routed_gate_up_graph(ctx, 4);
        ggml::hrx::GraphImportResult  imported   = import_qwen_routed_gate_up_graph(ctx, tensors);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        const ggml::hrx::CommandPlan  empty_plan;
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, empty_plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 3);
        REQUIRE(gate_up_match.transients.size() == 2);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                "loom_libs:ggml_moe_build_expert_table");
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[1].kernel.kernel_id) ==
                "loom_libs:ggml_moe_build_expert_partition_table");
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[2].kernel.kernel_id) ==
                "loom_libs:ggml_mul_mat_id_swiglu_f32_f32_wmma");
        REQUIRE(gate_up_match.dispatches[2].bindings.size() == 6);
        REQUIRE(gate_up_match.dispatches[2].bindings[1].value == gate_up_match.transients[0].value);
        REQUIRE(gate_up_match.dispatches[2].bindings[2].value == gate_up_match.transients[1].value);
        REQUIRE(gate_up_match.metadata.generated_resources().size() == 2);
        REQUIRE(gate_up_match.metadata.moe_routing_bundles().size() == 1);
    }

    {
        const QwenRoutedGateUpTensors tensors  = build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_GEGLU);
        ggml::hrx::GraphImportResult  imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                "loom_libs:ggml_mul_mat_id_f32_f32_wmma");
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q6_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                "loom_libs:ggml_mul_mat_id_swiglu_f32_f32_wmma");
        REQUIRE(gate_up_match.dispatches[0].bindings.size() == 6);
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, false);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                "loom_libs:ggml_mul_mat_id_swiglu_f32_f32_wmma");
        REQUIRE(gate_up_match.dispatches[0].bindings.size() == 6);
    }

    {
        const QwenRoutedGateUpTensors tensors  = build_qwen_routed_gate_up_graph(ctx, 4);
        ggml::hrx::GraphImportResult  imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  down_index = producer_index_for_tensor(imported.graph, tensors.output);
        ggml::hrx::DispatchMatch      down_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, down_index, down_match));
        REQUIRE(down_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(down_match.dispatches[0].kernel.kernel_id) ==
                "loom_libs:ggml_mul_mat_id_f32_f32_wmma");
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, true, GGML_TYPE_Q5_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        const size_t             down_index = producer_index_for_tensor(imported.graph, tensors.output);
        ggml::hrx::DispatchMatch down_match;
        REQUIRE(!match_dispatch_at_index(imported.graph, plan, covered_nodes, down_index, down_match));
    }

    ggml_free(ctx);
}

static void run_qwen_router_top8_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 1);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 13);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids, GGML_SORT_ORDER_DESC,
                                                            kQwenRouterRouteCount, 0.00006103515625f);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 512);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids, 64, kQwenRouterRouteCount);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids, GGML_SORT_ORDER_DESC, 4);
        schedule_qwen_router_top8_command(ctx, output, route_ids, kQwenRouterExpertCount, 4);
    }
    {
        ManualQwenRouterTop8Graph manual = build_manual_qwen_router_top8_graph(ctx, 64, 4, 5);
        schedule_manual_qwen_router_top8_command(manual.graph, manual.route_ids, 5, 64, 4);
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, nullptr, GGML_SORT_ORDER_ASC);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, nullptr, GGML_SORT_ORDER_DESC, 33);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);
        REQUIRE(!graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
}

static void bind_external_values(ggml::hrx::ValueMap & values) {
    uintptr_t buffer = 0x1000;
    for (const ggml::hrx::ValueId id : values.external_value_ids()) {
        const ggml::hrx::Value * value = values.find(id);
        REQUIRE(value != nullptr);
        REQUIRE(values.bind_buffer(id, { dummy_hrx_buffer(buffer), 0, value->byte_count }));
        buffer += 0x1000;
    }
}

static void run_alias_value_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * source = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * view   = ggml_view_1d(ctx, source, 4, 2 * sizeof(float));
    REQUIRE(source != nullptr);
    REQUIRE(view != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, view);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_VIEW);

    const ggml::hrx::Value * source_value = imported.graph.values().find_tensor(source);
    const ggml::hrx::Value * view_value   = imported.graph.values().find_tensor(view);
    REQUIRE(source_value != nullptr);
    REQUIRE(view_value != nullptr);
    REQUIRE(source_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(imported.graph.values().same_storage(source_value->id, view_value->id));
    REQUIRE(view_value->storage_root == source_value->id);
    REQUIRE(view_value->alias_source == source_value->id);
    REQUIRE(view_value->storage_offset == 2 * sizeof(float));
    REQUIRE(view_value->byte_count == 4 * sizeof(float));
    REQUIRE(ggml::hrx::is_layout_alias_node(imported.graph, imported.graph.nodes()[0]));

    REQUIRE(imported.graph.values().bind_buffer(source_value->id,
                                                { dummy_hrx_buffer(0x4000), 128, source_value->byte_count }));
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    const ggml::hrx::CommandProgramBinding * view_binding = bindings.find(view_value->id);
    REQUIRE(view_binding != nullptr);
    REQUIRE(view_binding->buffer == dummy_hrx_buffer(0x4000));
    REQUIRE(view_binding->offset == 128 + 2 * sizeof(float));
    REQUIRE(view_binding->length == view_value->byte_count);

    ggml_tensor * internal_source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * internal_bias   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    ggml_tensor * internal_view =
        ggml_view_2d(ctx, internal_source, 8, 1, internal_source->nb[1], internal_source->nb[1]);
    ggml_tensor * internal_out = ggml_add(ctx, internal_view, internal_bias);
    REQUIRE(internal_source != nullptr);
    REQUIRE(internal_bias != nullptr);
    REQUIRE(internal_view != nullptr);
    REQUIRE(internal_out != nullptr);

    ggml_cgraph * internal_graph = ggml_new_graph(ctx);
    REQUIRE(internal_graph != nullptr);
    ggml_build_forward_expand(internal_graph, internal_out);

    ggml::hrx::GraphImportResult internal_imported = ggml::hrx::import_ggml_graph(*internal_graph);
    REQUIRE(internal_imported.valid());
    REQUIRE(internal_imported.graph.nodes().size() == 2);
    REQUIRE(internal_imported.graph.nodes()[0].op == GGML_OP_VIEW);
    REQUIRE(internal_imported.graph.nodes()[1].op == GGML_OP_ADD);

    const ggml::hrx::Value * internal_source_value = internal_imported.graph.values().find_tensor(internal_source);
    const ggml::hrx::Value * internal_view_value   = internal_imported.graph.values().find_tensor(internal_view);
    const ggml::hrx::Value * internal_bias_value   = internal_imported.graph.values().find_tensor(internal_bias);
    const ggml::hrx::Value * internal_out_value    = internal_imported.graph.values().find_tensor(internal_out);
    REQUIRE(internal_source_value != nullptr);
    REQUIRE(internal_view_value != nullptr);
    REQUIRE(internal_bias_value != nullptr);
    REQUIRE(internal_out_value != nullptr);
    REQUIRE(internal_source_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(internal_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(internal_view_value->storage_root == internal_source_value->id);
    REQUIRE(internal_view_value->storage_offset == internal_source->nb[1]);

    ggml::hrx::DispatchScheduler internal_scheduler;
    REQUIRE(!internal_scheduler.schedule_graph(internal_imported.graph, test_dispatch_target()));

    REQUIRE(internal_imported.graph.values().bind_buffer(
        internal_source_value->id, { dummy_hrx_buffer(0x5000), 256, internal_source_value->byte_count }));
    REQUIRE(internal_imported.graph.values().bind_buffer(
        internal_bias_value->id, { dummy_hrx_buffer(0x6000), 0, internal_bias_value->byte_count }));
    REQUIRE(internal_imported.graph.values().bind_buffer(
        internal_out_value->id, { dummy_hrx_buffer(0x7000), 0, internal_out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings internal_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(internal_imported.graph.values());
    REQUIRE(internal_bindings.valid());
    const ggml::hrx::CommandProgramBinding * internal_view_binding = internal_bindings.find(internal_view_value->id);
    REQUIRE(internal_view_binding != nullptr);
    REQUIRE(internal_view_binding->buffer == dummy_hrx_buffer(0x5000));
    REQUIRE(internal_view_binding->offset == 256 + internal_source->nb[1]);
    REQUIRE(internal_view_binding->length == internal_view_value->byte_count);

    ggml_tensor * cache       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * rows        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * row_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    ggml_tensor * updated     = ggml_set_rows(ctx, cache, rows, row_indices);
    REQUIRE(cache != nullptr);
    REQUIRE(rows != nullptr);
    REQUIRE(row_indices != nullptr);
    REQUIRE(updated != nullptr);
    ggml_tensor * updated_view = ggml_view_2d(ctx, updated, 8, 2, updated->nb[1], 0);
    ggml_tensor * updated_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * updated_out  = ggml_add(ctx, updated_view, updated_bias);
    REQUIRE(updated_view != nullptr);
    REQUIRE(updated_bias != nullptr);
    REQUIRE(updated_out != nullptr);

    ggml_cgraph * set_rows_graph = ggml_new_graph(ctx);
    REQUIRE(set_rows_graph != nullptr);
    ggml_build_forward_expand(set_rows_graph, updated_out);

    ggml::hrx::GraphImportResult set_rows_imported = ggml::hrx::import_ggml_graph(*set_rows_graph);
    REQUIRE(set_rows_imported.valid());
    REQUIRE(set_rows_imported.graph.nodes().size() == 3);
    REQUIRE(set_rows_imported.graph.nodes()[0].op == GGML_OP_SET_ROWS);
    REQUIRE(set_rows_imported.graph.nodes()[1].op == GGML_OP_VIEW);
    REQUIRE(set_rows_imported.graph.nodes()[2].op == GGML_OP_ADD);

    const ggml::hrx::Value * cache_value        = set_rows_imported.graph.values().find_tensor(cache);
    const ggml::hrx::Value * updated_value      = set_rows_imported.graph.values().find_tensor(updated);
    const ggml::hrx::Value * updated_view_value = set_rows_imported.graph.values().find_tensor(updated_view);
    REQUIRE(cache_value != nullptr);
    REQUIRE(updated_value != nullptr);
    REQUIRE(updated_view_value != nullptr);
    REQUIRE(cache_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(updated_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(updated_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(updated_value->storage_root == cache_value->id);
    REQUIRE(updated_view_value->storage_root == cache_value->id);
    REQUIRE(updated_view_value->alias_source == cache_value->id);

    ggml::hrx::ValueMap                      value_map;
    const std::array<int64_t, GGML_MAX_DIMS> ne = { 8, 1, 1, 1 };
    const std::array<size_t, GGML_MAX_DIMS>  nb = { sizeof(float), 8 * sizeof(float), 8 * sizeof(float),
                                                    8 * sizeof(float) };
    REQUIRE(value_map.add_snapshot_storage({ ggml::hrx::ValueStorageId(0), ggml::hrx::ValueId(0), 8 * sizeof(float) })
                .success());
    REQUIRE(
        value_map
            .add_snapshot_value({ ggml::hrx::ValueId(0), ggml::hrx::ValueKind::External, ggml::hrx::ValueStorageId(0),
                                  ggml::hrx::ValueId(0), ggml::hrx::ValueId(), 0, 8 * sizeof(float), GGML_TYPE_F32, ne,
                                  nb, 8, 8 * sizeof(float), true, nullptr, std::nullopt })
            .success());
    REQUIRE(value_map.add_snapshot_storage({ ggml::hrx::ValueStorageId(1), ggml::hrx::ValueId(1), 8 * sizeof(float) })
                .success());
    REQUIRE(
        value_map
            .add_snapshot_value({ ggml::hrx::ValueId(1), ggml::hrx::ValueKind::Transient, ggml::hrx::ValueStorageId(1),
                                  ggml::hrx::ValueId(1), ggml::hrx::ValueId(), 0, 8 * sizeof(float), GGML_TYPE_F32, ne,
                                  nb, 8, 8 * sizeof(float), true, nullptr, std::nullopt })
            .success());
    REQUIRE(value_map.alias_storage(ggml::hrx::ValueId(1), ggml::hrx::ValueId(0)).success());
    const ggml::hrx::Value * aliased_value = value_map.find(ggml::hrx::ValueId(1));
    REQUIRE(aliased_value != nullptr);
    REQUIRE(aliased_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(aliased_value->alias_source == ggml::hrx::ValueId(0));
    REQUIRE(aliased_value->storage_root == ggml::hrx::ValueId(0));
    REQUIRE(value_map.same_storage(ggml::hrx::ValueId(0), ggml::hrx::ValueId(1)));
    REQUIRE(value_map.bind_buffer(ggml::hrx::ValueId(0), { dummy_hrx_buffer(0x8000), 64, 8 * sizeof(float) }));
    const std::optional<ggml::hrx::ValueBufferBinding> aliased_binding =
        value_map.resolve_buffer_binding(ggml::hrx::ValueId(1));
    REQUIRE(aliased_binding.has_value());
    REQUIRE(aliased_binding->buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(aliased_binding->offset == 64);
    REQUIRE(aliased_binding->length == 8 * sizeof(float));

    ggml_free(ctx);
}

static void run_multi_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    ggml_tensor * out1 = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);
    graph->uid = 1001;

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    const ggml::hrx::Value * out0_value = imported.graph.values().find_tensor(out0);
    const ggml::hrx::Value * out1_value = imported.graph.values().find_tensor(out1);
    REQUIRE(out0_value != nullptr);
    REQUIRE(out1_value != nullptr);
    REQUIRE(imported.graph.nodes()[0].output == out0_value->id);
    REQUIRE(imported.graph.nodes()[1].output == out1_value->id);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands[0].ordinal == 0);
    REQUIRE(commands.commands[0].dependencies.empty());
    REQUIRE(commands.commands[1].ordinal == 1);
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(command_program_verifies(commands));

    bind_external_values(imported.graph.values());
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    ggml::hrx::ResolvedCommandProgram resolved = ggml::hrx::resolve_command_program_bindings(commands, bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.size() == 2);

    ggml_free(ctx);
}

static void run_layout_alias_scheduler_elision_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum      = ggml_add(ctx, a, b);
    ggml_tensor * reshaped = ggml_reshape_2d(ctx, sum, 4, 2);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(reshaped != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, reshaped);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_RESHAPE);

    const ggml::hrx::Value * sum_value      = imported.graph.values().find_tensor(sum);
    const ggml::hrx::Value * reshaped_value = imported.graph.values().find_tensor(reshaped);
    REQUIRE(sum_value != nullptr);
    REQUIRE(reshaped_value != nullptr);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(reshaped_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(imported.graph.values().same_storage(sum_value->id, reshaped_value->id));
    REQUIRE(reshaped_value->storage_root == sum_value->id);
    REQUIRE(reshaped_value->alias_source == sum_value->id);
    REQUIRE(ggml::hrx::is_layout_alias_node(imported.graph, imported.graph.nodes()[1]));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[2].value == sum_value->id);
    REQUIRE(commands.commands[0].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.transients.allocations.size() == 1);
    REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, sum_value->id) != nullptr);
    REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, reshaped_value->id) == nullptr);
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_transient_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_sin(ctx, sum);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_SIN);

    const ggml::hrx::Value * a_value   = imported.graph.values().find_tensor(a);
    const ggml::hrx::Value * sum_value = imported.graph.values().find_tensor(sum);
    const ggml::hrx::Value * out_value = imported.graph.values().find_tensor(out);
    REQUIRE(a_value != nullptr);
    REQUIRE(sum_value != nullptr);
    REQUIRE(out_value != nullptr);
    REQUIRE(a_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(out_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(
        !imported.graph.values().bind_buffer(sum_value->id, { dummy_hrx_buffer(0x3000), 0, sum_value->byte_count }));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(!scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(!scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.empty());
    REQUIRE(status_contains(scheduler.plan().status, "unsupported HRX node 1"));

    ggml_free(ctx);
}

static void run_chained_dispatch_requires_transients() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_add(ctx, sum, c);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_ADD);

    const ggml::hrx::Value * sum_value = imported.graph.values().find_tensor(sum);
    REQUIRE(sum_value != nullptr);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[1].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[0].bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[0].bindings[2].value == sum_value->id);
    REQUIRE(commands.commands[0].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[0].value == sum_value->id);
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[1].bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.transients.allocations.size() == 1);
    const ggml::hrx::TransientAllocation * sum_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum_value->id);
    REQUIRE(sum_allocation != nullptr);
    REQUIRE(sum_allocation->value == sum_value->id);
    REQUIRE(sum_allocation->size == sum_value->byte_count);
    REQUIRE(sum_allocation->alignment == 256);
    REQUIRE(sum_allocation->arena_offset == 0);
    REQUIRE(commands.transients.arena_size == 256);
    REQUIRE(command_program_verifies(commands));

    const std::string transient_binding_text = ggml::hrx::format_command_binding(commands.commands[1].bindings[0]);
    REQUIRE(string_contains(transient_binding_text, "origin=Transient"));

    bind_external_values(imported.graph.values());
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    REQUIRE(bindings.find(sum_value->id) == nullptr);

    const ggml::hrx::ResolvedCommandProgram resolved = ggml::hrx::resolve_command_program_bindings(commands, bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "no transient arena"));
    REQUIRE(status_contains(resolved.status, "origin=Transient"));
    REQUIRE(status_contains(resolved.status, "value="));

    const ggml::hrx::TransientArenaAllocationRef transient_arena = {
        dummy_hrx_buffer(0x8000),
        commands.transients.arena_size,
        7,
    };
    const ggml::hrx::ResolvedCommandProgram resolved_with_transients =
        ggml::hrx::resolve_command_program_bindings(commands, bindings, &transient_arena);
    REQUIRE(resolved_with_transients.valid());
    REQUIRE(resolved_with_transients.commands.size() == 2);
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.offset == 0);
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.length == sum_value->byte_count);
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.offset == 0);
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.length == sum_value->byte_count);

    ggml::hrx::PreparedCommandProgram prepared_shape;
    for (const ggml::hrx::Command & prepared_source : commands.commands) {
        ggml::hrx::PreparedCommand prepared_command;
        prepared_command.ordinal               = prepared_source.ordinal;
        prepared_command.kind                  = prepared_source.kind;
        prepared_command.kernel.specialization = prepared_source.kernel;
        for (const ggml::hrx::CommandBinding & binding : prepared_source.bindings) {
            prepared_command.kernel.bindings.push_back({
                binding, { dummy_hrx_buffer(0x4000), 123, binding.length }
            });
        }
        prepared_shape.commands.push_back(prepared_command);
    }
    prepared_shape.bound_transient_arena_allocation_id = 1;

    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, transient_arena, prepared_shape));
    REQUIRE(prepared_shape.bound_transient_arena_allocation_id == transient_arena.allocation_id);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.offset == 0);
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.offset == 0);

    const ggml::hrx::TransientArenaAllocationRef grown_transient_arena = {
        dummy_hrx_buffer(0x9000),
        commands.transients.arena_size + 256,
        8,
    };
    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, grown_transient_arena, prepared_shape));
    REQUIRE(prepared_shape.bound_transient_arena_allocation_id == grown_transient_arena.allocation_id);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0x9000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x9000));

    prepared_shape.commands[1].kernel.bindings[0].binding.origin = ggml::hrx::CommandBindingOrigin::ProgramConstant;
    prepared_shape.commands[1].kernel.bindings[0].ref = { dummy_hrx_buffer(0xb000), 32, sum_value->byte_count };
    const ggml::hrx::TransientArenaAllocationRef rebinding_transient_arena = {
        dummy_hrx_buffer(0xc000),
        commands.transients.arena_size + 512,
        9,
    };
    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, rebinding_transient_arena, prepared_shape));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0xc000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0xb000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.offset == 32);

    const ggml::hrx::TransientArenaAllocationRef invalid_transient_arena = {
        dummy_hrx_buffer(0xa000),
        commands.transients.arena_size,
        ggml::hrx::kInvalidTransientArenaAllocationId,
    };
    const ggml::hrx::ResolvedCommandProgram invalid_transient_resolved =
        ggml::hrx::resolve_command_program_bindings(commands, bindings, &invalid_transient_arena);
    REQUIRE(!invalid_transient_resolved.valid());
    REQUIRE(status_contains(invalid_transient_resolved.status, "no transient arena allocation id"));

    ggml::hrx::CommandProgram missing_allocation = copy_command_program_shape(commands);
    missing_allocation.transients.allocations.clear();
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(missing_allocation, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "no transient allocation"));

    ggml::hrx::CommandProgram out_of_range      = copy_command_program_shape(commands);
    out_of_range.commands[0].bindings[2].length = sum_value->byte_count + 1;
    verification = ggml::hrx::verify_command_program(out_of_range, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "outside transient allocation length"));

    ggml_free(ctx);
}

static void run_graph_replay_host_staging_is_not_ineligible() {
    ggml::hrx::CommandProgram commands;

    std::vector<uint8_t> source0(64, 1);
    std::vector<uint8_t> source1(64, 2);

    ggml::hrx::PreparedCommandProgram prepared;
    ggml::hrx::HostStagingBuffer      staging;
    staging.buffer    = dummy_hrx_buffer(0x1000);
    staging.host_data = source0.data();
    staging.value     = 7;
    staging.length    = source0.size();
    staging.upload    = true;
    prepared.host_staging.push_back(std::move(staging));

    ggml::hrx::CommandProgramBinding live_binding;
    live_binding.value     = ggml::hrx::ValueId(7);
    live_binding.length    = source1.size();
    live_binding.capacity  = source1.size();
    live_binding.host_data = source1.data();
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_bindings({ live_binding });
    REQUIRE(bindings.valid());

    ggml::hrx::RecordedCommandGraph recorded;
    recorded.exec                                = dummy_hrx_graph_exec(0x2000);
    recorded.bound_transient_arena_allocation_id = ggml::hrx::kInvalidTransientArenaAllocationId;

    ggml::hrx::CommandProgramExecutionContext context;
    context.stream = dummy_hrx_stream(0x3000);

    const ggml::hrx::RecordedCommandGraphExecutionResult result =
        ggml::hrx::bind_and_launch_recorded_command_graph(context, commands, bindings, prepared, recorded);
    recorded.exec = nullptr;

    REQUIRE(!result.success);
    REQUIRE(result.event == ggml::hrx::HrxGraphReplayEvent::LaunchFailed);
    REQUIRE(result.ineligible_reason.empty());
    REQUIRE(status_contains(result.status, "missing HRX host transfer manager"));
    REQUIRE(prepared.host_staging.size() == 1);
    REQUIRE(prepared.host_staging[0].buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(prepared.host_staging[0].host_data == source1.data());
    prepared.host_staging[0].buffer = nullptr;
}

static void run_multiple_transient_plan_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum0 = ggml_add(ctx, a, b);
    ggml_tensor * sum1 = ggml_add(ctx, c, d);
    ggml_tensor * out  = ggml_add(ctx, sum0, sum1);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(sum0 != nullptr);
    REQUIRE(sum1 != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);

    const ggml::hrx::Value * sum0_value = imported.graph.values().find_tensor(sum0);
    const ggml::hrx::Value * sum1_value = imported.graph.values().find_tensor(sum1);
    REQUIRE(sum0_value != nullptr);
    REQUIRE(sum1_value != nullptr);
    REQUIRE(sum0_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(sum1_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 3);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.transients.allocations.size() == 2);
    REQUIRE(commands.transients.arena_size == 512);
    const ggml::hrx::TransientAllocation * sum0_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum0_value->id);
    const ggml::hrx::TransientAllocation * sum1_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum1_value->id);
    REQUIRE(sum0_allocation != nullptr);
    REQUIRE(sum1_allocation != nullptr);
    REQUIRE(sum0_allocation->arena_offset != sum1_allocation->arena_offset);
    REQUIRE(sum0_allocation->arena_offset % 256 == 0);
    REQUIRE(sum1_allocation->arena_offset % 256 == 0);

    ggml::hrx::CommandProgram        overlapping_live_transients = copy_command_program_shape(commands);
    ggml::hrx::TransientAllocation * overlapping_sum0            = nullptr;
    ggml::hrx::TransientAllocation * overlapping_sum1            = nullptr;
    for (ggml::hrx::TransientAllocation & allocation : overlapping_live_transients.transients.allocations) {
        if (allocation.value == sum0_value->id) {
            overlapping_sum0 = &allocation;
        } else if (allocation.value == sum1_value->id) {
            overlapping_sum1 = &allocation;
        }
    }
    REQUIRE(overlapping_sum0 != nullptr);
    REQUIRE(overlapping_sum1 != nullptr);
    overlapping_sum1->arena_offset = overlapping_sum0->arena_offset;
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(overlapping_live_transients, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "transient allocations overlap"));

    ggml_free(ctx);
}

static void run_disjoint_transient_plan_packing_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * e    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * f    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum0 = ggml_add(ctx, a, b);
    ggml_tensor * out0 = ggml_add(ctx, sum0, c);
    ggml_tensor * sum1 = ggml_add(ctx, d, e);
    ggml_tensor * out1 = ggml_add(ctx, sum1, f);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(e != nullptr);
    REQUIRE(f != nullptr);
    REQUIRE(sum0 != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(sum1 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 4);

    const ggml::hrx::Value * sum0_value = imported.graph.values().find_tensor(sum0);
    const ggml::hrx::Value * sum1_value = imported.graph.values().find_tensor(sum1);
    REQUIRE(sum0_value != nullptr);
    REQUIRE(sum1_value != nullptr);
    REQUIRE(sum0_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(sum1_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 4);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.transients.allocations.size() == 2);
    REQUIRE(commands.transients.arena_size == 256);
    const ggml::hrx::TransientAllocation * sum0_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum0_value->id);
    const ggml::hrx::TransientAllocation * sum1_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum1_value->id);
    REQUIRE(sum0_allocation != nullptr);
    REQUIRE(sum1_allocation != nullptr);
    REQUIRE(sum0_allocation->arena_offset == sum1_allocation->arena_offset);
    REQUIRE(sum0_allocation->arena_offset % 256 == 0);
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_graph_program_cache_uid_mismatch_checks() {
    ggml::hrx::GraphProgramCache    cache;
    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    ggml_tensor * out1 = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph0 = ggml_new_graph(ctx);
    REQUIRE(graph0 != nullptr);
    ggml_build_forward_expand(graph0, out0);
    graph0->uid = 3001;

    ggml::hrx::GraphProgramLookup lookup = cache.get_or_build(*graph0, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    ggml::hrx::GraphProgramCacheStats stats = cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 0);

    lookup = cache.get_or_build(*graph0, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 1);

    ggml_cgraph * graph1 = ggml_new_graph(ctx);
    REQUIRE(graph1 != nullptr);
    ggml_build_forward_expand(graph1, out0);
    ggml_build_forward_expand(graph1, out1);
    graph1->uid = 3001;

    lookup = cache.get_or_build(*graph1, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 2);
    REQUIRE(stats.hits == 1);

    ggml_tensor * unsupported = ggml_sin(ctx, a);
    REQUIRE(unsupported != nullptr);
    ggml_cgraph * graph2 = ggml_new_graph(ctx);
    REQUIRE(graph2 != nullptr);
    ggml_build_forward_expand(graph2, unsupported);
    graph2->uid = 3001;

    lookup = cache.get_or_build(*graph2, corpus, "gfx1151");
    REQUIRE(!lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 2);
    REQUIRE(stats.hits == 1);

    ggml::hrx::GraphProgramCache alias_cache;
    ggml_tensor *                alias_sum = ggml_add(ctx, a, b);
    ggml_tensor *                view0     = ggml_view_1d(ctx, alias_sum, 4, 0);
    ggml_tensor *                view1     = ggml_view_1d(ctx, alias_sum, 4, sizeof(float));
    REQUIRE(alias_sum != nullptr);
    REQUIRE(view0 != nullptr);
    REQUIRE(view1 != nullptr);

    ggml_cgraph * alias_graph0 = ggml_new_graph(ctx);
    REQUIRE(alias_graph0 != nullptr);
    ggml_build_forward_expand(alias_graph0, view0);
    alias_graph0->uid = 3002;
    lookup            = alias_cache.get_or_build(*alias_graph0, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = alias_cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 0);

    ggml_cgraph * alias_graph1 = ggml_new_graph(ctx);
    REQUIRE(alias_graph1 != nullptr);
    ggml_build_forward_expand(alias_graph1, view1);
    alias_graph1->uid = 3002;
    lookup            = alias_cache.get_or_build(*alias_graph1, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = alias_cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 1);

    ggml_free(ctx);
}

static void run_graph_executor_contract_checks() {
    ggml_backend_hrx_device_context device_context  = {};
    ggml_backend_hrx_context        backend_context = {};
    device_context.architecture                     = "gfx1151";
    backend_context.device                          = &device_context;
    const ggml::hrx::GraphExecutor executor(backend_context);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * add_out = ggml_add(ctx, a, b);
    ggml_tensor * sin_out = ggml_sin(ctx, a);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(add_out != nullptr);
    REQUIRE(sin_out != nullptr);

    ggml_cgraph * add_graph = ggml_new_graph(ctx);
    REQUIRE(add_graph != nullptr);
    ggml_build_forward_expand(add_graph, add_out);
    const ggml::hrx::GraphSupportResult add_support = executor.can_execute(*add_graph);
    REQUIRE(add_support.supported);
    REQUIRE(add_support.status.success());

    const ggml::hrx::GraphExecutionResult missing_binding = executor.execute(*add_graph);
    REQUIRE(!missing_binding.success());
    REQUIRE(missing_binding.code == GGML_STATUS_FAILED);
    REQUIRE(status_contains(missing_binding.status, "external value"));
    REQUIRE(status_contains(missing_binding.status, "not bound"));

    ggml_cgraph * sin_graph = ggml_new_graph(ctx);
    REQUIRE(sin_graph != nullptr);
    ggml_build_forward_expand(sin_graph, sin_out);
    const ggml::hrx::GraphSupportResult sin_support = executor.can_execute(*sin_graph);
    REQUIRE(!sin_support.supported);
    REQUIRE(status_contains(sin_support.status, "unsupported HRX node 0"));
    REQUIRE(status_contains(sin_support.status, "SIN"));

    ggml_free(ctx);
}

static void build_add_graph_program(ggml::hrx::GraphProgramCache & cache, uint64_t uid, bool two_outputs) {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out0 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    if (two_outputs) {
        ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * out1 = ggml_add(ctx, c, d);
        REQUIRE(c != nullptr);
        REQUIRE(d != nullptr);
        REQUIRE(out1 != nullptr);
        ggml_build_forward_expand(graph, out1);
    }
    graph->uid = uid;

    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();
    ggml::hrx::GraphProgramLookup   lookup = cache.get_or_build(*graph, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    ggml_free(ctx);
}

static void run_command_program_kernel_dump_checks() {
    static constexpr const char * kDumpEnv = "GGML_HRX_DUMP_COMMAND_PROGRAM_DIR";
    EnvironmentVariableGuard      env(kDumpEnv);

    {
        const std::filesystem::path dump_dir = fresh_test_directory("kernel-dump-disabled");
        env.unset();
        ggml::hrx::GraphProgramCache cache;
        build_add_graph_program(cache, 5001, false);
        REQUIRE(list_directories(dump_dir).empty());
        std::filesystem::remove_all(dump_dir);
    }

    const std::filesystem::path dump_dir = fresh_test_directory("kernel-dump-enabled");
    env.set(dump_dir);

    ggml::hrx::GraphProgramCache cache;
    build_add_graph_program(cache, 5002, false);
    std::vector<std::filesystem::path> dumps = list_directories(dump_dir);
    REQUIRE(dumps.size() == 1);
    REQUIRE(std::filesystem::exists(dumps[0] / "kernels.txt"));
    REQUIRE(std::filesystem::exists(dumps[0] / "kernels.dot"));

    const std::string add_kernels = read_text_file(dumps[0] / "kernels.txt");
    const std::string add_dot     = read_text_file(dumps[0] / "kernels.dot");
    REQUIRE(string_contains(add_kernels, "main 0 loom_libs:ggml_binary_f32"));
    REQUIRE(!string_contains(add_kernels, "binding"));
    REQUIRE(!string_contains(add_kernels, "transient"));
    REQUIRE(string_contains(add_dot, "digraph hrx_kernel_invocations"));
    REQUIRE(string_contains(add_dot, "main 0\\nloom_libs:ggml_binary_f32"));

    build_add_graph_program(cache, 5003, false);
    dumps = list_directories(dump_dir);
    REQUIRE(dumps.size() == 1);

    const std::filesystem::path second_dump_dir = fresh_test_directory("kernel-dump-second-directory");
    env.set(second_dump_dir);
    build_add_graph_program(cache, 5005, false);
    std::vector<std::filesystem::path> second_dumps = list_directories(second_dump_dir);
    REQUIRE(second_dumps.size() == 1);
    REQUIRE(std::filesystem::exists(second_dumps[0] / "kernels.txt"));
    std::filesystem::remove_all(second_dump_dir);
    env.set(dump_dir);

    build_add_graph_program(cache, 5004, true);
    dumps = list_directories(dump_dir);
    REQUIRE(dumps.size() == 2);

    const std::string two_add_kernels = read_text_file(dumps[1] / "kernels.txt");
    const std::string two_add_dot     = read_text_file(dumps[1] / "kernels.dot");
    REQUIRE(string_contains(two_add_kernels, "main 0 loom_libs:ggml_binary_f32"));
    REQUIRE(string_contains(two_add_kernels, "main 1 loom_libs:ggml_binary_f32 deps=0"));
    REQUIRE(string_contains(two_add_dot, "main0 -> main1"));
    REQUIRE(!std::filesystem::exists(dumps[1] / "commands.txt"));
    REQUIRE(!std::filesystem::exists(dumps[1] / "commands.json"));
    REQUIRE(!std::filesystem::exists(dumps[1] / "manifest.json"));

    std::filesystem::remove_all(dump_dir);
}

static std::vector<int32_t> read_transient_i32(ggml_backend_hrx_context *             context,
                                               ggml::hrx::TransientArenaAllocationRef arena,
                                               const ggml::hrx::TransientAllocation & allocation) {
    REQUIRE(context != nullptr);
    REQUIRE(arena.buffer != nullptr);
    REQUIRE(allocation.size % sizeof(int32_t) == 0);
    std::vector<int32_t> data(allocation.size / sizeof(int32_t));
    require_hrx_status(hrx_synchronous_d2h(context->device->device, arena.buffer, allocation.arena_offset, data.data(),
                                           allocation.size));
    return data;
}

static void write_transient_i32_value(ggml_backend_hrx_context *             context,
                                      ggml::hrx::TransientArenaAllocationRef arena,
                                      const ggml::hrx::TransientAllocation & allocation,
                                      int32_t                                value) {
    REQUIRE(context != nullptr);
    REQUIRE(arena.buffer != nullptr);
    REQUIRE(allocation.size >= sizeof(value));
    require_hrx_status(
        hrx_synchronous_h2d(context->device->device, &value, arena.buffer, allocation.arena_offset, sizeof(value)));
}

static void run_qwen_expert_table_partition_prefill_512_execution() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    auto * backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    REQUIRE(backend_context != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t token_count  = 512;
    constexpr int64_t route_count  = 8;
    constexpr int64_t route_stride = 8;
    constexpr int64_t expert_count = 128;

    ggml_tensor * route_ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_stride, token_count);
    REQUIRE(route_ids_tensor != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<int32_t> route_ids =
        make_qwen_route_ids_iota(token_count, route_count, route_stride, expert_count);
    const std::vector<int32_t> expected_expert_table =
        make_qwen_expert_table_reference(route_ids, token_count, route_count, route_stride, expert_count);
    const std::vector<int32_t> expected_partition_table =
        make_qwen_partition_table_reference(expected_expert_table, token_count, route_count, expert_count);
    ggml_backend_tensor_set(route_ids_tensor, route_ids.data(), 0, route_ids.size() * sizeof(int32_t));
    ggml_backend_synchronize(backend);

    ggml::hrx::Graph         graph;
    const ggml::hrx::ValueId route_ids_value =
        graph.values().get_or_add_tensor_value(route_ids_tensor, ggml::hrx::ValueKind::External);
    ggml::hrx::ValueBufferBinding route_ids_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(route_ids_tensor, route_ids_binding));
    REQUIRE(graph.values().bind_buffer(route_ids_value, route_ids_binding));

    const ggml::hrx::ValueId expert_table_value(static_cast<int32_t>(graph.values().size()));
    const ggml::hrx::ValueId partition_table_value(expert_table_value.value + 1);
    const ggml::hrx::ValueId completion_counter_value(expert_table_value.value + 2);
    ggml::hrx::CommandPlan   plan;
    plan.transients.push_back(
        { expert_table_value, "qwen.test.expert_table", qwen_expert_table_size(token_count, expert_count), 256 });
    plan.transients.push_back({ partition_table_value, "qwen.test.partition_table",
                                qwen_partition_table_size(token_count, route_count, expert_count), 256 });
    plan.completion_counter_requests.push_back({ completion_counter_value, "qwen.test.completion_counter", 1 });
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel = ggml::hrx::make_kernel_specialization(
        ggml::hrx::kernel_catalog_ref("qwen3_moe", "qwen3_moe_build_expert_table_partition_prefill_512"));
    dispatch.kernel.integer_parameters.emplace("token_count", token_count);
    dispatch.kernel.integer_parameters.emplace("route_count", route_count);
    dispatch.kernel.integer_parameters.emplace("route_stride", route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", expert_count);
    dispatch.bindings.push_back({ route_ids_value, 0, route_ids.size() * sizeof(int32_t) });
    dispatch.bindings.push_back({ expert_table_value, 0, qwen_expert_table_size(token_count, expert_count) });
    dispatch.bindings.push_back(
        { partition_table_value, 0, qwen_partition_table_size(token_count, route_count, expert_count) });
    dispatch.bindings.push_back({ completion_counter_value, 0, sizeof(int32_t) });
    plan.dispatches.push_back(std::move(dispatch));

    const ggml::hrx::KernelCorpus & corpus   = ggml::hrx::get_qwen_kernel_corpus();
    const char *                    target   = backend_context->device->architecture.c_str();
    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(graph, plan, corpus, target);
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(commands.completion_counters.count == 1);
    REQUIRE(commands.completion_counters.byte_count == sizeof(int32_t));
    REQUIRE(commands.transients.allocations.size() == 3);
    REQUIRE(ggml::hrx::verify_command_program(commands, corpus, target).valid());

    const ggml::hrx::TransientAllocation * expert_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, expert_table_value);
    const ggml::hrx::TransientAllocation * partition_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, partition_table_value);
    const ggml::hrx::TransientAllocation * completion_counter_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, completion_counter_value);
    REQUIRE(expert_table_allocation != nullptr);
    REQUIRE(partition_table_allocation != nullptr);
    REQUIRE(completion_counter_allocation != nullptr);
    REQUIRE(completion_counter_allocation->arena_offset == commands.completion_counters.arena_offset);

    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(graph.values());
    REQUIRE(bindings.valid());
    const ggml::hrx::CommandProgramExecutionContext execution_context = {
        backend_context->device->device,
        backend_context->stream,
        target,
        &corpus,
        &backend_context->kernel_executables,
        &backend_context->transient_arena,
        &backend_context->host_transfers,
        &backend_context->host_weights,
    };

    REQUIRE(ggml::hrx::execute_command_program(execution_context, commands, bindings));
    ggml_backend_synchronize(backend);
    ggml::hrx::TransientArenaAllocationRef arena = backend_context->transient_arena.current_allocation();
    require_qwen_expert_table_matches(read_transient_i32(backend_context, arena, *expert_table_allocation),
                                      expected_expert_table, token_count, expert_count);
    require_qwen_partition_table_matches(read_transient_i32(backend_context, arena, *partition_table_allocation),
                                         expected_partition_table);
    REQUIRE(read_transient_i32(backend_context, arena, *completion_counter_allocation)[0] == 0);

    write_transient_i32_value(backend_context, arena, *completion_counter_allocation, 17);
    REQUIRE(ggml::hrx::execute_command_program(execution_context, commands, bindings));
    ggml_backend_synchronize(backend);
    arena = backend_context->transient_arena.current_allocation();
    require_qwen_expert_table_matches(read_transient_i32(backend_context, arena, *expert_table_allocation),
                                      expected_expert_table, token_count, expert_count);
    require_qwen_partition_table_matches(read_transient_i32(backend_context, arena, *partition_table_allocation),
                                         expected_partition_table);
    REQUIRE(read_transient_i32(backend_context, arena, *completion_counter_allocation)[0] == 0);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_add_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     a             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out           = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        expected[i] = a_data[i] + b_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_two_independent_add_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     a             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     c             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     d             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out0          = ggml_add(ctx, a, b);
    ggml_tensor *     out1          = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);
    graph->uid = 1002;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> c_data(element_count);
    std::vector<float> d_data(element_count);
    std::vector<float> expected0(element_count);
    std::vector<float> expected1(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]    = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]    = static_cast<float>(i % 13) * -0.5f + 3.0f;
        c_data[i]    = static_cast<float>(i % 19) * 0.125f + 1.0f;
        d_data[i]    = static_cast<float>(i % 11) * 0.75f - 4.0f;
        expected0[i] = a_data[i] + b_data[i];
        expected1[i] = c_data[i] + d_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));
    ggml_backend_tensor_set(d, d_data.data(), 0, d_data.size() * sizeof(float));

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 0);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 0);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    std::vector<float> actual0(element_count);
    std::vector<float> actual1(element_count);
    ggml_backend_tensor_get(out0, actual0.data(), 0, actual0.size() * sizeof(float));
    ggml_backend_tensor_get(out1, actual1.data(), 0, actual1.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual0[i] == expected0[i]);
        REQUIRE(actual1[i] == expected1[i]);
    }

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]    = static_cast<float>(i % 23) * -0.25f + 5.0f;
        b_data[i]    = static_cast<float>(i % 7) * 0.5f - 1.0f;
        c_data[i]    = static_cast<float>(i % 5) * -0.125f + 2.0f;
        d_data[i]    = static_cast<float>(i % 29) * 0.75f - 6.0f;
        expected0[i] = a_data[i] + b_data[i];
        expected1[i] = c_data[i] + d_data[i];
    }
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));
    ggml_backend_tensor_set(d, d_data.data(), 0, d_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 1);

    ggml_backend_tensor_get(out0, actual0.data(), 0, actual0.size() * sizeof(float));
    ggml_backend_tensor_get(out1, actual1.data(), 0, actual1.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual0[i] == expected0[i]);
        REQUIRE(actual1[i] == expected1[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_chained_add_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     a             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     c             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     sum           = ggml_add(ctx, a, b);
    ggml_tensor *     out           = ggml_add(ctx, sum, c);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);
    graph->uid = 1004;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> c_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        c_data[i]   = static_cast<float>(i % 7) * 0.125f + 1.0f;
        expected[i] = a_data[i] + b_data[i] + c_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_builds == 1);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 11) * -0.25f + 5.0f;
        b_data[i]   = static_cast<float>(i % 5) * 0.5f - 1.0f;
        c_data[i]   = static_cast<float>(i % 19) * 0.75f - 6.0f;
        expected[i] = a_data[i] + b_data[i] + c_data[i];
    }
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_hits == 1);

    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_same_uid_distinct_graph_reuses_graph_program() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    constexpr int64_t  element_count = 1024;
    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> expected(element_count);

    ggml_init_params params0 = {};
    params0.mem_size         = 256 * 1024;
    params0.no_alloc         = true;
    ggml_context * ctx0      = ggml_init(params0);
    REQUIRE(ctx0 != nullptr);

    ggml_tensor * a0   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, element_count);
    ggml_tensor * b0   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, element_count);
    ggml_tensor * out0 = ggml_add(ctx0, a0, b0);
    REQUIRE(a0 != nullptr);
    REQUIRE(b0 != nullptr);
    REQUIRE(out0 != nullptr);

    ggml_cgraph * graph0 = ggml_new_graph(ctx0);
    REQUIRE(graph0 != nullptr);
    ggml_build_forward_expand(graph0, out0);
    graph0->uid = 1003;

    ggml_backend_buffer_t buffer0 = ggml_backend_alloc_ctx_tensors(ctx0, backend);
    REQUIRE(buffer0 != nullptr);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        expected[i] = a_data[i] + b_data[i];
    }
    ggml_backend_tensor_set(a0, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b0, b_data.data(), 0, b_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph0) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    ggml_init_params params1 = {};
    params1.mem_size         = 256 * 1024;
    params1.no_alloc         = true;
    ggml_context * ctx1      = ggml_init(params1);
    REQUIRE(ctx1 != nullptr);

    ggml_tensor * a1   = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, element_count);
    ggml_tensor * b1   = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, element_count);
    ggml_tensor * out1 = ggml_add(ctx1, a1, b1);
    REQUIRE(a1 != nullptr);
    REQUIRE(b1 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph1 = ggml_new_graph(ctx1);
    REQUIRE(graph1 != nullptr);
    ggml_build_forward_expand(graph1, out1);
    graph1->uid = 1003;

    ggml_backend_buffer_t buffer1 = ggml_backend_alloc_ctx_tensors(ctx1, backend);
    REQUIRE(buffer1 != nullptr);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 23) * -0.25f + 5.0f;
        b_data[i]   = static_cast<float>(i % 7) * 0.5f - 1.0f;
        expected[i] = a_data[i] + b_data[i];
    }
    ggml_backend_tensor_set(a1, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b1, b_data.data(), 0, b_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph1) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_builds == 2);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out1, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer1);
    ggml_free(ctx1);
    ggml_backend_buffer_free(buffer0);
    ggml_free(ctx0);
    ggml_backend_free(backend);
}

static void run_unsupported_op_fails() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_sin(ctx, a);
    REQUIRE(a != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> input(8, 2.0f);
    ggml_backend_tensor_set(a, input.data(), 0, input.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    run_status_checks();
    run_command_plan_metadata_checks();
    run_dispatch_registry_checks();
    run_graph_import_checks();
    run_binary_f32_broadcast_dispatch_checks();
    run_graph_snapshot_diagnostics_checks();
    run_unmatched_graph_diagnostics_checks();
    run_completion_counter_plan_checks();
    run_graph_index_checks();
    run_rope_set_rows_dispatch_checks();
    run_graph_traversal_checks();
    run_qwen_token_embedding_dispatch_checks();
    run_gather_add_dispatch_checks();
    run_qwen_flash_attention_dispatch_checks();
    run_qwen_attention_postprocess_dispatch_checks();
    run_qwen_matmul_dispatch_checks();
    run_llama_attention_matmul_dispatch_checks();
    schedule_qwen_terminal_q6k_q8_command(1);
    schedule_qwen_terminal_q6k_q8_command(18);
    schedule_get_rows_q8_1_alternate_command();
    run_qwen_router_top8_dispatch_checks();
    run_common_mul_mat_id_swiglu_dispatch_checks();
    run_qwen_routed_gate_up_dispatch_checks();
    run_common_mul_mat_id_postops_dispatch_checks();
    run_alias_value_import_checks();
    run_multi_dispatch_checks();
    run_layout_alias_scheduler_elision_checks();
    run_transient_import_checks();
    run_chained_dispatch_requires_transients();
    run_graph_replay_host_staging_is_not_ineligible();
    run_multiple_transient_plan_checks();
    run_disjoint_transient_plan_packing_checks();
    run_command_program_kernel_dump_checks();
    run_graph_program_cache_uid_mismatch_checks();
    run_graph_executor_contract_checks();
    REQUIRE(ggml::hrx::loom_async_jit_enabled_from_environment() == async_jit_expected_from_environment());

    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }

    run_qwen_expert_table_partition_prefill_512_execution();
    run_add_f32();
    run_two_independent_add_f32();
    run_chained_add_f32();
    run_same_uid_distinct_graph_reuses_graph_program();
    run_unsupported_op_fails();
    return 0;
}
