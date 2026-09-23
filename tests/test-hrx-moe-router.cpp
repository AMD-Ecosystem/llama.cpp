#include "dispatch/dispatch-scheduler.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                            \
    do {                                                                                              \
        if (!(condition)) {                                                                           \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition);    \
            std::abort();                                                                             \
        }                                                                                             \
    } while (false)

int main() {
    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }
    constexpr int64_t hidden_size = 2048;
    constexpr int64_t expert_count = 128;
    constexpr int64_t route_count = 8;
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    REQUIRE(ctx != nullptr);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 1);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, expert_count);
    ggml_tensor * logits = ggml_mul_mat(ctx, weight, input);
    ggml_tensor * probabilities = ggml_soft_max(ctx, logits);
    ggml_tensor * sorted = ggml_argsort(ctx, probabilities, GGML_SORT_ORDER_DESC);
    ggml_tensor * routes = ggml_view_2d(ctx, sorted, route_count, 1, route_count * sizeof(int32_t), 0);
    ggml_tensor * selected = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probabilities, 1, expert_count, 1), routes);
    selected = ggml_reshape_2d(ctx, selected, route_count, 1);
    ggml_tensor * total = ggml_clamp(ctx, ggml_sum_rows(ctx, selected), 6.103515625e-5f,
                                    std::numeric_limits<float>::infinity());
    ggml_tensor * normalized = ggml_div(ctx, selected, total);
    ggml_tensor * output = ggml_reshape_3d(ctx, normalized, 1, route_count, 1);
    ggml_set_output(logits);
    ggml_set_output(routes);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    auto imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    const auto resolved = ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151",
                                                             scheduler.plan().dispatches[0].kernel.kernel_id);
    REQUIRE(resolved.found());
    REQUIRE(ggml::hrx::kernel_definition_name(*resolved.definition) ==
            std::string("qwen3_moe:qwen3_moe_router_projection_top8_fused_decode_f32"));

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    std::vector<float> weights(hidden_size * expert_count);
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t channel = 0; channel < hidden_size; ++channel) {
            weights[expert * hidden_size + channel] =
                float(expert + 1) / 4096.0f + float((channel * 7 + expert * 3) % 19 - 9) / 1024.0f;
        }
    }
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(float));

    // Change the input and poison all scores before each replay. Positive and
    // negative inputs select experts from opposite halves of the projection.
    for (int pass = 0; pass < 4; ++pass) {
        std::array<float, hidden_size> values;
        for (int64_t channel = 0; channel < hidden_size; ++channel) {
            values[channel] = (pass % 2 ? -1.0f : 1.0f) *
                              (0.03125f + float((channel * 11 + pass) % 23) / 512.0f);
        }
        std::array<double, expert_count> expected;
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            double sum = 0.0;
            for (int64_t channel = 0; channel < hidden_size; ++channel) {
                sum += double(values[channel]) * double(weights[expert * hidden_size + channel]);
            }
            expected[expert] = sum;
        }
        std::array<int32_t, expert_count> order;
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return expected[a] > expected[b]; });
        std::array<float, expert_count> actual;
        actual.fill(-1000.0f - pass);
        ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
        ggml_backend_tensor_set(logits, actual.data(), 0, sizeof(actual));
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(logits, actual.data(), 0, sizeof(actual));
        double maximum_error = 0.0;
        size_t bad_scores = 0;
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            const double error = std::fabs(actual[expert] - expected[expert]);
            maximum_error = std::max(maximum_error, error);
            bad_scores += !std::isfinite(actual[expert]) || error > 1.0e-5 + 1.0e-5 * std::fabs(expected[expert]);
        }
        std::array<int32_t, route_count> actual_routes;
        std::array<float, route_count> actual_weights;
        ggml_backend_tensor_get(routes, actual_routes.data(), 0, sizeof(actual_routes));
        ggml_backend_tensor_get(output, actual_weights.data(), 0, sizeof(actual_weights));
        double total_weight = 0.0;
        for (int64_t route = 0; route < route_count; ++route) {
            total_weight += std::exp(expected[order[route]] - expected[order[0]]);
        }
        std::printf("pass %d: bad_scores=%zu max_error=%.9g top_expert=%d expected=%d\n",
                    pass, bad_scores, maximum_error, actual_routes[0], order[0]);
        std::fflush(stdout);
        REQUIRE(bad_scores == 0);
        for (int64_t route = 0; route < route_count; ++route) {
            const double expected_weight = std::exp(expected[order[route]] - expected[order[0]]) / total_weight;
            REQUIRE(actual_routes[route] == order[route]);
            REQUIRE(std::isfinite(actual_weights[route]));
            REQUIRE(std::fabs(actual_weights[route] - expected_weight) < 1.0e-5);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
