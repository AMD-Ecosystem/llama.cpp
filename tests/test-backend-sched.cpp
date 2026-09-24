#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

struct deferred_backend {
    std::vector<std::function<void()>> pending;
    std::vector<std::array<float, 4>> results;
};

// Model a device that reads shared host memory after graph_compute returns.
static ggml_status deferred_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * state = static_cast<deferred_backend *>(backend->context);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_NONE || ggml_is_view(node)) {
            continue;
        }
        GGML_ASSERT(node->op == GGML_OP_SCALE);
        GGML_ASSERT(node->type == GGML_TYPE_F32 && ggml_nelements(node) == 4);
        const auto * src = static_cast<const float *>(node->src[0]->data);
        auto * dst = static_cast<float *>(node->data);
        float scale;
        std::memcpy(&scale, node->op_params, sizeof(scale));
        state->pending.push_back([=]() {
            std::array<float, 4> result;
            for (size_t j = 0; j < result.size(); ++j) {
                result[j] = dst[j] = src[j] * scale;
            }
            state->results.push_back(result);
        });
    }
    return GGML_STATUS_SUCCESS;
}

static void deferred_synchronize(ggml_backend_t backend) {
    auto * state = static_cast<deferred_backend *>(backend->context);
    for (const auto & compute : state->pending) {
        compute();
    }
    state->pending.clear();
}

static bool test_input_lifetime(int view_mode, bool parallel, bool shared_buffer, bool preallocated) {
    const bool view = view_mode != 0;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);
    deferred_backend state;
    ggml_backend_buffer_type buft = *ggml_backend_cpu_buffer_type();
    buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
        buffer->buft = type;
        return buffer;
    };
    ggml_backend_device device = {};
    device.context = &buft;
    device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
        return static_cast<ggml_backend_buffer_type_t>(dev->context);
    };
    device.iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t type) {
        return ggml_backend_buft_is_host(type);
    };
    device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * op) {
        return op->op == GGML_OP_SCALE || op->op == GGML_OP_NONE || ggml_is_view(op);
    };
    ggml_backend deferred = {};
    deferred.device = &device;
    deferred.context = &state;
    deferred.iface.get_name = [](ggml_backend_t) { return "deferred"; };
    deferred.iface.graph_compute = deferred_compute;
    deferred.iface.synchronize = deferred_synchronize;

    ggml_backend_t backends[] = { &deferred, cpu };
    ggml_backend_buffer_type_t bufts[] = { shared_buffer ? ggml_backend_cpu_buffer_type() : &buft, ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 64, parallel, false);
    ggml_context * ctx = ggml_init({ 16384, nullptr, true });
    GGML_ASSERT(sched && ctx);
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, view ? 8 : 4);
    if (view_mode != 2) {
        ggml_set_input(input);
    }
    ggml_backend_buffer_t input_buffer = preallocated ? ggml_backend_alloc_ctx_tensors(ctx, cpu) : nullptr;
    GGML_ASSERT(!preallocated || input_buffer);
    ggml_tensor * src = view ? ggml_view_1d(ctx, input, 4, 2 * sizeof(float)) : input;
    if (view_mode == 2) {
        ggml_set_input(src);
    }
    ggml_tensor * output = ggml_scale(ctx, src, 2.0f);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_set_tensor_backend(sched, output, &deferred);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));

    for (int iteration = 0; iteration < 3; ++iteration) {
        std::array<float, 8> values;
        for (size_t j = 0; j < values.size(); ++j) {
            values[j] = float(10 * iteration + j);
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
        // Reusing the caller's input must not change any queued computation.
        values.fill(-100.0f);
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    }
    ggml_backend_sched_synchronize(sched);

    bool ok = state.results.size() == 3;
    for (size_t iteration = 0; iteration < state.results.size(); ++iteration) {
        for (size_t j = 0; j < state.results[iteration].size(); ++j) {
            const float expected = 2.0f * float(10 * iteration + j + (view ? 2 : 0));
            if (state.results[iteration][j] != expected) {
                std::fprintf(stderr, "view=%d parallel=%d shared_buffer=%d preallocated=%d iteration=%zu element=%zu: got %g, expected %g\n",
                             view_mode, parallel, shared_buffer, preallocated, iteration, j, state.results[iteration][j], expected);
                ok = false;
            }
        }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return ok;
}

int main() {
    bool ok = true;
    for (int view_mode : { 0, 1, 2 }) {
        for (bool parallel : { false, true }) {
            for (bool shared_buffer : { false, true }) {
                for (bool preallocated : { false, true }) {
                    ok = test_input_lifetime(view_mode, parallel, shared_buffer, preallocated) && ok;
                }
            }
        }
    }
    return ok ? 0 : 1;
}
