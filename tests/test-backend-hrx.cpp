#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-hrx.h>
#include <ggml-hrx-test.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

static ggml_context_ptr make_context() {
    ggml_init_params params = {
        /* .mem_size   = */ 256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(96, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

static void expect_eq(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n",
                label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void expect_near(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float tolerance,
        const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g tolerance %.9g\n",
                label, i, actual[i], expected[i], tolerance);
            std::abort();
        }
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * tensor) {
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());

    const auto * traits = ggml_get_type_traits(tensor->type);
    const size_t block_size = ggml_blck_size(tensor->type);
    const bool quantized = ggml_is_quantized(tensor->type);
    std::vector<float> block_values(block_size);
    std::vector<float> values;
    values.reserve(ggml_nelements(tensor));

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; i0 += block_size) {
                    const size_t offset =
                        static_cast<size_t>(i3) * tensor->nb[3] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(i1) * tensor->nb[1] +
                        static_cast<size_t>(i0 / block_size) * tensor->nb[0];
                    if (tensor->type == GGML_TYPE_F32) {
                        float value = 0.0f;
                        std::memcpy(&value, data.data() + offset, sizeof(value));
                        values.push_back(value);
                    } else if (tensor->type == GGML_TYPE_F16) {
                        ggml_fp16_t value = 0;
                        std::memcpy(&value, data.data() + offset, sizeof(value));
                        values.push_back(ggml_fp16_to_fp32(value));
                    } else if (quantized) {
                        traits->to_float(data.data() + offset, block_values.data(), block_size);
                        values.insert(values.end(), block_values.begin(), block_values.end());
                    } else {
                        GGML_ABORT("unsupported tensor_to_float type");
                    }
                }
            }
        }
    }
    return values;
}

static std::vector<int32_t> tensor_to_i32(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_I32);
    std::vector<int32_t> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(int32_t));
    return values;
}

static std::vector<float> reference_mul_mat(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        int64_t k,
        int64_t rows,
        int64_t cols) {
    std::vector<float> output(static_cast<size_t>(rows * cols), 0.0f);
    for (int64_t col = 0; col < cols; ++col) {
        for (int64_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += lhs[static_cast<size_t>(row * k + i)] * rhs[static_cast<size_t>(col * k + i)];
            }
            output[static_cast<size_t>(col * rows + row)] = sum;
        }
    }
    return output;
}

static std::vector<float> reference_gated_delta_net(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & g,
        const std::vector<float> & beta,
        const std::vector<float> & state,
        int64_t S_v,
        int64_t H,
        int64_t q_heads,
        int64_t n_tokens,
        int64_t n_seqs) {
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    std::vector<float> output(static_cast<size_t>(attn_score_elems + S_v * S_v * H * n_seqs), 0.0f);
    std::vector<float> state_out = state;
    const float scale = 1.0f / std::sqrt(static_cast<float>(S_v));

    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t head = 0; head < H; ++head) {
            const int64_t q_head = head % q_heads;
            float * s = state_out.data() + static_cast<size_t>((seq * H + head) * S_v * S_v);
            for (int64_t token = 0; token < n_tokens; ++token) {
                const float * q_t =
                    q.data() + static_cast<size_t>((seq * n_tokens * q_heads + token * q_heads + q_head) * S_v);
                const float * k_t =
                    k.data() + static_cast<size_t>((seq * n_tokens * q_heads + token * q_heads + q_head) * S_v);
                const float * v_t =
                    v.data() + static_cast<size_t>((seq * n_tokens * H + token * H + head) * S_v);
                const float g_val = std::exp(g[static_cast<size_t>(seq * n_tokens * H + token * H + head)]);
                const float beta_val = beta[static_cast<size_t>(seq * n_tokens * H + token * H + head)];

                std::vector<float> delta(static_cast<size_t>(S_v), 0.0f);
                for (int64_t col = 0; col < S_v; ++col) {
                    float kv = 0.0f;
                    for (int64_t row = 0; row < S_v; ++row) {
                        kv += s[static_cast<size_t>(col * S_v + row)] * k_t[row];
                    }
                    delta[static_cast<size_t>(col)] = (v_t[col] - g_val * kv) * beta_val;
                }

                for (int64_t col = 0; col < S_v; ++col) {
                    for (int64_t row = 0; row < S_v; ++row) {
                        s[static_cast<size_t>(col * S_v + row)] =
                            g_val * s[static_cast<size_t>(col * S_v + row)] + k_t[row] * delta[static_cast<size_t>(col)];
                    }
                }

                for (int64_t col = 0; col < S_v; ++col) {
                    float attn = 0.0f;
                    for (int64_t row = 0; row < S_v; ++row) {
                        attn += s[static_cast<size_t>(col * S_v + row)] * q_t[row];
                    }
                    output[static_cast<size_t>((seq * n_tokens * H + token * H + head) * S_v + col)] = attn * scale;
                }
            }
        }
    }

    std::copy(state_out.begin(), state_out.end(), output.begin() + attn_score_elems);
    return output;
}

static std::vector<float> reference_mul_mat_batched(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        int64_t k,
        int64_t rows,
        int64_t cols,
        int64_t lhs_ne2,
        int64_t lhs_ne3,
        int64_t out_ne2,
        int64_t out_ne3) {
    std::vector<float> output(static_cast<size_t>(rows * cols * out_ne2 * out_ne3), 0.0f);
    for (int64_t i3 = 0; i3 < out_ne3; ++i3) {
        const int64_t lhs_i3 = lhs_ne3 == out_ne3 ? i3 : i3 / (out_ne3 / lhs_ne3);
        for (int64_t i2 = 0; i2 < out_ne2; ++i2) {
            const int64_t lhs_i2 = lhs_ne2 == out_ne2 ? i2 : i2 / (out_ne2 / lhs_ne2);
            for (int64_t col = 0; col < cols; ++col) {
                for (int64_t row = 0; row < rows; ++row) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < k; ++i) {
                        const size_t lhs_idx = static_cast<size_t>(
                            i + k * (row + rows * (lhs_i2 + lhs_ne2 * lhs_i3)));
                        const size_t rhs_idx = static_cast<size_t>(
                            i + k * (col + cols * (i2 + out_ne2 * i3)));
                        sum += lhs[lhs_idx] * rhs[rhs_idx];
                    }
                    output[static_cast<size_t>(row + rows * (col + cols * (i2 + out_ne2 * i3)))] = sum;
                }
            }
        }
    }
    return output;
}

static std::vector<float> reference_mul_mat_id_q4_k(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const std::vector<int32_t> & ids,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs) {
    std::vector<float> output(static_cast<size_t>(rows * n_ids * n_tokens), 0.0f);
    const int64_t rhs_cols = broadcast_rhs ? 1 : n_ids;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id_pos = 0; id_pos < n_ids; ++id_pos) {
            const int32_t expert = ids[static_cast<size_t>(id_pos + n_ids * token)];
            GGML_ASSERT(expert >= 0 && expert < n_experts);
            const int64_t rhs_col = broadcast_rhs ? 0 : id_pos;
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    const size_t lhs_idx = static_cast<size_t>(i + k * (row + rows * expert));
                    const size_t rhs_idx = static_cast<size_t>(i + k * (rhs_col + rhs_cols * token));
                    sum += lhs[lhs_idx] * rhs[rhs_idx];
                }
                output[static_cast<size_t>(row + rows * (id_pos + n_ids * token))] = sum;
            }
        }
    }
    return output;
}

static float reference_mul_mat_id_q4_k_value(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const std::vector<int32_t> & ids,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs,
        int64_t row,
        int64_t id_pos,
        int64_t token) {
    (void) rows;
    (void) n_tokens;
    const int32_t expert = ids[static_cast<size_t>(id_pos + n_ids * token)];
    GGML_ASSERT(expert >= 0 && expert < n_experts);
    const int64_t rhs_cols = broadcast_rhs ? 1 : n_ids;
    const int64_t rhs_col = broadcast_rhs ? 0 : id_pos;
    float sum = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        const size_t lhs_idx = static_cast<size_t>(i + k * (row + rows * expert));
        const size_t rhs_idx = static_cast<size_t>(i + k * (rhs_col + rhs_cols * token));
        sum += lhs[lhs_idx] * rhs[rhs_idx];
    }
    return sum;
}

static std::vector<int64_t> sample_positions(int64_t n) {
    std::vector<int64_t> positions = { 0 };
    if (n > 2) {
        positions.push_back(n / 2);
    }
    if (n > 1) {
        positions.push_back(n - 1);
    }
    return positions;
}

static void expect_mul_mat_id_q4_k_samples(
        const std::vector<float> & actual,
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const std::vector<int32_t> & ids,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs,
        float tolerance,
        const char * label) {
    const std::vector<int64_t> row_samples = sample_positions(rows);
    const std::vector<int64_t> id_samples = sample_positions(n_ids);
    const std::vector<int64_t> token_samples = sample_positions(n_tokens);
    for (const int64_t token : token_samples) {
        for (const int64_t id_pos : id_samples) {
            for (const int64_t row : row_samples) {
                const size_t idx = static_cast<size_t>(row + rows * (id_pos + n_ids * token));
                const float expected = reference_mul_mat_id_q4_k_value(
                    lhs, rhs, ids, k, rows, n_ids, n_tokens, n_experts, broadcast_rhs, row, id_pos, token);
                const float got = actual[idx];
                if (std::fabs(got - expected) > tolerance) {
                    std::fprintf(stderr,
                        "%s[%lld,%lld,%lld]: got %.9g expected %.9g tolerance %.9g\n",
                        label,
                        static_cast<long long>(row),
                        static_cast<long long>(id_pos),
                        static_cast<long long>(token),
                        got,
                        expected,
                        tolerance);
                    std::abort();
                }
            }
        }
    }
}

static void prepare_rows(
        ggml_type type,
        int64_t ncols,
        int64_t nrows,
        const std::vector<float> & input,
        std::vector<float> & reference,
        std::vector<uint8_t> & storage) {
    reference.resize(input.size());
    if (type == GGML_TYPE_F32) {
        storage.resize(input.size() * sizeof(float));
        std::memcpy(storage.data(), input.data(), storage.size());
        reference = input;
        return;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    GGML_ASSERT(traits->from_float_ref != nullptr);
    GGML_ASSERT(traits->to_float != nullptr);
    const size_t row_bytes = ggml_row_size(type, ncols);
    storage.assign(row_bytes * static_cast<size_t>(nrows), 0);
    for (int64_t row = 0; row < nrows; ++row) {
        const int64_t row_offset = row * ncols;
        uint8_t * row_data = storage.data() + row_bytes * static_cast<size_t>(row);
        traits->from_float_ref(input.data() + row_offset, row_data, ncols);
        traits->to_float(row_data, reference.data() + row_offset, ncols);
    }
}

static void prepare_mul_mat_lhs(
        ggml_type type,
        int64_t k,
        int64_t rows,
        const std::vector<float> & input,
        std::vector<float> & reference,
        std::vector<uint8_t> & storage) {
    prepare_rows(type, k, rows, input, reference, storage);
}

static void run_mul_mat_vec_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        ggml_type lhs_type,
        int64_t k,
        int64_t rows,
        int64_t cols,
        float tolerance,
        const char * label) {
    GGML_ASSERT(k % ggml_blck_size(lhs_type) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), lhs_type, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows));
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 5) % 101) - 50) / 37.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 89) - 44) / 41.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(lhs_type, k, rows, lhs_f32, lhs_reference, lhs_storage);

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(static_cast<size_t>(rows * cols), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols), tolerance, label);
}

static void run_mul_mat_vec_batched_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        ggml_type lhs_type,
        int64_t k,
        int64_t rows,
        int64_t cols,
        int64_t lhs_ne2,
        int64_t lhs_ne3,
        int64_t out_ne2,
        int64_t out_ne3,
        float tolerance,
        const char * label) {
    GGML_ASSERT(k % ggml_blck_size(lhs_type) == 0);
    GGML_ASSERT(out_ne2 % lhs_ne2 == 0);
    GGML_ASSERT(out_ne3 % lhs_ne3 == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_4d(ctx.get(), lhs_type, k, rows, lhs_ne2, lhs_ne3);
    ggml_tensor * rhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, k, cols, out_ne2, out_ne3);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows * lhs_ne2 * lhs_ne3));
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols * out_ne2 * out_ne3));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 7) % 113) - 56) / 43.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 3) % 97) - 48) / 47.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(lhs_type, k, rows * lhs_ne2 * lhs_ne3, lhs_f32, lhs_reference, lhs_storage);

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(
        tensor_to_float(out),
        reference_mul_mat_batched(lhs_reference, rhs_f32, k, rows, cols, lhs_ne2, lhs_ne3, out_ne2, out_ne3),
        tolerance, label);
}

static bool ggml_type_from_name(const char * name, ggml_type * out_type) {
    if (!name || !out_type) {
        return false;
    }
    if (std::strcmp(name, "F32") == 0) {
        *out_type = GGML_TYPE_F32;
        return true;
    }
    if (std::strcmp(name, "F16") == 0) {
        *out_type = GGML_TYPE_F16;
        return true;
    }
    if (std::strcmp(name, "BF16") == 0) {
        *out_type = GGML_TYPE_BF16;
        return true;
    }
    if (std::strcmp(name, "Q4_K") == 0) {
        *out_type = GGML_TYPE_Q4_K;
        return true;
    }
    if (std::strcmp(name, "Q5_K") == 0) {
        *out_type = GGML_TYPE_Q5_K;
        return true;
    }
    if (std::strcmp(name, "Q6_K") == 0) {
        *out_type = GGML_TYPE_Q6_K;
        return true;
    }
    if (std::strcmp(name, "Q8_0") == 0) {
        *out_type = GGML_TYPE_Q8_0;
        return true;
    }
    if (std::strcmp(name, "Q8_1") == 0) {
        *out_type = GGML_TYPE_Q8_1;
        return true;
    }
    if (std::strcmp(name, "I32") == 0) {
        *out_type = GGML_TYPE_I32;
        return true;
    }
    if (std::strcmp(name, "I64") == 0) {
        *out_type = GGML_TYPE_I64;
        return true;
    }
    return false;
}

static void run_catalog_mul_mat_schedule_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const ggml_backend_hrx_test_case & cfg) {
    GGML_ASSERT(cfg.op == "MUL_MAT");
    GGML_ASSERT(!cfg.id.empty() && !cfg.expected_route_id.empty() && cfg.repeat > 0);

    ggml_type lhs_type = GGML_TYPE_COUNT;
    ggml_type rhs_type = GGML_TYPE_COUNT;
    ggml_type dst_type = GGML_TYPE_COUNT;
    const auto support_value = [&](const char * key) -> const char * {
        const auto it = cfg.supports.find(key);
        return it == cfg.supports.end() ? "" : it->second.c_str();
    };
    const auto shape_value = [&](const char * key) -> int64_t {
        const auto it = cfg.shape.find(key);
        return it == cfg.shape.end() ? 0 : it->second;
    };
    GGML_ASSERT(ggml_type_from_name(support_value("src0_type"), &lhs_type));
    GGML_ASSERT(ggml_type_from_name(support_value("src1_type"), &rhs_type));
    GGML_ASSERT(ggml_type_from_name(support_value("dst_type"), &dst_type));
    GGML_ASSERT(rhs_type == GGML_TYPE_F32);
    GGML_ASSERT(dst_type == GGML_TYPE_F32);
    const int64_t k = shape_value("k");
    const int64_t rows = shape_value("rows");
    const int64_t cols = shape_value("cols");
    GGML_ASSERT(k > 0 && rows > 0 && cols > 0);

    ggml_backend_hrx_test_reset_dispatch_record();
    for (uint32_t i = 0; i < cfg.repeat; ++i) {
        run_mul_mat_vec_case(
            backend, dev, lhs_type, k, rows, cols,
            cfg.tolerance, cfg.id.c_str());
    }

    const ggml_backend_hrx_test_route_record record =
        ggml_backend_hrx_test_get_route_record(cfg.expected_route_id);
    GGML_ASSERT(record.dispatch_count == cfg.repeat);
    GGML_ASSERT(record.jit_compile_count <= 1);
    GGML_ASSERT(record.jit_compile_count + record.jit_cache_hit_count == cfg.repeat);
}

static int64_t catalog_shape(const ggml_backend_hrx_test_case & cfg, const char * key);
static int64_t catalog_shape_or(const ggml_backend_hrx_test_case & cfg, const char * key, int64_t fallback);

static void run_catalog_mul_mat_q4_k_f32_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 256, 3);
        ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 1);
        ggml_tensor * out = ggml_mul_mat(ctx.get(), lhs, rhs);
        GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));
    }
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, 256, 3, 2);
        ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 1, 2);
        ggml_tensor * out = ggml_mul_mat(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 256, 3);
        ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 256, 1);
        ggml_tensor * out = ggml_mul_mat(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }

    const std::vector<ggml_backend_hrx_test_case> cases =
        ggml_backend_hrx_test_cases(dev, "mul_mat_q4_k_f32");
    GGML_ASSERT(!cases.empty());
    for (const ggml_backend_hrx_test_case & cfg : cases) {
        run_catalog_mul_mat_schedule_case(backend, dev, cfg);
    }
}

static void run_catalog_mul_mat_f16_batched_cont_once(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const ggml_backend_hrx_test_case & cfg) {
    const int64_t k = catalog_shape(cfg, "k");
    const int64_t rows = catalog_shape(cfg, "rows");
    const int64_t cols = catalog_shape(cfg, "cols");
    const int64_t nheads = catalog_shape_or(cfg, "nheads", 24);
    const int64_t nheads_kv = catalog_shape_or(cfg, "nheads_kv", 8);
    GGML_ASSERT(nheads % nheads_kv == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, k, rows, nheads_kv);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, cols, nheads);
    ggml_tensor * mm = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * permute = ggml_permute(ctx.get(), mm, 0, 2, 1, 3);
    ggml_tensor * out = ggml_cont_2d(ctx.get(), permute, rows * nheads, cols);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows * nheads_kv));
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols * nheads));
    std::vector<uint16_t> lhs_f16(lhs_f32.size());
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 67) - 33) / 59.0f;
        lhs_f16[i] = ggml_fp32_to_fp16(lhs_f32[i]);
        lhs_f32[i] = ggml_fp16_to_fp32(lhs_f16[i]);
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 5) % 71) - 35) / 61.0f;
    }
    ggml_backend_tensor_set(lhs, lhs_f16.data(), 0, lhs_f16.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const std::vector<float> actual = tensor_to_float(out);
    const int64_t head_group = nheads / nheads_kv;
    for (int64_t head : sample_positions(nheads)) {
        const int64_t kv_head = head / head_group;
        for (int64_t row : sample_positions(rows)) {
            float expected = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                const float a = lhs_f32[static_cast<size_t>(i + k * (row + rows * kv_head))];
                const float b = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    rhs_f32[static_cast<size_t>(i + k * (0 + cols * head))]));
                expected += a * b;
            }
            const size_t out_index = static_cast<size_t>(row + rows * head);
            const float got = actual[out_index];
            const float tolerance = cfg.tolerance + std::fabs(expected) * 3.0e-3f;
            if (!std::isfinite(got) || std::fabs(got - expected) > tolerance) {
                std::fprintf(stderr,
                    "%s[row=%lld head=%lld]: got %.9g expected %.9g tolerance %.9g\n",
                    cfg.id.c_str(), (long long) row, (long long) head, got, expected, tolerance);
                std::abort();
            }
        }
    }
}

static void run_catalog_softmax_kqv_once(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const ggml_backend_hrx_test_case & cfg) {
    const int64_t kv = catalog_shape(cfg, "kv");
    const int64_t n = catalog_shape(cfg, "n");
    const int64_t nheads = catalog_shape(cfg, "nheads");
    const int64_t nheads_kv = catalog_shape_or(cfg, "nheads_kv", 8);
    const int64_t d = catalog_shape_or(cfg, "d", 128);

    auto build_graph = [&](ggml_context * ctx, ggml_tensor ** kq, ggml_tensor ** mask, ggml_tensor ** v) {
        *kq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv, n, nheads);
        *mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv, n, nheads);
        *v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kv, d, nheads_kv);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, *kq, *mask, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, *v, probs);
        ggml_tensor * permute = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        return ggml_cont_2d(ctx, permute, d * nheads, n);
    };

    ggml_context_ptr ctx = make_context();
    ggml_tensor * kq = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * out = build_graph(ctx.get(), &kq, &mask, &v);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> kq_data(static_cast<size_t>(kv * n * nheads));
    std::vector<float> mask_data(kq_data.size());
    std::vector<float> v_f32(static_cast<size_t>(kv * d * nheads_kv));
    std::vector<uint16_t> v_f16(v_f32.size());
    for (size_t i = 0; i < kq_data.size(); ++i) {
        kq_data[i] = static_cast<float>(static_cast<int>((i * 11 + 7) % 41) - 20) / 16.0f;
        mask_data[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 17) - 8) / 32.0f;
    }
    for (size_t i = 0; i < v_f32.size(); ++i) {
        v_f32[i] = static_cast<float>(static_cast<int>((i * 13 + 5) % 43) - 21) / 19.0f;
        v_f16[i] = ggml_fp32_to_fp16(v_f32[i]);
    }
    ggml_backend_tensor_set(kq, kq_data.data(), 0, kq_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(uint16_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> actual = tensor_to_float(out);

    ggml_context_ptr cpu_ctx = make_context();
    ggml_tensor * cpu_kq = nullptr;
    ggml_tensor * cpu_mask = nullptr;
    ggml_tensor * cpu_v = nullptr;
    ggml_tensor * cpu_out = build_graph(cpu_ctx.get(), &cpu_kq, &cpu_mask, &cpu_v);
    ggml_cgraph * cpu_graph = ggml_new_graph_custom(cpu_ctx.get(), 32, false);
    ggml_build_forward_expand(cpu_graph, cpu_out);
    ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(cpu_backend != nullptr);
    ggml_backend_buffer_ptr cpu_buffer(ggml_backend_alloc_ctx_tensors(cpu_ctx.get(), cpu_backend.get()));
    GGML_ASSERT(cpu_buffer != nullptr);
    ggml_backend_tensor_set(cpu_kq, kq_data.data(), 0, kq_data.size() * sizeof(float));
    ggml_backend_tensor_set(cpu_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    ggml_backend_tensor_set(cpu_v, v_f16.data(), 0, v_f16.size() * sizeof(uint16_t));
    GGML_ASSERT(ggml_backend_graph_compute(cpu_backend.get(), cpu_graph) == GGML_STATUS_SUCCESS);
    expect_near(actual, tensor_to_float(cpu_out), cfg.tolerance, cfg.id.c_str());
}

static int64_t catalog_shape(const ggml_backend_hrx_test_case & cfg, const char * key) {
    const auto it = cfg.shape.find(key);
    GGML_ASSERT(it != cfg.shape.end());
    return it->second;
}

static int64_t catalog_shape_or(const ggml_backend_hrx_test_case & cfg, const char * key, int64_t fallback) {
    const auto it = cfg.shape.find(key);
    return it == cfg.shape.end() ? fallback : it->second;
}

static const std::string & catalog_support(const ggml_backend_hrx_test_case & cfg, const char * key) {
    const auto it = cfg.supports.find(key);
    GGML_ASSERT(it != cfg.supports.end());
    return it->second;
}

static void run_cpy_f32_f16_case(ggml_backend_t backend);
static void run_argsort_case(ggml_backend_t backend, int64_t ncols, int64_t nrows, ggml_sort_order order);

static void run_catalog_pointwise_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nrows = catalog_shape(cfg, "nrows");
    const std::string & layout = catalog_support(cfg, "layout");
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * rhs = nullptr;
    if (layout == "contiguous_src0_rhs_column_broadcast") {
        rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, nrows);
    } else if (layout == "contiguous_src0_rhs_row_broadcast") {
        rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, 1);
    } else {
        rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    }
    ggml_tensor * out = cfg.op == "ADD" ? ggml_add(ctx.get(), lhs, rhs) :
        (cfg.op == "MUL" ? ggml_mul(ctx.get(), lhs, rhs) : ggml_div(ctx.get(), lhs, rhs));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(static_cast<size_t>(ncols * nrows));
    std::vector<float> rhs_data(static_cast<size_t>(ggml_nelements(rhs)));
    std::vector<float> expected(lhs_data.size());
    for (int64_t i = 0; i < ncols * nrows; ++i) {
        lhs_data[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    for (size_t i = 0; i < rhs_data.size(); ++i) {
        rhs_data[i] = static_cast<float>(static_cast<int>(i % 11) + 1) * 0.25f;
    }
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            const float a = lhs_data[static_cast<size_t>(row * ncols + col)];
            float b = 0.0f;
            if (layout == "contiguous_src0_rhs_column_broadcast") {
                b = rhs_data[static_cast<size_t>(row)];
            } else if (layout == "contiguous_src0_rhs_row_broadcast") {
                b = rhs_data[static_cast<size_t>(col)];
            } else {
                b = rhs_data[static_cast<size_t>(row * ncols + col)];
            }
            expected[static_cast<size_t>(row * ncols + col)] =
                cfg.op == "ADD" ? a + b : (cfg.op == "MUL" ? a * b : a / b);
        }
    }
    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_unary_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nrows = catalog_shape(cfg, "nrows");
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = nullptr;
    if (cfg.op == "SCALE") {
        out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);
    } else if (cfg.op == "CLAMP") {
        out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);
    } else if (cfg.op == "RMS_NORM") {
        out = ggml_rms_norm(ctx.get(), src, 1.0e-6f);
    } else if (cfg.op == "SUM_ROWS") {
        out = ggml_sum_rows(ctx.get(), src);
    } else if (cfg.op == "SOFT_MAX") {
        out = ggml_soft_max_ext(ctx.get(), src, nullptr, 0.25f, 0.0f);
    } else {
        GGML_ABORT("unsupported catalog unary op");
    }
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nrows));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.0625f;
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    if (cfg.op == "SCALE" || cfg.op == "CLAMP" || cfg.op == "RMS_NORM" || cfg.op == "SOFT_MAX") {
        std::vector<float> expected(src_data.size());
        for (int64_t row = 0; row < nrows; ++row) {
            if (cfg.op == "RMS_NORM") {
                float sum = 0.0f;
                for (int64_t col = 0; col < ncols; ++col) {
                    const float value = src_data[static_cast<size_t>(row * ncols + col)];
                    sum += value * value;
                }
                const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + 1.0e-6f);
                for (int64_t col = 0; col < ncols; ++col) {
                    expected[static_cast<size_t>(row * ncols + col)] = src_data[static_cast<size_t>(row * ncols + col)] * scale;
                }
            } else if (cfg.op == "SOFT_MAX") {
                float max_value = -INFINITY;
                for (int64_t col = 0; col < ncols; ++col) {
                    max_value = std::max(max_value, src_data[static_cast<size_t>(row * ncols + col)] * 0.25f);
                }
                float sum = 0.0f;
                for (int64_t col = 0; col < ncols; ++col) {
                    sum += std::exp(src_data[static_cast<size_t>(row * ncols + col)] * 0.25f - max_value);
                }
                for (int64_t col = 0; col < ncols; ++col) {
                    expected[static_cast<size_t>(row * ncols + col)] =
                        std::exp(src_data[static_cast<size_t>(row * ncols + col)] * 0.25f - max_value) / sum;
                }
            } else {
                for (int64_t col = 0; col < ncols; ++col) {
                    const size_t index = static_cast<size_t>(row * ncols + col);
                    expected[index] = cfg.op == "SCALE" ? src_data[index] * 1.25f - 0.75f :
                        (src_data[index] < -1.5f ? -1.5f : (src_data[index] > 2.0f ? 2.0f : src_data[index]));
                }
            }
        }
        expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
    } else {
        std::vector<float> expected(static_cast<size_t>(nrows));
        for (int64_t row = 0; row < nrows; ++row) {
            float sum = 0.0f;
            for (int64_t col = 0; col < ncols; ++col) {
                sum += src_data[static_cast<size_t>(row * ncols + col)];
            }
            expected[static_cast<size_t>(row)] = sum;
        }
        expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
    }
}

using catalog_case_runner = void (*)(ggml_backend_t, ggml_backend_dev_t, const ggml_backend_hrx_test_case &);

static void run_catalog_checked_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const ggml_backend_hrx_test_case & cfg,
        catalog_case_runner runner) {
    GGML_ASSERT(!cfg.id.empty() && !cfg.expected_route_id.empty() && cfg.repeat > 0);
    ggml_backend_hrx_test_reset_dispatch_record();
    for (uint32_t i = 0; i < cfg.repeat; ++i) {
        runner(backend, dev, cfg);
    }
    const ggml_backend_hrx_test_route_record record =
        ggml_backend_hrx_test_get_route_record(cfg.expected_route_id);
    GGML_ASSERT(record.dispatch_count == cfg.repeat);
    GGML_ASSERT(record.jit_compile_count <= 1);
    GGML_ASSERT(record.jit_compile_count + record.jit_cache_hit_count == cfg.repeat);
}

static void run_catalog_argsort_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    GGML_UNUSED(dev);
    run_argsort_case(backend, catalog_shape(cfg, "ncols"), catalog_shape(cfg, "nrows"), GGML_SORT_ORDER_DESC);
}

static void run_catalog_copy_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    GGML_UNUSED(dev);
    GGML_ASSERT(catalog_shape(cfg, "ncols") == 257);
    run_cpy_f32_f16_case(backend);
}

static void run_catalog_get_rows_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    if (catalog_support(cfg, "layout") == "moe_weights_topk_view") {
        const int64_t nselected = catalog_shape_or(cfg, "nselected", catalog_shape(cfg, "ncols"));
        const int64_t ntokens = catalog_shape_or(cfg, "ntokens", catalog_shape(cfg, "nrows"));
        const int64_t nexperts = catalog_shape_or(cfg, "nexperts", 128);

        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, nexperts, ntokens);
        ggml_tensor * rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, nselected, ntokens);
        ggml_tensor * out = ggml_get_rows(ctx.get(), src, rows);
        GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        GGML_ASSERT(buffer != nullptr);

        std::vector<float> src_data(static_cast<size_t>(nexperts * ntokens));
        std::vector<int32_t> row_data(static_cast<size_t>(nselected * ntokens));
        for (size_t i = 0; i < src_data.size(); ++i) {
            src_data[i] = static_cast<float>(static_cast<int>((i * 19 + 3) % 127) - 63) * 0.03125f;
        }
        for (int64_t token = 0; token < ntokens; ++token) {
            for (int64_t selected = 0; selected < nselected; ++selected) {
                row_data[static_cast<size_t>(selected + nselected * token)] =
                    static_cast<int32_t>((selected * 7 + token * 11) % nexperts);
            }
        }

        ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
        ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int32_t));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

        std::vector<float> expected(static_cast<size_t>(nselected * ntokens));
        for (int64_t token = 0; token < ntokens; ++token) {
            for (int64_t selected = 0; selected < nselected; ++selected) {
                const int64_t expert = row_data[static_cast<size_t>(selected + nselected * token)];
                expected[static_cast<size_t>(selected + nselected * token)] =
                    src_data[static_cast<size_t>(expert + nexperts * token)];
            }
        }
        expect_eq(tensor_to_float(out), expected, cfg.id.c_str());
        return;
    }

    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t rows_to_get = catalog_shape(cfg, "nrows");
    ggml_type src_type = GGML_TYPE_COUNT;
    GGML_ASSERT(ggml_type_from_name(catalog_support(cfg, "src0_type").c_str(), &src_type));
    GGML_ASSERT(src_type == GGML_TYPE_F32 || ggml_is_quantized(src_type));
    GGML_ASSERT(ncols % ggml_blck_size(src_type) == 0);

    const int64_t src_rows = std::max<int64_t>(8, rows_to_get + 3);
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, ncols, src_rows);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rows_to_get);
    ggml_tensor * out = ggml_get_rows(ctx.get(), src, rows);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * src_rows));
    std::vector<int32_t> row_data(static_cast<size_t>(rows_to_get));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 17 + 5) % 251) - 125) * 0.015625f;
    }
    for (int64_t row = 0; row < rows_to_get; ++row) {
        row_data[static_cast<size_t>(row)] = static_cast<int32_t>((row * 3) % src_rows);
    }

    std::vector<float> src_reference;
    std::vector<uint8_t> src_storage;
    prepare_rows(src_type, ncols, src_rows, src_data, src_reference, src_storage);

    ggml_backend_tensor_set(src, src_storage.data(), 0, src_storage.size());
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(ncols * rows_to_get));
    for (int64_t row = 0; row < rows_to_get; ++row) {
        const int64_t selected = row_data[static_cast<size_t>(row)];
        for (int64_t col = 0; col < ncols; ++col) {
            expected.push_back(src_reference[static_cast<size_t>(selected * ncols + col)]);
        }
    }
    if (cfg.tolerance == 0.0f) {
        expect_eq(tensor_to_float(out), expected, cfg.id.c_str());
    } else {
        expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
    }
}

static void run_catalog_mul_mat_id_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    ggml_type lhs_type = GGML_TYPE_COUNT;
    GGML_ASSERT(ggml_type_from_name(catalog_support(cfg, "src0_type").c_str(), &lhs_type));
    GGML_ASSERT(lhs_type == GGML_TYPE_Q4_K || lhs_type == GGML_TYPE_Q5_K || lhs_type == GGML_TYPE_Q6_K);
    const int64_t k = catalog_shape(cfg, "k");
    const int64_t rows = catalog_shape(cfg, "rows");
    const int64_t n_ids = catalog_shape(cfg, "nselected");
    const int64_t n_tokens = catalog_shape(cfg, "ntokens");
    const int64_t n_experts = catalog_shape(cfg, "nexperts");
    GGML_ASSERT(k % ggml_blck_size(lhs_type) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), lhs_type, k, rows, n_experts);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, n_ids, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_ids, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx.get(), lhs, rhs, ids);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows * n_experts));
    std::vector<float> rhs_f32(static_cast<size_t>(k * n_ids * n_tokens));
    std::vector<int32_t> ids_i32(static_cast<size_t>(n_ids * n_tokens));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 29 + 17) % 127) - 63) / 53.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 31 + 5) % 109) - 54) / 59.0f;
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id_pos = 0; id_pos < n_ids; ++id_pos) {
            ids_i32[static_cast<size_t>(id_pos + n_ids * token)] =
                static_cast<int32_t>((id_pos * 3 + token * 5) % n_experts);
        }
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(lhs_type, k, rows * n_experts, lhs_f32, lhs_reference, lhs_storage);

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(
        tensor_to_float(out),
        reference_mul_mat_id_q4_k(lhs_reference, rhs_f32, ids_i32, k, rows, n_ids, n_tokens, n_experts, false),
        cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_set_rows_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    ggml_type dst_type = GGML_TYPE_COUNT;
    GGML_ASSERT(ggml_type_from_name(catalog_support(cfg, "dst_type").c_str(), &dst_type));
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t rows_to_set = catalog_shape(cfg, "nrows");
    const int64_t dst_rows = catalog_shape_or(cfg, "dst_rows", rows_to_set + 2);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), dst_type, ncols, dst_rows);
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, rows_to_set);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, rows_to_set);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, src, rows);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    std::vector<float> src_data(static_cast<size_t>(ncols * rows_to_set));
    std::vector<int64_t> row_data(static_cast<size_t>(rows_to_set));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 17 + 7) % 97) - 48) * 0.03125f;
    }
    for (int64_t row = 0; row < rows_to_set; ++row) {
        row_data[static_cast<size_t>(row)] = (row * 2 + 1) % dst_rows;
    }
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int64_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(ncols * dst_rows), 0.0f);
    for (int64_t row = 0; row < rows_to_set; ++row) {
        const int64_t dst_row = row_data[static_cast<size_t>(row)];
        for (int64_t col = 0; col < ncols; ++col) {
            expected[static_cast<size_t>(dst_row * ncols + col)] = src_data[static_cast<size_t>(row * ncols + col)];
        }
    }
    expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
}

static void run_live_kv_set_rows_flat_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t head_dim = 128;
    const int64_t nheads_kv = 8;
    const int64_t ncols = head_dim * nheads_kv;
    const int64_t kv_size = 256;
    const int64_t ntokens = 42;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * cache = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, ncols, kv_size);
    ggml_tensor * v_cur_base = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, nheads_kv, ntokens);
    ggml_tensor * v_cur = ggml_view_2d(ctx.get(), v_cur_base, ncols, ntokens, ncols * sizeof(float), 0);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, ntokens);
    ggml_tensor * out = ggml_set_rows(ctx.get(), cache, v_cur, rows);

    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> cache_zero(ggml_nbytes(cache), 0);
    std::vector<float> v_data(static_cast<size_t>(head_dim * nheads_kv * ntokens));
    std::vector<int64_t> row_data = { 0, 1 };
    std::vector<float> expected(static_cast<size_t>(ncols * kv_size), 0.0f);

    for (int64_t token = 0; token < ntokens; ++token) {
        for (int64_t head = 0; head < nheads_kv; ++head) {
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const size_t index = static_cast<size_t>(dim + head_dim * (head + nheads_kv * token));
                v_data[index] = static_cast<float>(static_cast<int>((dim * 17 + head * 31 + token * 47) % 251) - 125) / 113.0f;
                expected[static_cast<size_t>(dim + head_dim * head + ncols * row_data[static_cast<size_t>(token)])] =
                    ggml_fp16_to_fp32(ggml_fp32_to_fp16(v_data[index]));
            }
        }
    }

    ggml_backend_tensor_set(cache, cache_zero.data(), 0, cache_zero.size());
    ggml_backend_tensor_set(v_cur_base, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int64_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    expect_near(tensor_to_float(out), expected, 1.0e-3f, "live_kv_set_rows_flat");
}

static void run_live_kv_kqv_direct_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t head_dim = 128;
    const int64_t nheads_kv = 8;
    const int64_t nheads = 24;
    const int64_t kv_size = 256;
    const int64_t ntokens = 2;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, kv_size, head_dim, nheads_kv);
    ggml_tensor * probs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, kv_size, ntokens, nheads);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), v, probs);

    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> v_f32(static_cast<size_t>(kv_size * head_dim * nheads_kv), 0.0f);
    std::vector<uint16_t> v_f16(v_f32.size(), 0);
    std::vector<float> probs_data(static_cast<size_t>(kv_size * ntokens * nheads), 0.0f);
    std::vector<float> expected(static_cast<size_t>(head_dim * ntokens * nheads), 0.0f);

    for (int64_t kv_head = 0; kv_head < nheads_kv; ++kv_head) {
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            for (int64_t token = 0; token < ntokens; ++token) {
                // Keep non-selected tokens initialized too, so the test catches
                // wrong token indexing instead of accidentally reading zeros.
                const size_t index = static_cast<size_t>(token + kv_size * (dim + head_dim * kv_head));
                v_f32[index] = static_cast<float>(static_cast<int>((dim * 17 + kv_head * 31 + token * 47) % 251) - 125) / 113.0f;
                v_f16[index] = ggml_fp32_to_fp16(v_f32[index]);
            }
        }
    }
    for (int64_t head = 0; head < nheads; ++head) {
        for (int64_t col = 0; col < ntokens; ++col) {
            probs_data[static_cast<size_t>(col + kv_size * (col + ntokens * head))] = 1.0f;
            const int64_t kv_head = head / (nheads / nheads_kv);
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const size_t v_index = static_cast<size_t>(col + kv_size * (dim + head_dim * kv_head));
                expected[static_cast<size_t>(dim + head_dim * (col + ntokens * head))] =
                    ggml_fp16_to_fp32(v_f16[v_index]);
            }
        }
    }

    ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(probs, probs_data.data(), 0, probs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    expect_near(tensor_to_float(out), expected, 1.0e-3f, "live_kv_kqv_direct");
}

static void run_live_kv_set_rows_update_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t head_dim = 128;
    const int64_t nheads_kv = 8;
    const int64_t nheads = 24;
    const int64_t ncols = head_dim * nheads_kv;
    const int64_t kv_size = 256;
    const int64_t ntokens = 2;
    const int64_t out_cols = 2;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * cache = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, ncols, kv_size);
    ggml_tensor * v_cur_base = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, nheads_kv, ntokens);
    ggml_tensor * v_cur = ggml_view_2d(ctx.get(), v_cur_base, ncols, ntokens, ncols * sizeof(float), 0);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, ntokens);
    ggml_tensor * set = ggml_set_rows(ctx.get(), cache, v_cur, rows);

    // This mirrors llama_kv_cache::get_v for the non-transposed V cache used
    // without flash attention, followed by build_attn_mha's transpose+cont
    // materialization before the V * softmax(KQ) matmul.
    ggml_tensor * v_view = ggml_view_4d(
        ctx.get(), set,
        head_dim, nheads_kv, kv_size, 1,
        head_dim * sizeof(ggml_fp16_t),
        ncols * sizeof(ggml_fp16_t),
        ncols * kv_size * sizeof(ggml_fp16_t),
        0);
    ggml_tensor * v_perm = ggml_permute(ctx.get(), v_view, 0, 2, 1, 3);
    ggml_tensor * v_cont = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), v_perm));
    ggml_tensor * probs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, kv_size, out_cols, nheads);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), v_cont, probs);

    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> cache_zero(ggml_nbytes(cache), 0);
    std::vector<float> v_data(static_cast<size_t>(head_dim * nheads_kv * ntokens));
    std::vector<int64_t> row_data = { 0, 1 };
    std::vector<float> probs_data(static_cast<size_t>(kv_size * out_cols * nheads), 0.0f);
    std::vector<float> expected(static_cast<size_t>(head_dim * out_cols * nheads), 0.0f);

    for (int64_t token = 0; token < ntokens; ++token) {
        for (int64_t head = 0; head < nheads_kv; ++head) {
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const size_t index = static_cast<size_t>(dim + head_dim * (head + nheads_kv * token));
                v_data[index] = static_cast<float>(static_cast<int>((dim * 17 + head * 31 + token * 47) % 251) - 125) / 113.0f;
            }
        }
    }
    for (int64_t head = 0; head < nheads; ++head) {
        for (int64_t col = 0; col < out_cols; ++col) {
            probs_data[static_cast<size_t>(col + kv_size * (col + out_cols * head))] = 1.0f;
            const int64_t kv_head = head / (nheads / nheads_kv);
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                expected[static_cast<size_t>(dim + head_dim * (col + out_cols * head))] =
                    ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        v_data[static_cast<size_t>(dim + head_dim * (kv_head + nheads_kv * col))]));
            }
        }
    }

    ggml_backend_tensor_set(cache, cache_zero.data(), 0, cache_zero.size());
    ggml_backend_tensor_set(v_cur_base, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int64_t));
    ggml_backend_tensor_set(probs, probs_data.data(), 0, probs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    expect_near(tensor_to_float(out), expected, 1.0e-3f, "live_kv_set_rows_kqv_readback");
}

static void run_catalog_cont_set_rows_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t cont_ncols = catalog_shape_or(cfg, "cont_ncols", 128);
    const int64_t cont_rows = catalog_shape_or(cfg, "cont_rows", 2);
    const int64_t set_rows_ncols = catalog_shape(cfg, "ncols");
    const int64_t rows_to_set = cont_ncols * cont_rows / set_rows_ncols;
    const int64_t dst_rows = std::max<int64_t>(catalog_shape_or(cfg, "dst_rows", rows_to_set + 2), rows_to_set + 2);
    GGML_ASSERT((cont_ncols * cont_rows) % set_rows_ncols == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cont_ncols + 1, cont_rows);
    ggml_tensor * view = ggml_view_2d(ctx.get(), base, cont_ncols, cont_rows, base->nb[1], sizeof(float));
    ggml_tensor * cont = ggml_cont(ctx.get(), view);
    ggml_tensor * rows_src = ggml_reshape_2d(ctx.get(), cont, set_rows_ncols, rows_to_set);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, rows_to_set);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, set_rows_ncols, dst_rows);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, rows_src, rows);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> base_data(static_cast<size_t>((cont_ncols + 1) * cont_rows));
    std::vector<int64_t> row_data(static_cast<size_t>(rows_to_set));
    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    for (size_t i = 0; i < base_data.size(); ++i) {
        base_data[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 101) - 50) * 0.015625f;
    }
    for (int64_t row = 0; row < rows_to_set; ++row) {
        row_data[static_cast<size_t>(row)] = row + 1;
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int64_t));
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> cont_expected;
    cont_expected.reserve(static_cast<size_t>(cont_ncols * cont_rows));
    for (int64_t row = 0; row < cont_rows; ++row) {
        for (int64_t col = 1; col <= cont_ncols; ++col) {
            cont_expected.push_back(base_data[static_cast<size_t>(row * (cont_ncols + 1) + col)]);
        }
    }
    std::vector<float> expected(static_cast<size_t>(set_rows_ncols * dst_rows), 0.0f);
    for (int64_t row = 0; row < rows_to_set; ++row) {
        const int64_t dst_row = row_data[static_cast<size_t>(row)];
        for (int64_t col = 0; col < set_rows_ncols; ++col) {
            const float value = cont_expected[static_cast<size_t>(row * set_rows_ncols + col)];
            expected[static_cast<size_t>(dst_row * set_rows_ncols + col)] =
                ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        }
    }
    expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_quantize_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nrows = catalog_shape(cfg, "nrows");
    GGML_ASSERT(ncols % ggml_blck_size(GGML_TYPE_Q8_1) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_1, ncols, nrows);
    ggml_tensor * out = ggml_cpy(ctx.get(), src, dst);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nrows));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 13 + 3) % 101) - 50) * 0.03125f;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q8_1);
    GGML_ASSERT(traits->from_float_ref != nullptr);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_1, ncols);
    std::vector<uint8_t> expected(row_bytes * static_cast<size_t>(nrows), 0);
    for (int64_t row = 0; row < nrows; ++row) {
        traits->from_float_ref(
            src_data.data() + static_cast<size_t>(row * ncols),
            expected.data() + row_bytes * static_cast<size_t>(row),
            ncols);
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> actual(ggml_nbytes(out));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size());
    GGML_ASSERT(actual == expected);
}

static std::vector<float> run_catalog_rope_cpu_reference(
        const std::vector<float> & src_data,
        const std::vector<int32_t> & pos_data,
        const std::vector<float> & freq_data,
        int64_t ncols,
        int64_t nheads,
        int64_t ntokens,
        int64_t n_dims,
        int mode,
        float freq_scale,
        float freq_base = 10000.0f) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, nheads, ntokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ntokens);
    ggml_tensor * freq = freq_data.empty() ? nullptr :
        ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, std::max<int64_t>(1, n_dims / 2));
    ggml_tensor * out = ggml_rope_ext(ctx.get(), src, pos, freq, static_cast<int>(n_dims), mode, 0, freq_base, freq_scale, 0.0f, 1.0f, 1.0f, 1.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(cpu_backend != nullptr);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu_backend.get()));
    GGML_ASSERT(buffer != nullptr);
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    if (freq) {
        ggml_backend_tensor_set(freq, freq_data.data(), 0, freq_data.size() * sizeof(float));
    }
    GGML_ASSERT(ggml_backend_graph_compute(cpu_backend.get(), graph) == GGML_STATUS_SUCCESS);
    return tensor_to_float(out);
}

static void run_catalog_rope_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nheads = catalog_shape(cfg, "nheads");
    const int64_t ntokens = catalog_shape(cfg, "ntokens");
    const int64_t n_dims = catalog_shape(cfg, "n_dims");
    const float freq_scale = static_cast<float>(catalog_shape_or(cfg, "freq_scale", 1));
    const int mode = catalog_support(cfg, "mode") == "NEOX" ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;
    const bool has_freq = catalog_support(cfg, "freq_factor") == "src2";

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, nheads, ntokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ntokens);
    ggml_tensor * freq = has_freq ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, std::max<int64_t>(1, n_dims / 2)) : nullptr;
    ggml_tensor * out = ggml_rope_ext(ctx.get(), src, pos, freq, static_cast<int>(n_dims), mode, 0, 10000.0f, freq_scale, 0.0f, 1.0f, 1.0f, 1.0f);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nheads * ntokens));
    std::vector<int32_t> pos_data(static_cast<size_t>(ntokens));
    std::vector<float> freq_data(has_freq ? static_cast<size_t>(std::max<int64_t>(1, n_dims / 2)) : 0);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 11 + 5) % 59) - 29) * 0.015625f;
    }
    for (int64_t i = 0; i < ntokens; ++i) {
        pos_data[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    for (size_t i = 0; i < freq_data.size(); ++i) {
        freq_data[i] = std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(n_dims));
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    if (freq) {
        ggml_backend_tensor_set(freq, freq_data.data(), 0, freq_data.size() * sizeof(float));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(
        tensor_to_float(out),
        run_catalog_rope_cpu_reference(src_data, pos_data, freq_data, ncols, nheads, ntokens, n_dims, mode, freq_scale),
        cfg.tolerance, cfg.id.c_str());
}

static void run_live_rope_normal_freq_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t ncols = 128;
    const int64_t nheads = 24;
    const int64_t ntokens = 2;
    const int64_t n_dims = 128;
    const int mode = GGML_ROPE_TYPE_NORMAL;
    const float freq_scale = 1.0f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, nheads, ntokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ntokens);
    ggml_tensor * freq = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_dims / 2);
    ggml_tensor * out = ggml_rope_ext(
        ctx.get(), src, pos, freq, static_cast<int>(n_dims), mode, 0,
        10000.0f, freq_scale, 0.0f, 1.0f, 1.0f, 1.0f);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nheads * ntokens));
    std::vector<int32_t> pos_data(static_cast<size_t>(ntokens));
    // ggml_rope_ext src2 is a correction factor tensor, not precomputed
    // inverse frequencies. Use a smooth non-uniform profile so this catches
    // src2 indexing and theta/factor arithmetic bugs that all-ones factors
    // would hide.
    std::vector<float> freq_data(static_cast<size_t>(n_dims / 2));
    for (size_t i = 0; i < freq_data.size(); ++i) {
        const float t = freq_data.size() == 1 ? 0.0f : static_cast<float>(i) / static_cast<float>(freq_data.size() - 1);
        freq_data[i] = 1.0f + 7.0f * t;
    }
    for (int64_t i = 0; i < ntokens; ++i) {
        pos_data[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 101) - 50) * 0.03125f;
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(freq, freq_data.data(), 0, freq_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(
        tensor_to_float(out),
        run_catalog_rope_cpu_reference(src_data, pos_data, freq_data, ncols, nheads, ntokens, n_dims, mode, freq_scale),
        2.0e-5f, "live_rope_normal_freq");
}

static void run_live_rope_normal_freq_inplace_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t ncols = 128;
    const int64_t nheads = 24;
    const int64_t ntokens = 42;
    const int64_t n_dims = 128;
    const int mode = GGML_ROPE_TYPE_NORMAL;
    const float freq_scale = 1.0f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, nheads, ntokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ntokens);
    ggml_tensor * freq = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_dims / 2);
    // ggml's graph allocator is allowed to place an out-of-place ROPE result
    // over a dead producer. Use the explicit in-place API here so the test
    // always exercises the same src0/dst aliasing contract, independent of
    // allocator heuristics.
    ggml_tensor * out = ggml_rope_ext_inplace(
        ctx.get(), src, pos, freq, static_cast<int>(n_dims), mode, 0,
        10000.0f, freq_scale, 0.0f, 1.0f, 1.0f, 1.0f);
    GGML_ASSERT(out->view_src == src);
    GGML_ASSERT(out->view_offs == 0);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(out->buffer == src->buffer);
    GGML_ASSERT(out->data == src->data);

    std::vector<float> src_data(static_cast<size_t>(ncols * nheads * ntokens));
    std::vector<int32_t> pos_data(static_cast<size_t>(ntokens));
    std::vector<float> freq_data(static_cast<size_t>(n_dims / 2));
    for (size_t i = 0; i < freq_data.size(); ++i) {
        const float t = freq_data.size() == 1 ? 0.0f : static_cast<float>(i) / static_cast<float>(freq_data.size() - 1);
        freq_data[i] = 1.0f + 7.0f * t;
    }
    for (int64_t i = 0; i < ntokens; ++i) {
        pos_data[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 101) - 50) * 0.03125f;
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(freq, freq_data.data(), 0, freq_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(
        tensor_to_float(out),
        run_catalog_rope_cpu_reference(src_data, pos_data, freq_data, ncols, nheads, ntokens, n_dims, mode, freq_scale),
        2.0e-5f, "live_rope_normal_freq_inplace");
}

static void run_live_rope_normal_freq_reused_producer_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t ncols = 128;
    const int64_t nheads = 24;
    const int64_t ntokens = 42;
    const int64_t n_dims = 128;
    const int mode = GGML_ROPE_TYPE_NORMAL;
    const float freq_scale = 1.0f;
    const float producer_scale = 1.25f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, nheads, ntokens);
    ggml_tensor * produced = ggml_scale(ctx.get(), src, producer_scale);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ntokens);
    ggml_tensor * freq = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_dims / 2);
    // This mirrors the live Q path more closely than ggml_rope_ext_inplace:
    // ROPE is logically out-of-place, but ggml's graph allocator may reuse the
    // dead producer storage for the ROPE result. The HRX kernel must therefore
    // be correct when src0 and dst bindings are the same allocation.
    ggml_tensor * out = ggml_rope_ext(
        ctx.get(), produced, pos, freq, static_cast<int>(n_dims), mode, 0,
        10000.0f, freq_scale, 0.0f, 1.0f, 1.0f, 1.0f);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, produced));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nheads * ntokens));
    std::vector<int32_t> pos_data(static_cast<size_t>(ntokens));
    std::vector<float> freq_data(static_cast<size_t>(n_dims / 2));
    for (size_t i = 0; i < freq_data.size(); ++i) {
        const float t = freq_data.size() == 1 ? 0.0f : static_cast<float>(i) / static_cast<float>(freq_data.size() - 1);
        freq_data[i] = 1.0f + 7.0f * t;
    }
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 101) - 50) * 0.03125f;
    }

    std::vector<float> produced_data(src_data.size());
    for (size_t i = 0; i < src_data.size(); ++i) {
        produced_data[i] = src_data[i] * producer_scale;
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(freq, freq_data.data(), 0, freq_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(
        tensor_to_float(out),
        run_catalog_rope_cpu_reference(produced_data, pos_data, freq_data, ncols, nheads, ntokens, n_dims, mode, freq_scale),
        2.0e-5f, "live_rope_normal_freq_reused_producer");
}

static void run_live_rope_normal_freq_reused_producer_sched_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t ncols = 128;
    const int64_t nheads = 24;
    const int64_t n_dims = 128;
    const int mode = GGML_ROPE_TYPE_NORMAL;
    const float freq_scale = 1.0f;
    const float freq_base = 500000.0f;
    const float producer_scale = 1.25f;
    const std::vector<float> freq_data = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.23493707f, 1.65132928f, 2.27665949f,
        3.29226279f, 5.171597f, 9.66673088f, 32.0f, 32.0f, 32.0f,
        32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f,
        32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f,
        32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f, 32.0f,
        32.0f, 32.0f,
    };

    auto run_case = [&](int64_t ntokens, int32_t pos_base, const char * label) {
        ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        GGML_ASSERT(cpu_backend != nullptr);
        ggml_backend_t sched_backends[] = { backend, cpu_backend.get() };
        ggml_backend_sched_t sched = ggml_backend_sched_new(sched_backends, nullptr, 2, 32, false, true);
        GGML_ASSERT(sched != nullptr);

        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, nheads, ntokens);
        ggml_tensor * produced = ggml_scale(ctx.get(), src, producer_scale);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ntokens);
        ggml_tensor * freq = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_dims / 2);
        ggml_tensor * out = ggml_rope_ext(
            ctx.get(), produced, pos, freq, static_cast<int>(n_dims), mode, 0,
            freq_base, freq_scale, 0.0f, 1.0f, 1.0f, 1.0f);
        GGML_ASSERT(ggml_backend_dev_supports_op(dev, produced));
        GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, out);
        ggml_backend_sched_set_tensor_backend(sched, src, backend);
        ggml_backend_sched_set_tensor_backend(sched, pos, backend);
        ggml_backend_sched_set_tensor_backend(sched, freq, backend);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));

        std::vector<float> src_data(static_cast<size_t>(ncols * nheads * ntokens));
        std::vector<int32_t> pos_data(static_cast<size_t>(ntokens));
        for (int64_t i = 0; i < ntokens; ++i) {
            pos_data[static_cast<size_t>(i)] = pos_base + static_cast<int32_t>(i);
        }
        for (size_t i = 0; i < src_data.size(); ++i) {
            src_data[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 101) - 50) * 0.03125f;
        }

        std::vector<float> produced_data(src_data.size());
        for (size_t i = 0; i < src_data.size(); ++i) {
            produced_data[i] = src_data[i] * producer_scale;
        }

        ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
        ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(freq, freq_data.data(), 0, freq_data.size() * sizeof(float));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

        expect_near(
            tensor_to_float(out),
            run_catalog_rope_cpu_reference(produced_data, pos_data, freq_data, ncols, nheads, ntokens, n_dims, mode, freq_scale, freq_base),
            2.0e-5f, label);
        ggml_backend_sched_free(sched);
    };

    run_case(42, 0, "live_rope_normal_freq_reused_producer_sched_prompt");
    run_case(2, 42, "live_rope_normal_freq_reused_producer_sched_pair");
    run_case(1, 44, "live_rope_normal_freq_reused_producer_sched_decode");
}

static void run_catalog_rope_set_rows_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nheads = catalog_shape(cfg, "nheads");
    const int64_t ntokens = catalog_shape(cfg, "ntokens");
    const int64_t n_dims = catalog_shape(cfg, "n_dims");
    const int64_t dst_rows = catalog_shape_or(cfg, "dst_rows", ntokens + 2);
    const int mode = catalog_support(cfg, "mode") == "NEOX" ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, nheads, ntokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ntokens);
    ggml_tensor * freq = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, std::max<int64_t>(1, n_dims / 2));
    ggml_tensor * rope = ggml_rope_ext(
        ctx.get(), src, pos, freq, static_cast<int>(n_dims), mode, 0,
        10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    ggml_tensor * view = ggml_view_2d(ctx.get(), rope, ncols * nheads, ntokens, rope->nb[2], 0);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, ntokens);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, ncols * nheads, dst_rows);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, view, rows);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nheads * ntokens));
    std::vector<int32_t> pos_data(static_cast<size_t>(ntokens));
    std::vector<float> freq_data(static_cast<size_t>(std::max<int64_t>(1, n_dims / 2)));
    std::vector<int64_t> row_data(static_cast<size_t>(ntokens));
    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 19 + 7) % 71) - 35) * 0.015625f;
    }
    for (int64_t i = 0; i < ntokens; ++i) {
        pos_data[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        row_data[static_cast<size_t>(i)] = i + 1;
    }
    for (size_t i = 0; i < freq_data.size(); ++i) {
        freq_data[i] = std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(n_dims));
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(freq, freq_data.data(), 0, freq_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int64_t));
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const std::vector<float> rope_expected =
        run_catalog_rope_cpu_reference(src_data, pos_data, freq_data, ncols, nheads, ntokens, n_dims, mode, 1.0f);
    std::vector<float> expected(static_cast<size_t>(ncols * nheads * dst_rows), 0.0f);
    for (int64_t token = 0; token < ntokens; ++token) {
        const int64_t dst_row = row_data[static_cast<size_t>(token)];
        for (int64_t col = 0; col < ncols * nheads; ++col) {
            const float value = rope_expected[static_cast<size_t>(token * ncols * nheads + col)];
            expected[static_cast<size_t>(dst_row * ncols * nheads + col)] =
                ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        }
    }
    expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_cont_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nrows = catalog_shape(cfg, "nrows");
    ggml_context_ptr ctx = make_context();
    ggml_tensor * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols + 2, nrows);
    ggml_tensor * view = ggml_view_2d(ctx.get(), base, ncols, nrows, base->nb[1], sizeof(float));
    ggml_tensor * out = ggml_cont(ctx.get(), view);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> base_data(static_cast<size_t>((ncols + 2) * nrows));
    for (size_t i = 0; i < base_data.size(); ++i) {
        base_data[i] = static_cast<float>(static_cast<int>(i % 127) - 63) * 0.0625f;
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(ncols * nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 1; col <= ncols; ++col) {
            expected.push_back(base_data[static_cast<size_t>(row * (ncols + 2) + col)]);
        }
    }
    expect_eq(tensor_to_float(out), expected, cfg.id.c_str());
}

static void run_catalog_swiglu_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nrows = catalog_shape(cfg, "nrows");
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(static_cast<size_t>(ncols * nrows));
    std::vector<float> rhs_data(lhs_data.size());
    std::vector<float> expected(lhs_data.size());
    for (size_t i = 0; i < lhs_data.size(); ++i) {
        lhs_data[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        rhs_data[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25f;
        expected[i] = lhs_data[i] / (1.0f + std::exp(-lhs_data[i])) * rhs_data[i];
    }
    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_mul_mat_q4_k_swiglu_once(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const ggml_backend_hrx_test_case & cfg) {
    const int64_t k = catalog_shape(cfg, "k");
    const int64_t rows = catalog_shape(cfg, "rows");
    const int64_t cols = catalog_shape(cfg, "cols");
    GGML_ASSERT(k % ggml_blck_size(GGML_TYPE_Q4_K) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * x_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, k, rows);
    ggml_tensor * gate_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * x = ggml_mul_mat(ctx.get(), x_lhs, rhs);
    ggml_tensor * gate = ggml_mul_mat(ctx.get(), gate_lhs, rhs);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), x, gate);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> x_f32(static_cast<size_t>(k * rows));
    std::vector<float> gate_f32(x_f32.size());
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols));
    for (size_t i = 0; i < x_f32.size(); ++i) {
        x_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 5) % 101) - 50) / 67.0f;
        gate_f32[i] = static_cast<float>(static_cast<int>((i * 29 + 11) % 103) - 51) / 71.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 31 + 3) % 107) - 53) / 73.0f;
    }

    std::vector<float> x_reference;
    std::vector<float> gate_reference;
    std::vector<uint8_t> x_storage;
    std::vector<uint8_t> gate_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, k, rows, x_f32, x_reference, x_storage);
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, k, rows, gate_f32, gate_reference, gate_storage);
    ggml_backend_tensor_set(x_lhs, x_storage.data(), 0, x_storage.size());
    ggml_backend_tensor_set(gate_lhs, gate_storage.data(), 0, gate_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> x_expected = reference_mul_mat(x_reference, rhs_f32, k, rows, cols);
    std::vector<float> gate_expected = reference_mul_mat(gate_reference, rhs_f32, k, rows, cols);
    for (size_t i = 0; i < x_expected.size(); ++i) {
        x_expected[i] = x_expected[i] / (1.0f + std::exp(-x_expected[i])) * gate_expected[i];
    }
    expect_near(tensor_to_float(out), x_expected, cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_rms_norm_mul_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    static constexpr float eps = 1.0e-6f;
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nrows = catalog_shape(cfg, "nrows");
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ncols);
    ggml_tensor * rms = ggml_rms_norm(ctx.get(), src, eps);
    ggml_tensor * out = ggml_mul(ctx.get(), rms, weight);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nrows));
    std::vector<float> weight_data(static_cast<size_t>(ncols));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 17 + 11) % 47) - 23) / 41.0f;
    }
    for (int64_t col = 0; col < ncols; ++col) {
        weight_data[static_cast<size_t>(col)] = 0.5f + static_cast<float>(col % 13) * 0.0625f;
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = src_data[static_cast<size_t>(row * ncols + col)];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            expected[static_cast<size_t>(row * ncols + col)] =
                src_data[static_cast<size_t>(row * ncols + col)] * scale * weight_data[static_cast<size_t>(col)];
        }
    }
    expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_add_rms_norm_mul_once(ggml_backend_t backend, ggml_backend_dev_t dev, const ggml_backend_hrx_test_case & cfg) {
    static constexpr float eps = 1.0e-6f;
    const int64_t ncols = catalog_shape(cfg, "ncols");
    const int64_t nrows = catalog_shape(cfg, "nrows");
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ncols);
    ggml_tensor * add = ggml_add(ctx.get(), src0, src1);
    ggml_tensor * rms = ggml_rms_norm(ctx.get(), add, eps);
    ggml_tensor * out = ggml_mul(ctx.get(), rms, weight);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src0_data(static_cast<size_t>(ncols * nrows));
    std::vector<float> src1_data(src0_data.size());
    std::vector<float> weight_data(static_cast<size_t>(ncols));
    std::vector<float> added(src0_data.size());
    for (size_t i = 0; i < src0_data.size(); ++i) {
        src0_data[i] = static_cast<float>(static_cast<int>((i * 17 + 11) % 47) - 23) / 41.0f;
        src1_data[i] = static_cast<float>(static_cast<int>((i * 13 + 7) % 43) - 21) / 37.0f;
        added[i] = src0_data[i] + src1_data[i];
    }
    for (int64_t col = 0; col < ncols; ++col) {
        weight_data[static_cast<size_t>(col)] = 0.5f + static_cast<float>(col % 13) * 0.0625f;
    }
    ggml_backend_tensor_set(src0, src0_data.data(), 0, src0_data.size() * sizeof(float));
    ggml_backend_tensor_set(src1, src1_data.data(), 0, src1_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src0_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = added[static_cast<size_t>(row * ncols + col)];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            expected[static_cast<size_t>(row * ncols + col)] =
                added[static_cast<size_t>(row * ncols + col)] * scale * weight_data[static_cast<size_t>(col)];
        }
    }
    expect_near(tensor_to_float(out), expected, cfg.tolerance, cfg.id.c_str());
}

static void run_catalog_schedule_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const ggml_backend_hrx_test_case & cfg) {
    if (cfg.op == "MUL_MAT") {
        run_catalog_mul_mat_schedule_case(backend, dev, cfg);
    } else if (cfg.scenario == "rope_set_rows") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_rope_set_rows_once);
    } else if (cfg.scenario == "cont_set_rows") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_cont_set_rows_once);
    } else if (cfg.scenario == "add_rms_norm_mul") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_add_rms_norm_mul_once);
    } else if (cfg.scenario == "rms_norm_mul") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_rms_norm_mul_once);
    } else if (cfg.scenario == "mul_mat_q4_k_swiglu") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_mul_mat_q4_k_swiglu_once);
    } else if (cfg.scenario == "mul_mat_f16_batched_cont") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_mul_mat_f16_batched_cont_once);
    } else if (cfg.scenario == "softmax_kqv") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_softmax_kqv_once);
    } else if (cfg.op == "ADD" || cfg.op == "MUL" || cfg.op == "DIV") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_pointwise_once);
    } else if (cfg.op == "SCALE" || cfg.op == "CLAMP" || cfg.op == "RMS_NORM" ||
               cfg.op == "SUM_ROWS" || cfg.op == "SOFT_MAX") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_unary_once);
    } else if (cfg.op == "ARGSORT") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_argsort_once);
    } else if (cfg.op == "GET_ROWS") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_get_rows_once);
    } else if (cfg.op == "MUL_MAT_ID") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_mul_mat_id_once);
    } else if (cfg.op == "SET_ROWS") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_set_rows_once);
    } else if (cfg.op == "QUANTIZE") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_quantize_once);
    } else if (cfg.op == "ROPE" || cfg.op == "ROPE_SCALE") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_rope_once);
    } else if (cfg.op == "CONT") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_cont_once);
    } else if (cfg.op == "CPY") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_copy_once);
    } else if (cfg.op == "GLU") {
        run_catalog_checked_case(backend, dev, cfg, run_catalog_swiglu_once);
    } else {
        std::fprintf(stderr, "unsupported catalog test op %s for case %s\n", cfg.op.c_str(), cfg.id.c_str());
        std::abort();
    }
}

static void run_mul_mat_id_q4_k_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs,
        float tolerance,
        const char * label,
        bool sampled_reference = false) {
    GGML_ASSERT(k % ggml_blck_size(GGML_TYPE_Q4_K) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, k, rows, n_experts);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, broadcast_rhs ? 1 : n_ids, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_ids, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx.get(), lhs, rhs, ids);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows * n_experts));
    std::vector<float> rhs_f32(static_cast<size_t>(k * (broadcast_rhs ? 1 : n_ids) * n_tokens));
    std::vector<int32_t> ids_i32(static_cast<size_t>(n_ids * n_tokens));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 29 + 17) % 127) - 63) / 53.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 31 + 5) % 109) - 54) / 59.0f;
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id_pos = 0; id_pos < n_ids; ++id_pos) {
            ids_i32[static_cast<size_t>(id_pos + n_ids * token)] =
                static_cast<int32_t>((id_pos * 3 + token * 5) % n_experts);
        }
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, k, rows * n_experts, lhs_f32, lhs_reference, lhs_storage);

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    if (sampled_reference) {
        expect_mul_mat_id_q4_k_samples(
            tensor_to_float(out),
            lhs_reference,
            rhs_f32,
            ids_i32,
            k,
            rows,
            n_ids,
            n_tokens,
            n_experts,
            broadcast_rhs,
            tolerance,
            label);
        return;
    }

    expect_near(
        tensor_to_float(out),
        reference_mul_mat_id_q4_k(
            lhs_reference, rhs_f32, ids_i32, k, rows, n_ids, n_tokens, n_experts, broadcast_rhs),
        tolerance, label);
}

static size_t index_4d(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t ne0, int64_t ne1, int64_t ne2) {
    return static_cast<size_t>(i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3)));
}

static std::vector<float> reference_flash_attn_ext_decode(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & mask,
        const std::vector<float> & sinks,
        int64_t d,
        int64_t n,
        int64_t h,
        int64_t h_kv,
        int64_t kv,
        int64_t s,
        float scale,
        bool has_mask,
        bool has_sinks) {
    std::vector<float> output(static_cast<size_t>(d * h * n * s), 0.0f);
    std::vector<float> logits(static_cast<size_t>(kv));
    for (int64_t seq = 0; seq < s; ++seq) {
        for (int64_t token = 0; token < n; ++token) {
            for (int64_t head = 0; head < h; ++head) {
                const int64_t kv_head = head / (h / h_kv);
                float max_score = has_sinks ? sinks[static_cast<size_t>(head)] : -INFINITY;
                for (int64_t t = 0; t < kv; ++t) {
                    float score = 0.0f;
                    for (int64_t col = 0; col < d; ++col) {
                        score +=
                            q[index_4d(col, token, head, seq, d, n, h)] *
                            k[index_4d(col, t, kv_head, seq, d, kv, h_kv)];
                    }
                    score *= scale;
                    if (has_mask) {
                        score += mask[index_4d(t, token, 0, seq, kv, n, 1)];
                    }
                    logits[static_cast<size_t>(t)] = score;
                    max_score = std::max(max_score, score);
                }

                float sum = has_sinks ? std::exp(sinks[static_cast<size_t>(head)] - max_score) : 0.0f;
                for (int64_t t = 0; t < kv; ++t) {
                    logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_score);
                    sum += logits[static_cast<size_t>(t)];
                }

                for (int64_t col = 0; col < d; ++col) {
                    float value = 0.0f;
                    for (int64_t t = 0; t < kv; ++t) {
                        value +=
                            logits[static_cast<size_t>(t)] *
                            v[index_4d(col, t, kv_head, seq, d, kv, h_kv)];
                    }
                    output[index_4d(col, head, token, seq, d, h, n)] = value / sum;
                }
            }
        }
    }
    return output;
}

static void run_add_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(n);
    std::vector<float> rhs_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs_data[i] = static_cast<float>(i % 17) - 8.0f;
        rhs_data[i] = static_cast<float>(i % 11) * 0.5f;
        expected[i] = lhs_data[i] + rhs_data[i];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ADD graph failed for n=%" PRId64 ": %s\n", n, ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(sum, actual.data(), 0, actual.size() * sizeof(float));
    expect_eq(actual, expected, "add");
}

static void run_add_activation_boundary_case(ggml_backend_dev_t dev) {
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, sum));
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 257, 3, 2, 1);
        ggml_tensor * rhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, 3, 1, 1);
        ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);
        GGML_ASSERT(ggml_backend_dev_supports_op(dev, sum));
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
        ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, sum));
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        ggml_tensor * rhs_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        ggml_tensor * rhs = ggml_transpose(ctx.get(), rhs_base);
        GGML_ASSERT(rhs->ne[0] == lhs->ne[0]);
        GGML_ASSERT(rhs->ne[1] == lhs->ne[1]);
        GGML_ASSERT(rhs->nb[0] != sizeof(float));
        ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, sum));
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
        ggml_tensor * lhs = ggml_view_2d(ctx.get(), lhs_base, 4, 4, 2 * lhs_base->nb[1], 0);
        ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_is_contiguous(lhs));
        GGML_ASSERT(ggml_backend_dev_supports_op(dev, sum));
    }
}

static void run_mul_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * product = ggml_mul(ctx.get(), lhs, rhs);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, product);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(n);
    std::vector<float> rhs_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs_data[i] = static_cast<float>(i % 17) - 8.0f;
        rhs_data[i] = static_cast<float>(i % 11) * 0.25f;
        expected[i] = lhs_data[i] * rhs_data[i];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "MUL graph failed for n=%" PRId64 ": %s\n", n, ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(product, actual.data(), 0, actual.size() * sizeof(float));
    expect_eq(actual, expected, "mul");
}

static void run_broadcast_case(ggml_backend_t backend, enum ggml_op op) {
    static constexpr int64_t ne0 = 257;
    static constexpr int64_t ne1 = 3;
    static constexpr int64_t ne2 = 2;
    static constexpr int64_t ne3 = 1;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_tensor * rhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, ne1, 1, ne3);
    ggml_tensor * out = nullptr;
    if (op == GGML_OP_ADD) {
        out = ggml_add(ctx.get(), lhs, rhs);
    } else if (op == GGML_OP_MUL) {
        out = ggml_mul(ctx.get(), lhs, rhs);
    } else {
        out = ggml_div(ctx.get(), lhs, rhs);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t lhs_count = ne0 * ne1 * ne2 * ne3;
    const int64_t rhs_count = ne1 * ne3;
    std::vector<float> lhs_data(lhs_count);
    std::vector<float> rhs_data(rhs_count);
    std::vector<float> expected(lhs_count);
    for (int64_t i = 0; i < lhs_count; ++i) {
        lhs_data[i] = static_cast<float>(i % 19) - 9.0f;
    }
    for (int64_t i = 0; i < rhs_count; ++i) {
        rhs_data[i] = static_cast<float>(i + 2);
    }
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const int64_t lhs_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;
                    const float a = lhs_data[lhs_idx];
                    const float b = rhs_data[i3 * ne1 + i1];
                    expected[lhs_idx] = op == GGML_OP_ADD ? a + b : (op == GGML_OP_MUL ? a * b : a / b);
                }
            }
        }
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "broadcast graph failed for op=%d: %s\n", op, ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> actual(lhs_count, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "broadcast");
}

static void run_scale_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 13) - 6.0f;
        expected[i] = src_data[i] * 1.25f - 0.75f;
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "scale");
}

static void run_unary_case(ggml_backend_t backend, enum ggml_unary_op op, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_unary(ctx.get(), src, op);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 17) * 0.25f - 2.0f;
        const float x = src_data[i];
        if (op == GGML_UNARY_OP_SILU) {
            expected[i] = x / (1.0f + std::exp(-x));
        } else if (op == GGML_UNARY_OP_SIGMOID) {
            expected[i] = 1.0f / (1.0f + std::exp(-x));
        } else {
            expected[i] = std::log1p(std::exp(x));
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-4f, ggml_unary_op_name(op));
}

static void run_swiglu_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), lhs, rhs);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(n);
    std::vector<float> rhs_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs_data[i] = static_cast<float>(i % 17) * 0.25f - 2.0f;
        rhs_data[i] = static_cast<float>(i % 7) - 3.0f;
        expected[i] = lhs_data[i] / (1.0f + std::exp(-lhs_data[i])) * rhs_data[i];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-4f, "swiglu");
}

static void run_clamp_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 17) - 8.0f;
        expected[i] = src_data[i] < -1.5f ? -1.5f : (src_data[i] > 2.0f ? 2.0f : src_data[i]);
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_eq(actual, expected, "clamp");
}

static void run_cpy_strided_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 6, 4);
    ggml_tensor * view = ggml_view_2d(ctx.get(), base, 3, 4, base->nb[1], sizeof(float));
    ggml_tensor * target = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 4);
    ggml_tensor * out = ggml_cpy(ctx.get(), view, target);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> base_data(24);
    for (size_t i = 0; i < base_data.size(); ++i) {
        base_data[i] = static_cast<float>(i);
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    for (int row = 0; row < 4; ++row) {
        for (int col = 1; col < 4; ++col) {
            expected.push_back(base_data[row * 6 + col]);
        }
    }
    expect_eq(tensor_to_float(out), expected, "cpy_strided");
}

static void run_cpy_f32_f16_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 257);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 257);
    ggml_tensor * out = ggml_cpy(ctx.get(), src, dst);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(257);
    std::vector<float> expected(257);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125f;
        expected[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(src_data[i]));
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_eq(tensor_to_float(out), expected, "cpy_f32_f16");
}

static void run_cont_slice_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 7, 3);
    ggml_tensor * view = ggml_view_2d(ctx.get(), base, 4, 3, base->nb[1], 2 * sizeof(float));
    ggml_tensor * out = ggml_cont(ctx.get(), view);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> base_data(21);
    for (size_t i = 0; i < base_data.size(); ++i) {
        base_data[i] = static_cast<float>(100 + i);
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    for (int row = 0; row < 3; ++row) {
        for (int col = 2; col < 6; ++col) {
            expected.push_back(base_data[row * 7 + col]);
        }
    }
    expect_eq(tensor_to_float(out), expected, "cont_slice");
}

static void run_get_rows_case(ggml_backend_t backend, int64_t rows_to_get) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 5, 6, 2);
    ggml_tensor * rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, rows_to_get, 2);
    ggml_tensor * out = ggml_get_rows(ctx.get(), src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(5 * 6 * 2);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(i);
    }
    std::vector<int32_t> row_data(rows_to_get * 2);
    for (int64_t batch = 0; batch < 2; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            row_data[batch * rows_to_get + row] = static_cast<int32_t>((row * 2 + batch) % 6);
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    for (int64_t batch = 0; batch < 2; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            const int64_t selected = row_data[batch * rows_to_get + row];
            for (int64_t col = 0; col < 5; ++col) {
                expected.push_back(src_data[batch * 30 + selected * 5 + col]);
            }
        }
    }
    expect_eq(tensor_to_float(out), expected, rows_to_get == 1 ? "get_rows_nr1" : "get_rows");
}

static void run_scale_get_rows_fusion_case(ggml_backend_t backend) {
    constexpr int64_t cols = 257;
    constexpr int64_t rows = 4;
    constexpr float scale_value = 0.25f;
    constexpr float bias_value = -0.125f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * scaled = ggml_scale_bias_inplace(ctx.get(), src, scale_value, bias_value);
    ggml_tensor * out = ggml_get_rows(ctx.get(), scaled, idx);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(cols * rows));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = 0.01f * static_cast<float>(static_cast<int>(i % 101) - 50);
    }
    const int32_t row_index = 2;
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(idx, &row_index, 0, sizeof(row_index));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(cols));
    for (int64_t col = 0; col < cols; ++col) {
        expected[static_cast<size_t>(col)] =
            src_data[static_cast<size_t>(row_index) * cols + static_cast<size_t>(col)] * scale_value + bias_value;
    }
    expect_near(tensor_to_float(out), expected, 1.0e-6f, "scale_get_rows_fusion");
}

static void run_get_rows_q5_k_case(
        ggml_backend_t backend,
        int64_t ncols,
        int64_t src_rows,
        int64_t rows_to_get,
        int64_t batches) {
    const int64_t block_size = ggml_blck_size(GGML_TYPE_Q5_K);
    GGML_ASSERT(ncols % block_size == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q5_K, ncols, src_rows, batches);
    ggml_tensor * rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, rows_to_get, batches);
    ggml_tensor * out = ggml_get_rows(ctx.get(), src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_K, ncols);
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q5_K);
    GGML_ASSERT(traits->from_float_ref != nullptr);
    GGML_ASSERT(traits->to_float != nullptr);
    std::vector<float> src_f32(static_cast<size_t>(ncols * src_rows * batches));
    std::vector<float> src_dequant(src_f32.size());
    std::vector<uint8_t> src_q5(row_bytes * static_cast<size_t>(src_rows * batches));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < src_rows; ++row) {
            const int64_t row_offset = (batch * src_rows + row) * ncols;
            for (int64_t col = 0; col < ncols; ++col) {
                const int64_t i = row_offset + col;
                src_f32[static_cast<size_t>(i)] =
                    static_cast<float>(static_cast<int>((i * 17 + batch * 13 + row * 7) % 97) - 48) / 19.0f;
            }
            uint8_t * q5_row = src_q5.data() + row_bytes * static_cast<size_t>(batch * src_rows + row);
            traits->from_float_ref(src_f32.data() + row_offset, q5_row, ncols);
            traits->to_float(q5_row, src_dequant.data() + row_offset, ncols);
        }
    }

    std::vector<int32_t> row_data(static_cast<size_t>(rows_to_get * batches));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            row_data[static_cast<size_t>(batch * rows_to_get + row)] =
                static_cast<int32_t>((row * 3 + batch + 1) % src_rows);
        }
    }

    ggml_backend_tensor_set(src, src_q5.data(), 0, src_q5.size());
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(ncols * rows_to_get * batches));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            const int64_t selected = row_data[static_cast<size_t>(batch * rows_to_get + row)];
            const int64_t row_offset = (batch * src_rows + selected) * ncols;
            expected.insert(
                expected.end(),
                src_dequant.begin() + row_offset,
                src_dequant.begin() + row_offset + ncols);
        }
    }
    expect_eq(tensor_to_float(out), expected, "get_rows_q5_k");
}

static void run_concat_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_tensor * out = ggml_concat(ctx.get(), lhs, rhs, 0);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> lhs_data = { 1, 2, 3, 4, 5, 6 };
    const std::vector<float> rhs_data = { 10, 11, 12, 13 };
    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_eq(tensor_to_float(out), { 1, 2, 3, 10, 11, 4, 5, 6, 12, 13 }, "concat");
}

static void run_set_rows_case(ggml_backend_t backend, ggml_type type) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), type, 32, 4);
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 2);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 2);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    std::vector<float> src_data(64);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    const int64_t row_data[2] = { 1, 3 };

    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data, 0, sizeof(row_data));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const std::vector<float> actual = tensor_to_float(out);
    const float tolerance = type == GGML_TYPE_Q4_0 ? 0.15f : (type == GGML_TYPE_Q8_0 ? 0.01f : 0.0f);
    for (int row = 0; row < 4; ++row) {
        const int src_row = row == 1 ? 0 : (row == 3 ? 1 : -1);
        for (int col = 0; col < 32; ++col) {
            const float expected = src_row >= 0 ? src_data[src_row * 32 + col] : 0.0f;
            const float got = actual[row * 32 + col];
            if (std::fabs(got - expected) > tolerance) {
                std::fprintf(stderr, "set_rows(%s)[%d,%d]: got %.9g expected %.9g tolerance %.9g\n",
                    ggml_type_name(type), row, col, got, expected, tolerance);
                std::abort();
            }
        }
    }
}

struct test_block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[32];
};

static_assert(sizeof(test_block_q8_0) == sizeof(ggml_fp16_t) + 32, "unexpected q8_0 test block size");

static void run_set_rows_q8_0_tie_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 32, 2);
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    std::vector<float> src_data(32, 0.0f);
    src_data[0] = 127.0f;
    src_data[1] = 0.5f;
    src_data[2] = -0.5f;
    src_data[3] = 1.5f;
    src_data[4] = -1.5f;
    src_data[5] = 2.5f;
    src_data[6] = -2.5f;
    const int64_t row_data[1] = { 1 };

    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data, 0, sizeof(row_data));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> actual(ggml_nbytes(out));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size());
    const test_block_q8_0 * blocks = reinterpret_cast<const test_block_q8_0 *>(actual.data());
    const int8_t expected[] = { 127, 1, -1, 2, -2, 3, -3 };
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        if (blocks[1].qs[i] != expected[i]) {
            std::fprintf(stderr, "set_rows(q8_0 tie) qs[%zu]: got %d expected %d\n",
                i, static_cast<int>(blocks[1].qs[i]), static_cast<int>(expected[i]));
            std::abort();
        }
    }
}

static std::vector<float> rowwise_input(int64_t ncols, int64_t nrows) {
    std::vector<float> data(static_cast<size_t>(ncols * nrows));
    for (size_t i = 0; i < data.size(); ++i) {
        const int value = static_cast<int>((i * 17 + 11) % 29) - 14;
        data[i] = static_cast<float>(value) * 0.0625f;
    }
    return data;
}

static void run_rms_norm_case(ggml_backend_t backend, int64_t ncols, int64_t ne1, int64_t ne2) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2);
    constexpr float eps = 1.0e-6f;
    ggml_tensor * out = ggml_rms_norm(ctx.get(), src, eps);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    const std::vector<float> src_data = rowwise_input(ncols, nrows);
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = src_data[row * ncols + col];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            expected[row * ncols + col] = src_data[row * ncols + col] * scale;
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "rms_norm");
}

static void run_sum_rows_case(ggml_backend_t backend, int64_t ncols, int64_t ne1, int64_t ne2) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2);
    ggml_tensor * out = ggml_sum_rows(ctx.get(), src);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    const std::vector<float> src_data = rowwise_input(ncols, nrows);
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            sum += src_data[row * ncols + col];
        }
        expected[row] = sum;
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "sum_rows");
}

static void run_l2_norm_case(ggml_backend_t backend, int64_t ncols, int64_t ne1, int64_t ne2) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2);
    constexpr float eps = 1.0e-7f;
    ggml_tensor * out = ggml_l2_norm(ctx.get(), src, eps);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    const std::vector<float> src_data = rowwise_input(ncols, nrows);
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = src_data[row * ncols + col];
            sum += value * value;
        }
        const float denom = std::sqrt(sum);
        const float scale = 1.0f / (denom > eps ? denom : eps);
        for (int64_t col = 0; col < ncols; ++col) {
            expected[row * ncols + col] = src_data[row * ncols + col] * scale;
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "l2_norm");
}

static void run_soft_max_case(ggml_backend_t backend, int64_t ncols, bool mask) {
    static constexpr int64_t ne1 = 3;
    static constexpr int64_t ne2 = 2;
    static constexpr int64_t ne3 = 1;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2, ne3);
    ggml_tensor * mask_tensor = mask ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2, ne3) : nullptr;
    constexpr float scale = 0.25f;
    ggml_tensor * out = ggml_soft_max_ext(ctx.get(), src, mask_tensor, scale, 0.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2 * ne3;
    std::vector<float> src_data(static_cast<size_t>(ncols * nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            src_data[row * ncols + col] = static_cast<float>((row * 17 + col * 7) % 31) * 0.1f - 1.5f;
        }
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));

    std::vector<float> mask_data;
    if (mask_tensor) {
        mask_data.resize(static_cast<size_t>(ncols * ne1 * ne2 * ne3));
        for (int64_t row = 0; row < ne1 * ne2 * ne3; ++row) {
            for (int64_t col = 0; col < ncols; ++col) {
                mask_data[row * ncols + col] = static_cast<float>((row + col) % 5) * -0.125f;
            }
        }
        ggml_backend_tensor_set(mask_tensor, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        const int64_t i1 = row % ne1;
        const int64_t i2 = (row / ne1) % ne2;
        float max_value = -INFINITY;
        for (int64_t col = 0; col < ncols; ++col) {
            const float bias = mask_tensor ? mask_data[(i2 * ne1 + i1) * ncols + col] : 0.0f;
            max_value = std::max(max_value, src_data[row * ncols + col] * scale + bias);
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float bias = mask_tensor ? mask_data[(i2 * ne1 + i1) * ncols + col] : 0.0f;
            sum += std::exp(src_data[row * ncols + col] * scale + bias - max_value);
        }
        for (int64_t col = 0; col < ncols; ++col) {
            const float bias = mask_tensor ? mask_data[(i2 * ne1 + i1) * ncols + col] : 0.0f;
            expected[row * ncols + col] =
                std::exp(src_data[row * ncols + col] * scale + bias - max_value) / sum;
        }
    }
    expect_near(tensor_to_float(out), expected, 3.0e-6f, "soft_max");
}

static void run_argsort_case(ggml_backend_t backend, int64_t ncols, int64_t nrows, ggml_sort_order order) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_argsort(ctx.get(), src, order);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            src_data[row * ncols + col] =
                static_cast<float>((col * 37 + row * 11) % 257) + static_cast<float>(col) * 0.001f;
        }
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> expected(static_cast<size_t>(ncols * nrows));
    std::vector<int32_t> indices(static_cast<size_t>(ncols));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            indices[col] = static_cast<int32_t>(col);
        }
        std::sort(indices.begin(), indices.end(), [&](int32_t a, int32_t b) {
            const float av = src_data[row * ncols + a];
            const float bv = src_data[row * ncols + b];
            return order == GGML_SORT_ORDER_ASC ? av < bv : av > bv;
        });
        std::copy(indices.begin(), indices.end(), expected.begin() + row * ncols);
    }
    const std::vector<int32_t> actual = tensor_to_i32(out);
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "argsort[%zu]: got %" PRId32 " expected %" PRId32 "\n", i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void run_topk_moe_case(
        ggml_backend_t backend,
        int64_t n_experts,
        int64_t n_tokens,
        int64_t n_expert_used,
        bool with_norm,
        const char * label) {
    ggml_context_ptr ctx = make_context();
    int64_t ne[4] = { n_experts, n_tokens, 1, 1 };
    ggml_tensor * logits = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne);
    ggml_tensor * probs = ggml_soft_max(ctx.get(), logits);
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx.get(), probs, n_expert_used);
    ggml_tensor * weights = ggml_get_rows(
        ctx.get(), ggml_reshape_3d(ctx.get(), probs, 1, n_experts, n_tokens), selected_experts);
    if (with_norm) {
        weights = ggml_reshape_2d(ctx.get(), weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx.get(), weights);
        weights_sum = ggml_clamp(
            ctx.get(), weights_sum, 6.103515625e-5f, std::numeric_limits<float>::infinity());
        weights = ggml_div(ctx.get(), weights, weights_sum);
        weights = ggml_reshape_3d(ctx.get(), weights, 1, n_expert_used, n_tokens);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, weights);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> logits_data(static_cast<size_t>(n_experts * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            const int64_t raw = (expert * 37 + token * 19 + (expert / 5) * 3) % 29;
            float value = static_cast<float>(raw - 14) * 0.125f;
            if (token == 0 && expert < std::min<int64_t>(n_experts, 4)) {
                value = 1.25f;
            }
            logits_data[static_cast<size_t>(token * n_experts + expert)] = value;
        }
    }
    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> expected_ids(static_cast<size_t>(n_expert_used * n_tokens));
    std::vector<float> expected_weights(static_cast<size_t>(n_expert_used * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float * row = logits_data.data() + token * n_experts;
        float max_value = row[0];
        for (int64_t expert = 1; expert < n_experts; ++expert) {
            max_value = std::max(max_value, row[expert]);
        }

        std::vector<float> probabilities(static_cast<size_t>(n_experts));
        float sum = 0.0f;
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            probabilities[static_cast<size_t>(expert)] = std::exp(row[expert] - max_value);
            sum += probabilities[static_cast<size_t>(expert)];
        }
        for (float & value : probabilities) {
            value /= sum;
        }

        std::vector<int32_t> order(static_cast<size_t>(n_experts));
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            order[static_cast<size_t>(expert)] = static_cast<int32_t>(expert);
        }
        std::sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
            const float lhs_value = probabilities[static_cast<size_t>(lhs)];
            const float rhs_value = probabilities[static_cast<size_t>(rhs)];
            if (lhs_value != rhs_value) {
                return lhs_value > rhs_value;
            }
            return lhs < rhs;
        });

        float selected_sum = 0.0f;
        for (int64_t k = 0; k < n_expert_used; ++k) {
            selected_sum += probabilities[static_cast<size_t>(order[static_cast<size_t>(k)])];
        }
        const float denom = with_norm ? std::max(selected_sum, 6.103515625e-5f) : 1.0f;
        for (int64_t k = 0; k < n_expert_used; ++k) {
            const int32_t expert = order[static_cast<size_t>(k)];
            const size_t out_idx = static_cast<size_t>(token * n_expert_used + k);
            expected_ids[out_idx] = expert;
            expected_weights[out_idx] = probabilities[static_cast<size_t>(expert)] / denom;
        }
    }

    std::vector<int32_t> actual_ids(static_cast<size_t>(n_expert_used * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        ggml_backend_tensor_get(
            selected_experts,
            actual_ids.data() + token * n_expert_used,
            static_cast<size_t>(token) * selected_experts->nb[1],
            static_cast<size_t>(n_expert_used) * sizeof(int32_t));
    }
    GGML_ASSERT(actual_ids.size() == expected_ids.size());
    for (size_t i = 0; i < actual_ids.size(); ++i) {
        if (actual_ids[i] != expected_ids[i]) {
            std::fprintf(stderr, "%s ids[%zu]: got %" PRId32 " expected %" PRId32 "\n",
                label, i, actual_ids[i], expected_ids[i]);
            std::abort();
        }
    }
    expect_near(tensor_to_float(weights), expected_weights, 2.0e-5f, label);
}

enum class rope_test_mode {
    normal,
    neox,
    imrope,
    mrope,
    vision,
};

static int rope_mode_value(rope_test_mode mode) {
    switch (mode) {
        case rope_test_mode::normal:
            return GGML_ROPE_TYPE_NORMAL;
        case rope_test_mode::neox:
            return GGML_ROPE_TYPE_NEOX;
        case rope_test_mode::imrope:
            return GGML_ROPE_TYPE_IMROPE;
        case rope_test_mode::mrope:
            return GGML_ROPE_TYPE_MROPE;
        case rope_test_mode::vision:
            return GGML_ROPE_TYPE_VISION;
    }
    GGML_ABORT("unknown rope test mode");
}

static const char * rope_mode_name(rope_test_mode mode) {
    switch (mode) {
        case rope_test_mode::normal:
            return "normal";
        case rope_test_mode::neox:
            return "neox";
        case rope_test_mode::imrope:
            return "imrope";
        case rope_test_mode::mrope:
            return "mrope";
        case rope_test_mode::vision:
            return "vision";
    }
    return "unknown";
}

static bool rope_mode_uses_multi_pos(rope_test_mode mode) {
    return mode == rope_test_mode::imrope ||
        mode == rope_test_mode::mrope ||
        mode == rope_test_mode::vision;
}

static int64_t rope_pos_count(rope_test_mode mode, int64_t ne2) {
    return rope_mode_uses_multi_pos(mode) ? ne2 * 4 : ne2;
}

static void rope_sections_for_width(int sections[GGML_MROPE_SECTIONS], int64_t ne0) {
    sections[0] = static_cast<int32_t>(ne0 / 4);
    sections[1] = static_cast<int32_t>(ne0 / 4);
    sections[2] = static_cast<int32_t>(ne0 / 4);
    sections[3] = static_cast<int32_t>(ne0 - 3 * (ne0 / 4));
}

static ggml_tensor * build_rope_f32_op(
        ggml_context * ctx,
        ggml_tensor * src,
        ggml_tensor * pos,
        ggml_tensor * freq,
        rope_test_mode mode,
        int sections[GGML_MROPE_SECTIONS],
        int64_t ne0,
        float ext_factor,
        float freq_scale,
        float attn_factor) {
    if (rope_mode_uses_multi_pos(mode)) {
        return ggml_rope_multi(
            ctx, src, pos, freq, static_cast<int>(ne0), sections, rope_mode_value(mode), 0,
            10000.0f, freq_scale, ext_factor, attn_factor, 1.0f, 1.0f);
    }

    return ggml_rope_ext(
        ctx, src, pos, freq, static_cast<int>(ne0), rope_mode_value(mode), 0,
        10000.0f, freq_scale, ext_factor, attn_factor, 1.0f, 1.0f);
}

static std::vector<float> run_rope_f32_cpu_reference(
        const std::vector<float> & src_data,
        const std::vector<int32_t> & pos_data,
        rope_test_mode mode,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        float freq_scale,
        float attn_factor) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rope_pos_count(mode, ne2));
    int sections[GGML_MROPE_SECTIONS];
    rope_sections_for_width(sections, ne0);
    ggml_tensor * out = build_rope_f32_op(
        ctx.get(), src, pos, nullptr, mode, sections, ne0, 0.0f, freq_scale, attn_factor);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);

    return tensor_to_float(out);
}

struct rope_support_case {
    rope_test_mode mode;
    ggml_type type;
    bool freq_factors;
    float ext_factor;
    bool expected_supported;
    const char * label;
};

static void run_rope_support_case(ggml_backend_dev_t dev, const rope_support_case & cfg) {
    ggml_context_ptr ctx = make_context();
    constexpr int64_t ne0 = 128;
    constexpr int64_t ne1 = 4;
    constexpr int64_t ne2 = 2;
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), cfg.type, ne0, ne1, ne2);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rope_pos_count(cfg.mode, ne2));
    ggml_tensor * freq = cfg.freq_factors ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ne0 / 2) : nullptr;
    int sections[GGML_MROPE_SECTIONS];
    rope_sections_for_width(sections, ne0);
    ggml_tensor * out = build_rope_f32_op(
        ctx.get(), src, pos, freq, cfg.mode, sections, ne0, cfg.ext_factor, 1.0f, 1.0f);

    const bool supported = ggml_backend_dev_supports_op(dev, out);
    if (supported != cfg.expected_supported) {
        std::fprintf(stderr, "%s: ROPE %s support got %s expected %s\n",
            cfg.label, rope_mode_name(cfg.mode), supported ? "supported" : "unsupported",
            cfg.expected_supported ? "supported" : "unsupported");
        std::abort();
    }
}

static void run_rope_support_matrix(ggml_backend_dev_t dev) {
    const std::array<rope_support_case, 8> cases = {{
        { rope_test_mode::normal, GGML_TYPE_F32, false, 0.0f, true,  "rope_normal_f32" },
        { rope_test_mode::neox,   GGML_TYPE_F32, false, 0.0f, true,  "rope_neox_f32" },
        { rope_test_mode::imrope, GGML_TYPE_F32, false, 0.0f, true,  "rope_imrope_f32" },
        { rope_test_mode::normal, GGML_TYPE_F16, false, 0.0f, false, "rope_normal_f16" },
        { rope_test_mode::normal, GGML_TYPE_F32, true,  0.0f, false, "rope_normal_freq_factors" },
        { rope_test_mode::normal, GGML_TYPE_F32, false, 0.5f, false, "rope_normal_yarn" },
        { rope_test_mode::mrope,  GGML_TYPE_F32, false, 0.0f, false, "rope_mrope_f32" },
        { rope_test_mode::vision, GGML_TYPE_F32, false, 0.0f, false, "rope_vision_f32" },
    }};
    for (const rope_support_case & cfg : cases) {
        run_rope_support_case(dev, cfg);
    }
}

static void run_rope_f32_case(
        ggml_backend_t backend,
        rope_test_mode mode,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        float freq_scale = 1.0f,
        float attn_factor = 1.0f) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rope_pos_count(mode, ne2));
    int sections[GGML_MROPE_SECTIONS];
    rope_sections_for_width(sections, ne0);
    ggml_tensor * out = build_rope_f32_op(
        ctx.get(), src, pos, nullptr, mode, sections, ne0, 0.0f, freq_scale, attn_factor);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    std::vector<float> src_data(static_cast<size_t>(ne0 * nrows));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>((i * 13) % 29) * 0.05f - 0.7f;
    }
    std::vector<int32_t> pos_data(static_cast<size_t>(rope_pos_count(mode, ne2)));
    for (size_t i = 0; i < pos_data.size(); ++i) {
        pos_data[i] = static_cast<int32_t>((i * 3) % 17);
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const std::vector<float> expected =
        run_rope_f32_cpu_reference(src_data, pos_data, mode, ne0, ne1, ne2, freq_scale, attn_factor);
    expect_near(tensor_to_float(out), expected, 2.0e-5f, rope_mode_name(mode));
}

static void run_ssm_conv_case(ggml_backend_t backend, int64_t d_conv, int64_t d_inner, int64_t n_tokens, int64_t n_seqs) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d_conv - 1 + n_tokens, d_inner, n_seqs);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_conv, d_inner);
    ggml_tensor * out = ggml_ssm_conv(ctx.get(), src, weight);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t conv_width = d_conv - 1 + n_tokens;
    std::vector<float> src_data(static_cast<size_t>(conv_width * d_inner * n_seqs));
    std::vector<float> weight_data(static_cast<size_t>(d_conv * d_inner));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>((i * 5) % 23) * 0.1f - 1.0f;
    }
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((i * 7) % 19) * 0.05f - 0.4f;
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(d_inner * n_tokens * n_seqs));
    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t channel = 0; channel < d_inner; ++channel) {
                float sum = 0.0f;
                for (int64_t i = 0; i < d_conv; ++i) {
                    const float x = src_data[(seq * d_inner + channel) * conv_width + token + i];
                    const float w = weight_data[channel * d_conv + i];
                    sum += x * w;
                }
                expected[(seq * n_tokens + token) * d_inner + channel] = sum;
            }
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "ssm_conv");
}

static void run_gated_delta_net_case(ggml_backend_t backend, bool kda) {
    static constexpr int64_t S = 4;
    static constexpr int64_t H = 2;
    static constexpr int64_t T = 2;
    static constexpr int64_t B = 1;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, kda ? S : 1, H, T, B);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, S * S * H, B);
    ggml_tensor * out = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const auto fill = [](std::vector<float> & values, int mul, float scale, float bias) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>((i * mul) % 17) * scale + bias;
        }
    };
    std::vector<float> q_data(static_cast<size_t>(S * H * T * B));
    std::vector<float> k_data(q_data.size());
    std::vector<float> v_data(q_data.size());
    std::vector<float> g_data(static_cast<size_t>((kda ? S : 1) * H * T * B));
    std::vector<float> beta_data(static_cast<size_t>(H * T * B));
    std::vector<float> state_data(static_cast<size_t>(S * S * H * B));
    fill(q_data, 3, 0.025f, -0.2f);
    fill(k_data, 5, 0.02f, -0.15f);
    fill(v_data, 7, 0.03f, -0.1f);
    fill(g_data, 11, 0.01f, -0.05f);
    fill(beta_data, 13, 0.015f, 0.25f);
    fill(state_data, 2, 0.02f, -0.3f);
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta, beta_data.data(), 0, beta_data.size() * sizeof(float));
    ggml_backend_tensor_set(state, state_data.data(), 0, state_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(S * H * T * B + S * S * H * B));
    const float scale = 1.0f / std::sqrt(static_cast<float>(S));
    for (int64_t seq = 0; seq < B; ++seq) {
        for (int64_t head = 0; head < H; ++head) {
            for (int64_t col = 0; col < S; ++col) {
                float s_col[S];
                for (int64_t row = 0; row < S; ++row) {
                    s_col[row] = state_data[(seq * H + head) * S * S + col * S + row];
                }
                for (int64_t token = 0; token < T; ++token) {
                    float kv_col = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        const float g_row = kda ?
                            std::exp(g_data[((seq * T + token) * H + head) * S + row]) : 1.0f;
                        kv_col += g_row * s_col[row] * k_data[((seq * T + token) * H + head) * S + row];
                    }
                    const float beta_val = beta_data[(seq * T + token) * H + head];
                    const float v_col = v_data[((seq * T + token) * H + head) * S + col];
                    const float g_scalar = kda ? 1.0f : std::exp(g_data[(seq * T + token) * H + head]);
                    const float delta_col = (v_col - (kda ? kv_col : g_scalar * kv_col)) * beta_val;
                    float attn = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        const float g_row = kda ?
                            std::exp(g_data[((seq * T + token) * H + head) * S + row]) : g_scalar;
                        s_col[row] = g_row * s_col[row] +
                            k_data[((seq * T + token) * H + head) * S + row] * delta_col;
                        attn += s_col[row] * q_data[((seq * T + token) * H + head) * S + row];
                    }
                    expected[((seq * T + token) * H + head) * S + col] = attn * scale;
                }
                const int64_t state_offset = S * H * T * B;
                for (int64_t row = 0; row < S; ++row) {
                    expected[state_offset + (seq * H + head) * S * S + col * S + row] = s_col[row];
                }
            }
        }
    }
    expect_near(tensor_to_float(out), expected, 3.0e-5f, kda ? "gated_delta_net_kda" : "gated_delta_net");
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

class scoped_env_var {
public:
    scoped_env_var(const char * name, const char * value) : name(name) {
        if (const char * existing = std::getenv(name)) {
            old_value = existing;
            had_value = true;
        }
        ::setenv(name, value, 1);
    }

    ~scoped_env_var() {
        if (had_value) {
            ::setenv(name, old_value.c_str(), 1);
        } else {
            ::unsetenv(name);
        }
    }

private:
    const char * name;
    bool had_value = false;
    std::string old_value;
};

static void run_l2_norm_pair_fusion_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src0 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 127, 3, 2);
    ggml_tensor * src1 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 5, 1);
    constexpr float eps = 1.0e-7f;
    ggml_tensor * out0 = ggml_l2_norm(ctx.get(), src0, eps);
    ggml_tensor * out1 = ggml_l2_norm(ctx.get(), src1, eps);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> src0_data = rowwise_input(src0->ne[0], src0->ne[1] * src0->ne[2]);
    const std::vector<float> src1_data = rowwise_input(src1->ne[0], src1->ne[1] * src1->ne[2]);
    ggml_backend_tensor_set(src0, src0_data.data(), 0, src0_data.size() * sizeof(float));
    ggml_backend_tensor_set(src1, src1_data.data(), 0, src1_data.size() * sizeof(float));

    scoped_env_var disable_l2_norm("GGML_HRX_DISABLE_L2_NORM", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const auto make_expected = [eps](const std::vector<float> & input, int64_t ncols) {
        std::vector<float> expected(input.size());
        const int64_t nrows = static_cast<int64_t>(input.size()) / ncols;
        for (int64_t row = 0; row < nrows; ++row) {
            float sum = 0.0f;
            for (int64_t col = 0; col < ncols; ++col) {
                const float value = input[static_cast<size_t>(row * ncols + col)];
                sum += value * value;
            }
            const float scale = 1.0f / std::max(std::sqrt(sum), eps);
            for (int64_t col = 0; col < ncols; ++col) {
                expected[static_cast<size_t>(row * ncols + col)] =
                    input[static_cast<size_t>(row * ncols + col)] * scale;
            }
        }
        return expected;
    };
    expect_near(tensor_to_float(out0), make_expected(src0_data, src0->ne[0]), 2.0e-5f, "l2_norm_pair_out0");
    expect_near(tensor_to_float(out1), make_expected(src1_data, src1->ne[0]), 2.0e-5f, "l2_norm_pair_out1");
}

static void run_sigmoid_mul_add_add_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 32;
    constexpr int64_t rows = 5;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * sigmoid_src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * mul_src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
    ggml_tensor * add0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, 1);
    ggml_tensor * add1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, 1);
    ggml_tensor * sigmoid = ggml_sigmoid(ctx.get(), sigmoid_src);
    ggml_tensor * mul = ggml_mul(ctx.get(), mul_src, sigmoid);
    ggml_tensor * sum0 = ggml_add(ctx.get(), mul, add0);
    ggml_tensor * dst = ggml_add(ctx.get(), sum0, add1);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, dst));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const float sigmoid_x = -0.37f;
    const float sigmoid_value = 1.0f / (1.0f + std::exp(-sigmoid_x));
    std::vector<float> mul_data(static_cast<size_t>(cols * rows));
    std::vector<float> add0_data(static_cast<size_t>(cols));
    std::vector<float> add1_data(static_cast<size_t>(cols));
    std::vector<float> expected(mul_data.size());
    for (int64_t col = 0; col < cols; ++col) {
        add0_data[static_cast<size_t>(col)] = 0.01f * static_cast<float>(col - 9);
        add1_data[static_cast<size_t>(col)] = -0.02f * static_cast<float>(col - 5);
    }
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row * cols + col);
            mul_data[index] = 0.001f * static_cast<float>(row * cols + col) - 0.08f;
            expected[index] = mul_data[index] * sigmoid_value +
                add0_data[static_cast<size_t>(col)] + add1_data[static_cast<size_t>(col)];
        }
    }

    ggml_backend_tensor_set(sigmoid_src, &sigmoid_x, 0, sizeof(sigmoid_x));
    ggml_backend_tensor_set(mul_src, mul_data.data(), 0, mul_data.size() * sizeof(float));
    ggml_backend_tensor_set(add0, add0_data.data(), 0, add0_data.size() * sizeof(float));
    ggml_backend_tensor_set(add1, add1_data.data(), 0, add1_data.size() * sizeof(float));

    scoped_env_var disable_sigmoid("GGML_HRX_DISABLE_SIGMOID", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(tensor_to_float(dst), expected, 1.0e-5f, "sigmoid_mul_add_add_fusion_output");
}

static void run_sigmoid_mul_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 257;
    constexpr int64_t rows = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
    ggml_tensor * attn = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
    ggml_tensor * sigmoid = ggml_sigmoid(ctx.get(), gate);
    ggml_tensor * mul = ggml_mul(ctx.get(), attn, sigmoid);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, mul));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, mul);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_data(static_cast<size_t>(cols * rows));
    std::vector<float> attn_data(static_cast<size_t>(cols * rows));
    std::vector<float> expected(gate_data.size());
    for (size_t i = 0; i < gate_data.size(); ++i) {
        gate_data[i] = 0.02f * static_cast<float>(static_cast<int>(i % 37) - 18);
        attn_data[i] = 0.01f * static_cast<float>(static_cast<int>(i % 53) - 26);
        expected[i] = attn_data[i] / (1.0f + std::exp(-gate_data[i]));
    }

    ggml_backend_tensor_set(gate, gate_data.data(), 0, gate_data.size() * sizeof(float));
    ggml_backend_tensor_set(attn, attn_data.data(), 0, attn_data.size() * sizeof(float));

    scoped_env_var disable_sigmoid("GGML_HRX_DISABLE_SIGMOID", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(tensor_to_float(mul), expected, 1.0e-5f, "sigmoid_mul_fusion_output");
}

static void run_sigmoid_mul_strided_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 4;
    constexpr int64_t heads = 2;
    constexpr int64_t rows = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * attn_base = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, rows, heads);
    ggml_tensor * attn_view = ggml_permute(ctx.get(), attn_base, 0, 2, 1, 3);
    ggml_tensor * attn_cont = ggml_cont(ctx.get(), attn_view);

    ggml_tensor * gate_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols * 2, heads * rows);
    ggml_tensor * gate_view = ggml_view_3d(
        ctx.get(), gate_base, cols, heads, rows, gate_base->nb[1], gate_base->nb[1] * heads, cols * sizeof(float));
    ggml_tensor * gate_cont = ggml_cont(ctx.get(), gate_view);

    ggml_tensor * sigmoid = ggml_sigmoid(ctx.get(), gate_cont);
    ggml_tensor * mul = ggml_mul(ctx.get(), attn_cont, sigmoid);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, mul));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, mul);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> attn_data(static_cast<size_t>(cols * rows * heads), 0.0f);
    for (size_t i = 0; i < attn_data.size(); ++i) {
        attn_data[i] = static_cast<float>(static_cast<int>(i) - 11) / 7.0f;
    }
    std::vector<float> gate_data(static_cast<size_t>(cols * 2 * heads * rows), 0.0f);
    for (size_t i = 0; i < gate_data.size(); ++i) {
        gate_data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 5.0f;
    }

    ggml_backend_tensor_set(attn_base, attn_data.data(), 0, attn_data.size() * sizeof(float));
    ggml_backend_tensor_set(gate_base, gate_data.data(), 0, gate_data.size() * sizeof(float));

    scoped_env_var disable_sigmoid("GGML_HRX_DISABLE_SIGMOID", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    auto sigmoid_ref = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    std::vector<float> expected(static_cast<size_t>(cols * heads * rows), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t col = 0; col < cols; ++col) {
                const size_t dst_idx = static_cast<size_t>((row * heads + head) * cols + col);
                const size_t attn_idx = static_cast<size_t>((head * rows + row) * cols + col);
                const size_t gate_idx = static_cast<size_t>((row * heads + head) * (cols * 2) + cols + col);
                expected[dst_idx] = attn_data[attn_idx] * sigmoid_ref(gate_data[gate_idx]);
            }
        }
    }
    expect_near(tensor_to_float(mul), expected, 1.0e-5f, "sigmoid_mul_strided_fusion_output");
}

static void run_sigmoid_mul_gate_strided_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 4;
    constexpr int64_t heads = 2;
    constexpr int64_t rows = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * attn = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, heads, rows);
    ggml_tensor * gate_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols * 2, heads * rows);
    ggml_tensor * gate_view = ggml_view_3d(
        ctx.get(), gate_base, cols, heads, rows, gate_base->nb[1], gate_base->nb[1] * heads, cols * sizeof(float));
    ggml_tensor * gate_cont = ggml_cont(ctx.get(), gate_view);
    ggml_tensor * sigmoid = ggml_sigmoid(ctx.get(), gate_cont);
    ggml_tensor * mul = ggml_mul(ctx.get(), attn, sigmoid);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, mul));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, mul);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> attn_data(static_cast<size_t>(cols * heads * rows), 0.0f);
    for (size_t i = 0; i < attn_data.size(); ++i) {
        attn_data[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 11.0f;
    }
    std::vector<float> gate_data(static_cast<size_t>(cols * 2 * heads * rows), 0.0f);
    for (size_t i = 0; i < gate_data.size(); ++i) {
        gate_data[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 7.0f;
    }

    ggml_backend_tensor_set(attn, attn_data.data(), 0, attn_data.size() * sizeof(float));
    ggml_backend_tensor_set(gate_base, gate_data.data(), 0, gate_data.size() * sizeof(float));

    scoped_env_var disable_sigmoid("GGML_HRX_DISABLE_SIGMOID", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    auto sigmoid_ref = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    std::vector<float> expected(static_cast<size_t>(cols * heads * rows), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t col = 0; col < cols; ++col) {
                const size_t dst_idx = static_cast<size_t>((row * heads + head) * cols + col);
                const size_t gate_idx = static_cast<size_t>((row * heads + head) * (cols * 2) + cols + col);
                expected[dst_idx] = attn_data[dst_idx] * sigmoid_ref(gate_data[gate_idx]);
            }
        }
    }
    expect_near(tensor_to_float(mul), expected, 1.0e-5f, "sigmoid_mul_gate_strided_fusion_output");
}

static void run_sigmoid_mul_strided_negative_intervening_op_case(ggml_backend_t backend) {
    constexpr int64_t cols = 4;
    constexpr int64_t heads = 2;
    constexpr int64_t rows = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * attn_base = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cols, rows, heads);
    ggml_tensor * attn_view = ggml_permute(ctx.get(), attn_base, 0, 2, 1, 3);
    ggml_tensor * attn_cont = ggml_cont(ctx.get(), attn_view);
    ggml_tensor * intervening_add = ggml_add(ctx.get(), attn_base, attn_base);

    ggml_tensor * gate_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols * 2, heads * rows);
    ggml_tensor * gate_view = ggml_view_3d(
        ctx.get(), gate_base, cols, heads, rows, gate_base->nb[1], gate_base->nb[1] * heads, cols * sizeof(float));
    ggml_tensor * gate_cont = ggml_cont(ctx.get(), gate_view);
    ggml_tensor * sigmoid = ggml_sigmoid(ctx.get(), gate_cont);
    ggml_tensor * mul = ggml_mul(ctx.get(), attn_cont, sigmoid);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, attn_cont);
    ggml_build_forward_expand(graph, intervening_add);
    ggml_build_forward_expand(graph, mul);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> attn_data(static_cast<size_t>(cols * rows * heads), 0.25f);
    std::vector<float> gate_data(static_cast<size_t>(cols * 2 * heads * rows), -0.125f);
    ggml_backend_tensor_set(attn_base, attn_data.data(), 0, attn_data.size() * sizeof(float));
    ggml_backend_tensor_set(gate_base, gate_data.data(), 0, gate_data.size() * sizeof(float));

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_ssm_conv_silu_fusion_case(ggml_backend_t backend) {
    constexpr int64_t d_conv = 4;
    constexpr int64_t d_inner = 33;
    constexpr int64_t n_tokens = 17;
    constexpr int64_t n_seqs = 2;
    const int64_t conv_width = d_conv - 1 + n_tokens;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, conv_width, d_inner, n_seqs);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_conv, d_inner);
    ggml_tensor * ssm = ggml_ssm_conv(ctx.get(), src, weight);
    ggml_tensor * out = ggml_silu(ctx.get(), ssm);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(conv_width * d_inner * n_seqs));
    std::vector<float> weight_data(static_cast<size_t>(d_conv * d_inner));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>((i * 5) % 23) * 0.1f - 1.0f;
    }
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((i * 7) % 19) * 0.05f - 0.4f;
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(d_inner * n_tokens * n_seqs));
    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t channel = 0; channel < d_inner; ++channel) {
                float sum = 0.0f;
                for (int64_t i = 0; i < d_conv; ++i) {
                    const float x = src_data[static_cast<size_t>((seq * d_inner + channel) * conv_width + token + i)];
                    const float w = weight_data[static_cast<size_t>(channel * d_conv + i)];
                    sum += x * w;
                }
                const float sigmoid = 1.0f / (1.0f + std::exp(-sum));
                expected[static_cast<size_t>((seq * n_tokens + token) * d_inner + channel)] = sum * sigmoid;
            }
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "ssm_conv_silu_fusion_output");
}

static void run_gated_delta_net_s128_beta_sigmoid_state_update_fusion_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev) {
    constexpr int64_t S = 128;
    constexpr int64_t H = 32;
    constexpr int64_t Q_HEADS = 16;
    constexpr int64_t T = 1;
    constexpr int64_t B = 1;
    constexpr int64_t ATTN_ELEMS = S * H * T * B;
    constexpr int64_t STATE_ELEMS = S * S * H * B;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, Q_HEADS, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, Q_HEADS, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta_raw = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta = ggml_sigmoid(ctx.get(), beta_raw);
    ggml_tensor * state = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, S * S, H, B);
    ggml_tensor * gdn = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);
    ggml_tensor * state_view = ggml_view_1d(
        ctx.get(), gdn, STATE_ELEMS, static_cast<size_t>(ATTN_ELEMS) * sizeof(float));
    ggml_tensor * state_dst = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, STATE_ELEMS);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, gdn));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, gdn);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(S * Q_HEADS * T * B));
    std::vector<float> k_data(q_data.size());
    std::vector<float> v_data(static_cast<size_t>(S * H * T * B));
    std::vector<float> g_data(static_cast<size_t>(H * T * B));
    std::vector<float> beta_data(g_data.size());
    std::vector<float> state_data(static_cast<size_t>(STATE_ELEMS));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.05f * static_cast<float>(static_cast<int>(i % 17) - 8);
        k_data[i] = 0.04f * static_cast<float>(static_cast<int>(i % 5) - 2);
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = 0.03f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    for (size_t i = 0; i < g_data.size(); ++i) {
        g_data[i] = -0.35f + 0.2f * static_cast<float>(i % 3);
        beta_data[i] = 0.25f + 0.05f * static_cast<float>(i % 4);
    }
    for (size_t i = 0; i < state_data.size(); ++i) {
        state_data[i] = 0.01f * static_cast<float>(static_cast<int>(i % 13) - 6);
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta_raw, beta_data.data(), 0, beta_data.size() * sizeof(float));
    ggml_backend_tensor_set(state, state_data.data(), 0, state_data.size() * sizeof(float));

    scoped_env_var disable_sigmoid("GGML_HRX_DISABLE_SIGMOID", "1");
    scoped_env_var disable_gdn("GGML_HRX_DISABLE_GATED_DELTA_NET", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    for (float & value : beta_data) {
        value = 1.0f / (1.0f + std::exp(-value));
    }
    const std::vector<float> expected =
        reference_gated_delta_net(q_data, k_data, v_data, g_data, beta_data, state_data, S, H, Q_HEADS, T, B);
    const std::vector<float> gdn_output = tensor_to_float(gdn);
    expect_near(gdn_output, expected, 1.0e-4f, "gated_delta_net_beta_sigmoid_state_update_gdn");
    expect_near(
        tensor_to_float(state_update),
        std::vector<float>(expected.begin() + ATTN_ELEMS, expected.end()),
        1.0e-4f,
        "gated_delta_net_beta_sigmoid_state_update_state");
}

static void run_gated_delta_net_s128_beta_sigmoid_state_gather_fusion_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        bool scaled_state) {
    constexpr int64_t S = 128;
    constexpr int64_t H = 32;
    constexpr int64_t Q_HEADS = 16;
    constexpr int64_t T = 1;
    constexpr int64_t B = 1;
    constexpr int64_t ATTN_ELEMS = S * H * T * B;
    constexpr int64_t STATE_ELEMS = S * S * H * B;
    constexpr int64_t STATE_ROWS = 3;
    constexpr int32_t SELECTED_ROW = 2;
    constexpr float SCALE = 0.5f;
    constexpr float BIAS = -0.125f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, Q_HEADS, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, Q_HEADS, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta_raw = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta = ggml_sigmoid(ctx.get(), beta_raw);
    ggml_tensor * state_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_ELEMS, STATE_ROWS);
    ggml_tensor * state_idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * state_src = scaled_state ?
        ggml_scale_bias_inplace(ctx.get(), state_base, SCALE, BIAS) : state_base;
    ggml_tensor * state = ggml_get_rows(ctx.get(), state_src, state_idx);
    ggml_tensor * gdn = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);
    ggml_tensor * state_view = ggml_view_1d(
        ctx.get(), gdn, STATE_ELEMS, static_cast<size_t>(ATTN_ELEMS) * sizeof(float));
    ggml_tensor * state_dst = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, STATE_ELEMS);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, gdn));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, state);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, gdn);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(S * Q_HEADS * T * B));
    std::vector<float> k_data(q_data.size());
    std::vector<float> v_data(static_cast<size_t>(S * H * T * B));
    std::vector<float> g_data(static_cast<size_t>(H * T * B));
    std::vector<float> beta_data(g_data.size());
    std::vector<float> state_base_data(static_cast<size_t>(STATE_ELEMS * STATE_ROWS));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.04f * static_cast<float>(static_cast<int>(i % 17) - 8);
        k_data[i] = 0.03f * static_cast<float>(static_cast<int>(i % 7) - 3);
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = 0.025f * static_cast<float>(static_cast<int>(i % 13) - 6);
    }
    for (size_t i = 0; i < g_data.size(); ++i) {
        g_data[i] = -0.3f + 0.15f * static_cast<float>(i % 5);
        beta_data[i] = -0.2f + 0.1f * static_cast<float>(i % 7);
    }
    for (size_t i = 0; i < state_base_data.size(); ++i) {
        state_base_data[i] = 0.00625f * static_cast<float>(static_cast<int>(i % 31) - 15);
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta_raw, beta_data.data(), 0, beta_data.size() * sizeof(float));
    ggml_backend_tensor_set(state_base, state_base_data.data(), 0, state_base_data.size() * sizeof(float));
    ggml_backend_tensor_set(state_idx, &SELECTED_ROW, 0, sizeof(SELECTED_ROW));

    scoped_env_var expect_provider(
        "GGML_HRX_EXPECT_GATED_DELTA_NET_PROVIDER",
        scaled_state ?
            "hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_beta_sigmoid_gather_f32" :
            "hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_beta_sigmoid_direct_gather_f32");
    scoped_env_var disable_sigmoid("GGML_HRX_DISABLE_SIGMOID", "1");
    scoped_env_var disable_gdn("GGML_HRX_DISABLE_GATED_DELTA_NET", "1");
    scoped_env_var disable_scale_get_rows("GGML_HRX_DISABLE_SCALE_GET_ROWS_FUSION", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> beta_sigmoid_data = beta_data;
    for (float & value : beta_sigmoid_data) {
        value = 1.0f / (1.0f + std::exp(-value));
    }
    std::vector<float> gathered_state(static_cast<size_t>(STATE_ELEMS));
    for (size_t i = 0; i < gathered_state.size(); ++i) {
        gathered_state[i] = state_base_data[static_cast<size_t>(SELECTED_ROW) * static_cast<size_t>(STATE_ELEMS) + i];
        if (scaled_state) {
            gathered_state[i] = gathered_state[i] * SCALE + BIAS;
        }
    }
    const std::vector<float> expected =
        reference_gated_delta_net(q_data, k_data, v_data, g_data, beta_sigmoid_data, gathered_state,
                                  S, H, Q_HEADS, T, B);

    expect_near(
        tensor_to_float(gdn),
        expected,
        1.0e-4f,
        scaled_state ? "gated_delta_net_state_gather_scaled_gdn" : "gated_delta_net_state_gather_direct_gdn");
    expect_near(
        tensor_to_float(state_update),
        std::vector<float>(expected.begin() + ATTN_ELEMS, expected.end()),
        1.0e-4f,
        scaled_state ? "gated_delta_net_state_gather_scaled_state" : "gated_delta_net_state_gather_direct_state");
}

static void run_gated_delta_net_state_update_negative_truncated_case(ggml_backend_t backend) {
    constexpr int64_t S = 4;
    constexpr int64_t H = 2;
    constexpr int64_t T = 2;
    constexpr int64_t B = 1;
    constexpr int64_t ATTN_ELEMS = S * H * T * B;
    constexpr int64_t STATE_ELEMS = S * S * H * B;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, S * S * H, B);
    ggml_tensor * gdn = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);
    ggml_tensor * state_view = ggml_view_1d(
        ctx.get(), gdn, STATE_ELEMS - 1, static_cast<size_t>(ATTN_ELEMS) * sizeof(float));
    ggml_tensor * state_dst = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, STATE_ELEMS - 1);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(S * H * T * B), 0.1f);
    std::vector<float> g_data(static_cast<size_t>(H * T * B), 0.2f);
    std::vector<float> state_data(static_cast<size_t>(STATE_ELEMS), -0.1f);
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(state, state_data.data(), 0, state_data.size() * sizeof(float));

    scoped_env_var disable_gdn("GGML_HRX_DISABLE_GATED_DELTA_NET", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_gated_delta_net_state_update_negative_intervening_op_case(ggml_backend_t backend) {
    constexpr int64_t S = 4;
    constexpr int64_t H = 2;
    constexpr int64_t T = 2;
    constexpr int64_t B = 1;
    constexpr int64_t ATTN_ELEMS = S * H * T * B;
    constexpr int64_t STATE_ELEMS = S * S * H * B;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, S * S * H, B);
    ggml_tensor * gdn = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);
    ggml_tensor * state_view = ggml_view_1d(
        ctx.get(), gdn, STATE_ELEMS, static_cast<size_t>(ATTN_ELEMS) * sizeof(float));
    ggml_tensor * state_dst = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, STATE_ELEMS);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    ggml_tensor * intervening_add = ggml_add(ctx.get(), q, q);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, gdn);
    ggml_build_forward_expand(graph, intervening_add);
    ggml_build_forward_expand(graph, state_update);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(S * H * T * B), 0.1f);
    std::vector<float> g_data(static_cast<size_t>(H * T * B), 0.2f);
    std::vector<float> state_data(static_cast<size_t>(STATE_ELEMS), -0.1f);
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(state, state_data.data(), 0, state_data.size() * sizeof(float));

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static std::vector<float> reference_rms_norm_mul(
        const std::vector<float> & src,
        const std::vector<float> & weight,
        int64_t ncols,
        int64_t ne1,
        int64_t ne2,
        int64_t weight_ne1,
        int64_t weight_ne2,
        float eps) {
    std::vector<float> expected(src.size());
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        const int64_t wi2 = weight_ne2 == 1 ? 0 : i2;
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t wi1 = weight_ne1 == 1 ? 0 : i1;
            const int64_t row = i1 + ne1 * i2;
            float sum = 0.0f;
            for (int64_t col = 0; col < ncols; ++col) {
                const float value = src[static_cast<size_t>(row * ncols + col)];
                sum += value * value;
            }
            const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + eps);
            for (int64_t col = 0; col < ncols; ++col) {
                const float w = weight[static_cast<size_t>(col + ncols * (wi1 + weight_ne1 * wi2))];
                expected[static_cast<size_t>(row * ncols + col)] =
                    src[static_cast<size_t>(row * ncols + col)] * scale * w;
            }
        }
    }
    return expected;
}

static void apply_imrope_reference(
        std::vector<float> & data,
        const std::vector<int32_t> & pos,
        int64_t ncols,
        int64_t ne1,
        int64_t ne2,
        float freq_scale,
        float attn_factor) {
    data = run_rope_f32_cpu_reference(data, pos, rope_test_mode::imrope, ncols, ne1, ne2, freq_scale, attn_factor);
}

static void fill_rms_rope_inputs(
        ggml_tensor * src,
        ggml_tensor * weight,
        ggml_tensor * pos,
        int64_t weight_ne1,
        int64_t weight_ne2,
        std::vector<float> * src_data,
        std::vector<float> * weight_data,
        std::vector<int32_t> * pos_data) {
    src_data->resize(static_cast<size_t>(ggml_nelements(src)));
    weight_data->resize(static_cast<size_t>(ggml_nelements(weight)));
    pos_data->resize(static_cast<size_t>(ggml_nelements(pos)));
    for (size_t i = 0; i < src_data->size(); ++i) {
        (*src_data)[i] = static_cast<float>(static_cast<int>((i * 17 + 11) % 47) - 23) / 41.0f;
    }
    for (int64_t i2 = 0; i2 < weight_ne2; ++i2) {
        for (int64_t i1 = 0; i1 < weight_ne1; ++i1) {
            for (int64_t i0 = 0; i0 < weight->ne[0]; ++i0) {
                const size_t idx = static_cast<size_t>(i0 + weight->ne[0] * (i1 + weight_ne1 * i2));
                (*weight_data)[idx] = 0.5f + static_cast<float>((i0 + 3 * i1 + 5 * i2) % 13) * 0.0625f;
            }
        }
    }
    for (size_t i = 0; i < pos_data->size(); ++i) {
        (*pos_data)[i] = static_cast<int32_t>((i * 5 + 3) % 19);
    }

    ggml_backend_tensor_set(src, src_data->data(), 0, src_data->size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data->data(), 0, weight_data->size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data->data(), 0, pos_data->size() * sizeof(int32_t));
}

static void run_rms_norm_mul_fusion_case(
        ggml_backend_t backend,
        int64_t ncols,
        int64_t ne1,
        int64_t ne2,
        int64_t weight_ne1,
        int64_t weight_ne2,
        const char * label) {
    static constexpr float eps = 1.0e-6f;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, weight_ne1, weight_ne2);
    ggml_tensor * dummy_pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * rms = ggml_rms_norm(ctx.get(), src, eps);
    ggml_tensor * out = ggml_mul(ctx.get(), rms, weight);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data;
    std::vector<float> weight_data;
    std::vector<int32_t> pos_data;
    fill_rms_rope_inputs(src, weight, dummy_pos, weight_ne1, weight_ne2, &src_data, &weight_data, &pos_data);

    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> expected =
        reference_rms_norm_mul(src_data, weight_data, ncols, ne1, ne2, weight_ne1, weight_ne2, eps);
    expect_near(tensor_to_float(out), expected, 3.0e-5f, label);
}

static void run_add_rms_norm_mul_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev, const char * label) {
    static constexpr int64_t NCOLS = 128;
    static constexpr int64_t NE1 = 3;
    static constexpr int64_t NE2 = 2;
    static constexpr float EPS = 1.0e-6f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, NE1, NE2);
    ggml_tensor * add_rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, 1, 1);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, NE1, 1);
    ggml_tensor * add = ggml_add(ctx.get(), src, add_rhs);
    ggml_tensor * rms = ggml_rms_norm(ctx.get(), add, EPS);
    ggml_tensor * out = ggml_mul(ctx.get(), rms, weight);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(NCOLS * NE1 * NE2));
    std::vector<float> add_rhs_data(static_cast<size_t>(NCOLS));
    std::vector<float> weight_data(static_cast<size_t>(NCOLS * NE1));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 17 + 11) % 47) - 23) / 41.0f;
    }
    for (int64_t col = 0; col < NCOLS; ++col) {
        add_rhs_data[static_cast<size_t>(col)] = 0.01f * static_cast<float>((col % 17) - 8);
    }
    for (int64_t row = 0; row < NE1; ++row) {
        for (int64_t col = 0; col < NCOLS; ++col) {
            weight_data[static_cast<size_t>(col + NCOLS * row)] =
                0.5f + static_cast<float>((col + 3 * row) % 13) * 0.0625f;
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(add_rhs, add_rhs_data.data(), 0, add_rhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> add_expected(src_data.size());
    for (int64_t row = 0; row < NE1 * NE2; ++row) {
        for (int64_t col = 0; col < NCOLS; ++col) {
            add_expected[static_cast<size_t>(row * NCOLS + col)] =
                src_data[static_cast<size_t>(row * NCOLS + col)] + add_rhs_data[static_cast<size_t>(col)];
        }
    }
    expect_near(tensor_to_float(add), add_expected, 1.0e-6f, "add_rms_norm_mul_fusion_add");
    expect_near(
        tensor_to_float(out),
        reference_rms_norm_mul(add_expected, weight_data, NCOLS, NE1, NE2, NE1, 1, EPS),
        3.0e-5f,
        label);
}

static void run_add_add_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev, int64_t ncols, int64_t nrows) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * rhs0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, 1);
    ggml_tensor * rhs1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, 1);
    ggml_tensor * first = ggml_add(ctx.get(), lhs, rhs0);
    ggml_tensor * out = ggml_add(ctx.get(), first, rhs1);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(static_cast<size_t>(ncols * nrows));
    std::vector<float> rhs0_data(static_cast<size_t>(ncols));
    std::vector<float> rhs1_data(static_cast<size_t>(ncols));
    std::vector<float> expected(lhs_data.size());
    for (int64_t col = 0; col < ncols; ++col) {
        rhs0_data[static_cast<size_t>(col)] = 0.01f * static_cast<float>(col - 9);
        rhs1_data[static_cast<size_t>(col)] = -0.02f * static_cast<float>(col - 5);
    }
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            const size_t index = static_cast<size_t>(row * ncols + col);
            lhs_data[index] = 0.001f * static_cast<float>(index) - 0.08f;
            expected[index] = lhs_data[index] + rhs0_data[static_cast<size_t>(col)] +
                rhs1_data[static_cast<size_t>(col)];
        }
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs0, rhs0_data.data(), 0, rhs0_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs1, rhs1_data.data(), 0, rhs1_data.size() * sizeof(float));
    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(out), expected, 1.0e-6f, "add_add_fusion");
}

static void run_add_softplus_mul_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t cols = 32;
    constexpr int64_t rows = 7;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
    ggml_tensor * add_rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, 1);
    ggml_tensor * mul_rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, 1);
    ggml_tensor * sum = ggml_add(ctx.get(), lhs, add_rhs);
    ggml_tensor * softplus = ggml_softplus(ctx.get(), sum);
    ggml_tensor * out = ggml_mul(ctx.get(), softplus, mul_rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(static_cast<size_t>(cols * rows));
    std::vector<float> add_rhs_data(static_cast<size_t>(cols));
    std::vector<float> mul_rhs_data(static_cast<size_t>(cols));
    std::vector<float> expected(lhs_data.size());
    for (int64_t col = 0; col < cols; ++col) {
        add_rhs_data[static_cast<size_t>(col)] = 0.01f * static_cast<float>(col - 16);
        mul_rhs_data[static_cast<size_t>(col)] = 0.5f + 0.02f * static_cast<float>(col);
    }
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row * cols + col);
            const float x = 0.001f * static_cast<float>(index) - 0.1f;
            lhs_data[index] = x;
            const float y = x + add_rhs_data[static_cast<size_t>(col)];
            const float softplus_y = y > 20.0f ? y : std::log(1.0f + std::exp(y));
            expected[index] = softplus_y * mul_rhs_data[static_cast<size_t>(col)];
        }
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(add_rhs, add_rhs_data.data(), 0, add_rhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(mul_rhs, mul_rhs_data.data(), 0, mul_rhs_data.size() * sizeof(float));
    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    scoped_env_var disable_softplus("GGML_HRX_DISABLE_SOFTPLUS", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(out), expected, 1.0e-5f, "add_softplus_mul_fusion");
}

static void run_silu_mul_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev, int64_t nelements) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, nelements);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, nelements);
    ggml_tensor * silu = ggml_silu(ctx.get(), src);
    ggml_tensor * out = ggml_mul(ctx.get(), silu, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(nelements));
    std::vector<float> rhs_data(static_cast<size_t>(nelements));
    std::vector<float> expected(static_cast<size_t>(nelements));
    for (int64_t i = 0; i < nelements; ++i) {
        src_data[static_cast<size_t>(i)] = static_cast<float>((i * 7) % 23 - 11) * 0.125f;
        rhs_data[static_cast<size_t>(i)] = static_cast<float>((i * 5) % 17 - 8) * 0.0625f;
        const float sigmoid = 1.0f / (1.0f + std::exp(-src_data[static_cast<size_t>(i)]));
        expected[static_cast<size_t>(i)] = src_data[static_cast<size_t>(i)] * sigmoid * rhs_data[static_cast<size_t>(i)];
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    scoped_env_var disable_silu("GGML_HRX_DISABLE_SILU", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(out), expected, 1.0e-5f, "silu_mul_fusion");
}

static void run_rms_norm_mul_rope_fusion_case(ggml_backend_t backend, const char * label) {
    static constexpr int64_t NCOLS = 128;
    static constexpr int64_t NE1 = 3;
    static constexpr int64_t NE2 = 2;
    static constexpr float EPS = 1.0e-6f;
    static constexpr float FREQ_BASE = 10000.0f;
    static constexpr float FREQ_SCALE = 1.0f;
    static constexpr float ATTN_FACTOR = 1.0f;
    int sections[GGML_MROPE_SECTIONS] = { 32, 32, 32, 32 };

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, NE1, NE2);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, NCOLS);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NE2 * 4);
    ggml_tensor * rms = ggml_rms_norm(ctx.get(), src, EPS);
    ggml_tensor * mul = ggml_mul(ctx.get(), rms, weight);
    ggml_tensor * out = ggml_rope_multi(
        ctx.get(), mul, pos, nullptr, static_cast<int>(NCOLS), sections, GGML_ROPE_TYPE_IMROPE, 0,
        FREQ_BASE, FREQ_SCALE, 0.0f, ATTN_FACTOR, 1.0f, 1.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data;
    std::vector<float> weight_data;
    std::vector<int32_t> pos_data;
    fill_rms_rope_inputs(src, weight, pos, 1, 1, &src_data, &weight_data, &pos_data);

    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    scoped_env_var disable_rope("GGML_HRX_DISABLE_ROPE", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected = reference_rms_norm_mul(src_data, weight_data, NCOLS, NE1, NE2, 1, 1, EPS);
    apply_imrope_reference(expected, pos_data, NCOLS, NE1, NE2, FREQ_SCALE, ATTN_FACTOR);
    expect_near(tensor_to_float(out), expected, 5.0e-5f, label);
}

static void run_rms_norm_mul_rope_mode_not_fused_case(ggml_backend_t backend, rope_test_mode mode) {
    static constexpr int64_t NCOLS = 128;
    static constexpr int64_t NE1 = 3;
    static constexpr int64_t NE2 = 2;
    static constexpr float EPS = 1.0e-6f;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, NE1, NE2);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, NCOLS);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NE2);
    ggml_tensor * rms = ggml_rms_norm(ctx.get(), src, EPS);
    ggml_tensor * mul = ggml_mul(ctx.get(), rms, weight);
    ggml_tensor * out = ggml_rope_ext(
        ctx.get(), mul, pos, nullptr, static_cast<int>(NCOLS), rope_mode_value(mode), 0,
        10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    scoped_env_var disable_rope("GGML_HRX_DISABLE_ROPE", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_rope_set_rows_fusion_case(ggml_backend_t backend, bool with_rms_mul, const char * label) {
    static constexpr int64_t NCOLS = 64;
    static constexpr int64_t NE1 = 2;
    static constexpr int64_t NE2 = 3;
    static constexpr int64_t DST_ROWS = 4;
    static constexpr float EPS = 1.0e-6f;
    static constexpr float FREQ_BASE = 10000.0f;
    static constexpr float FREQ_SCALE = 1.0f;
    static constexpr float ATTN_FACTOR = 1.0f;
    int sections[GGML_MROPE_SECTIONS] = { 16, 16, 16, 16 };

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, NE1, NE2);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, NCOLS);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NE2 * 4);
    ggml_tensor * rope_src = src;
    if (with_rms_mul) {
        ggml_tensor * rms = ggml_rms_norm(ctx.get(), src, EPS);
        rope_src = ggml_mul(ctx.get(), rms, weight);
    }
    ggml_tensor * rope = ggml_rope_multi(
        ctx.get(), rope_src, pos, nullptr, static_cast<int>(NCOLS), sections, GGML_ROPE_TYPE_IMROPE, 0,
        FREQ_BASE, FREQ_SCALE, 0.0f, ATTN_FACTOR, 1.0f, 1.0f);
    ggml_tensor * view = ggml_view_2d(ctx.get(), rope, NCOLS * NE1, NE2, rope->nb[2], 0);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, NE2);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, NCOLS * NE1, DST_ROWS);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, view, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data;
    std::vector<float> weight_data;
    std::vector<int32_t> pos_data;
    fill_rms_rope_inputs(src, weight, pos, 1, 1, &src_data, &weight_data, &pos_data);
    const int64_t row_data[NE2] = { 2, -1, 0 };
    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    ggml_backend_tensor_set(rows, row_data, 0, sizeof(row_data));
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());

    scoped_env_var disable_rope("GGML_HRX_DISABLE_ROPE", "1");
    scoped_env_var disable_set_rows("GGML_HRX_DISABLE_SET_ROWS", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", with_rms_mul ? "1" : "0");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> rope_expected = with_rms_mul ?
        reference_rms_norm_mul(src_data, weight_data, NCOLS, NE1, NE2, 1, 1, EPS) :
        src_data;
    apply_imrope_reference(rope_expected, pos_data, NCOLS, NE1, NE2, FREQ_SCALE, ATTN_FACTOR);
    std::vector<float> expected(static_cast<size_t>(NCOLS * NE1 * DST_ROWS), 0.0f);
    for (int64_t token = 0; token < NE2; ++token) {
        const int64_t dst_row = row_data[token];
        if (dst_row < 0 || dst_row >= DST_ROWS) {
            continue;
        }
        for (int64_t col = 0; col < NCOLS * NE1; ++col) {
            const float value = rope_expected[static_cast<size_t>(token * NCOLS * NE1 + col)];
            expected[static_cast<size_t>(dst_row * NCOLS * NE1 + col)] =
                ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        }
    }
    expect_near(tensor_to_float(out), expected, 1.0e-3f, label);
}

static void run_rope_set_rows_mode_not_fused_case(ggml_backend_t backend, rope_test_mode mode) {
    static constexpr int64_t NCOLS = 64;
    static constexpr int64_t NE1 = 2;
    static constexpr int64_t NE2 = 3;
    static constexpr int64_t DST_ROWS = 4;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, NE1, NE2);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NE2);
    ggml_tensor * rope = ggml_rope_ext(
        ctx.get(), src, pos, nullptr, static_cast<int>(NCOLS), rope_mode_value(mode), 0,
        10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    ggml_tensor * view = ggml_view_2d(ctx.get(), rope, NCOLS * NE1, NE2, rope->nb[2], 0);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, NE2);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, NCOLS * NE1, DST_ROWS);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, view, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    scoped_env_var disable_rope("GGML_HRX_DISABLE_ROPE", "1");
    scoped_env_var disable_set_rows("GGML_HRX_DISABLE_SET_ROWS", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_rope_set_rows_negative_partial_view_case(ggml_backend_t backend) {
    static constexpr int64_t NCOLS = 64;
    static constexpr int64_t NE1 = 2;
    static constexpr int64_t NE2 = 3;
    static constexpr float FREQ_BASE = 10000.0f;
    static constexpr float FREQ_SCALE = 1.0f;
    static constexpr float ATTN_FACTOR = 1.0f;
    int sections[GGML_MROPE_SECTIONS] = { 16, 16, 16, 16 };

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, NCOLS, NE1, NE2);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NE2 * 4);
    ggml_tensor * rope = ggml_rope_multi(
        ctx.get(), src, pos, nullptr, static_cast<int>(NCOLS), sections, GGML_ROPE_TYPE_IMROPE, 0,
        FREQ_BASE, FREQ_SCALE, 0.0f, ATTN_FACTOR, 1.0f, 1.0f);
    ggml_tensor * view = ggml_view_2d(
        ctx.get(), rope, NCOLS * NE1, NE2 - 1, rope->nb[2], static_cast<size_t>(NCOLS) * sizeof(float));
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, NE2 - 1);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, NCOLS * NE1, NE2);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, view, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ggml_nelements(src)));
    std::vector<int32_t> pos_data(static_cast<size_t>(ggml_nelements(pos)));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>((i * 17 + 11) % 47) - 23) / 41.0f;
    }
    for (size_t i = 0; i < pos_data.size(); ++i) {
        pos_data[i] = static_cast<int32_t>((i * 5 + 3) % 19);
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    const int64_t row_data[NE2 - 1] = { 0, 1 };
    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    ggml_backend_tensor_set(rows, row_data, 0, sizeof(row_data));
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());

    scoped_env_var disable_rope("GGML_HRX_DISABLE_ROPE", "1");
    scoped_env_var disable_set_rows("GGML_HRX_DISABLE_SET_ROWS", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_mul_mat_id_q4_k_mul_fusion_case(
        ggml_backend_t backend,
        const char * label,
        int64_t K = 256,
        int64_t ROWS = 8,
        int64_t N_IDS = 2,
        int64_t N_TOKENS = 3,
        int64_t N_EXPERTS = 4,
        float tolerance = 1.0e-3f,
        bool sampled_reference = false) {
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, ROWS, N_EXPERTS);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, N_IDS, N_TOKENS);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, N_IDS, N_TOKENS);
    ggml_tensor * scale = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, N_IDS, 1);
    ggml_tensor * mmid = ggml_mul_mat_id(ctx.get(), lhs, rhs, ids);
    ggml_tensor * out = ggml_mul(ctx.get(), mmid, scale);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(K * ROWS * N_EXPERTS));
    std::vector<float> rhs_f32(static_cast<size_t>(K * N_IDS * N_TOKENS));
    std::vector<float> scale_f32(static_cast<size_t>(N_IDS));
    std::vector<int32_t> ids_i32(static_cast<size_t>(N_IDS * N_TOKENS));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 13) % 97) - 48) / 53.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 7) % 89) - 44) / 47.0f;
    }
    for (int64_t i = 0; i < N_IDS; ++i) {
        scale_f32[static_cast<size_t>(i)] = 0.25f + 0.125f * static_cast<float>(i);
    }
    for (int64_t token = 0; token < N_TOKENS; ++token) {
        for (int64_t id_pos = 0; id_pos < N_IDS; ++id_pos) {
            ids_i32[static_cast<size_t>(id_pos + N_IDS * token)] =
                static_cast<int32_t>((id_pos + 2 * token) % N_EXPERTS);
        }
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, K, ROWS * N_EXPERTS, lhs_f32, lhs_reference, lhs_storage);
    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    ggml_backend_tensor_set(scale, scale_f32.data(), 0, scale_f32.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> actual = tensor_to_float(out);
    if (sampled_reference) {
        const std::vector<int64_t> row_samples = sample_positions(ROWS);
        const std::vector<int64_t> id_samples = sample_positions(N_IDS);
        const std::vector<int64_t> token_samples = sample_positions(N_TOKENS);
        for (const int64_t token : token_samples) {
            for (const int64_t id_pos : id_samples) {
                for (const int64_t row : row_samples) {
                    const size_t idx = static_cast<size_t>(row + ROWS * (id_pos + N_IDS * token));
                    const float expected = reference_mul_mat_id_q4_k_value(
                        lhs_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false,
                        row, id_pos, token) * scale_f32[static_cast<size_t>(id_pos)];
                    const float got = actual[idx];
                    if (std::fabs(got - expected) > tolerance) {
                        std::fprintf(stderr,
                            "%s[%lld,%lld,%lld]: got %.9g expected %.9g tolerance %.9g\n",
                            label,
                            static_cast<long long>(row),
                            static_cast<long long>(id_pos),
                            static_cast<long long>(token),
                            got,
                            expected,
                            tolerance);
                        std::abort();
                    }
                }
            }
        }
        return;
    }

    std::vector<float> expected =
        reference_mul_mat_id_q4_k(lhs_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false);
    for (int64_t token = 0; token < N_TOKENS; ++token) {
        for (int64_t id_pos = 0; id_pos < N_IDS; ++id_pos) {
            const float factor = scale_f32[static_cast<size_t>(id_pos)];
            for (int64_t row = 0; row < ROWS; ++row) {
                expected[static_cast<size_t>(row + ROWS * (id_pos + N_IDS * token))] *= factor;
            }
        }
    }
    expect_near(actual, expected, tolerance, label);
}

static void run_mul_mat_id_q4_k_swiglu_fusion_case(
        ggml_backend_t backend,
        const char * label,
        int64_t K = 256,
        int64_t ROWS = 8,
        int64_t N_IDS = 2,
        int64_t N_TOKENS = 3,
        int64_t N_EXPERTS = 4,
        float tolerance = 2.0e-3f,
        bool sampled_reference = false) {
    scoped_env_var disable_swiglu("GGML_HRX_DISABLE_SWIGLU", "1");

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate_lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, ROWS, N_EXPERTS);
    ggml_tensor * up_lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, ROWS, N_EXPERTS);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, N_IDS, N_TOKENS);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, N_IDS, N_TOKENS);
    ggml_tensor * gate = ggml_mul_mat_id(ctx.get(), gate_lhs, rhs, ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx.get(), up_lhs, rhs, ids);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), gate, up);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_f32(static_cast<size_t>(K * ROWS * N_EXPERTS));
    std::vector<float> up_f32(gate_f32.size());
    std::vector<float> rhs_f32(static_cast<size_t>(K * N_IDS * N_TOKENS));
    std::vector<int32_t> ids_i32(static_cast<size_t>(N_IDS * N_TOKENS));
    for (size_t i = 0; i < gate_f32.size(); ++i) {
        gate_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 5) % 101) - 50) / 67.0f;
        up_f32[i] = static_cast<float>(static_cast<int>((i * 29 + 11) % 103) - 51) / 71.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 31 + 3) % 107) - 53) / 73.0f;
    }
    for (int64_t token = 0; token < N_TOKENS; ++token) {
        for (int64_t id_pos = 0; id_pos < N_IDS; ++id_pos) {
            ids_i32[static_cast<size_t>(id_pos + N_IDS * token)] =
                static_cast<int32_t>((id_pos * 3 + token) % N_EXPERTS);
        }
    }

    std::vector<float> gate_reference;
    std::vector<float> up_reference;
    std::vector<uint8_t> gate_storage;
    std::vector<uint8_t> up_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, K, ROWS * N_EXPERTS, gate_f32, gate_reference, gate_storage);
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, K, ROWS * N_EXPERTS, up_f32, up_reference, up_storage);
    ggml_backend_tensor_set(gate_lhs, gate_storage.data(), 0, gate_storage.size());
    ggml_backend_tensor_set(up_lhs, up_storage.data(), 0, up_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> actual = tensor_to_float(out);
    if (sampled_reference) {
        const std::vector<int64_t> row_samples = sample_positions(ROWS);
        const std::vector<int64_t> id_samples = sample_positions(N_IDS);
        const std::vector<int64_t> token_samples = sample_positions(N_TOKENS);
        for (const int64_t token : token_samples) {
            for (const int64_t id_pos : id_samples) {
                for (const int64_t row : row_samples) {
                    const size_t idx = static_cast<size_t>(row + ROWS * (id_pos + N_IDS * token));
                    const float gate_expected = reference_mul_mat_id_q4_k_value(
                        gate_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false,
                        row, id_pos, token);
                    const float up_expected = reference_mul_mat_id_q4_k_value(
                        up_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false,
                        row, id_pos, token);
                    const float expected = gate_expected / (1.0f + std::exp(-gate_expected)) * up_expected;
                    const float got = actual[idx];
                    if (std::fabs(got - expected) > tolerance) {
                        std::fprintf(stderr,
                            "%s[%lld,%lld,%lld]: got %.9g expected %.9g tolerance %.9g\n",
                            label,
                            static_cast<long long>(row),
                            static_cast<long long>(id_pos),
                            static_cast<long long>(token),
                            got,
                            expected,
                            tolerance);
                        std::abort();
                    }
                }
            }
        }
        return;
    }

    std::vector<float> gate_expected =
        reference_mul_mat_id_q4_k(gate_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false);
    std::vector<float> up_expected =
        reference_mul_mat_id_q4_k(up_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false);
    for (size_t i = 0; i < gate_expected.size(); ++i) {
        gate_expected[i] = gate_expected[i] / (1.0f + std::exp(-gate_expected[i])) * up_expected[i];
    }
    expect_near(actual, gate_expected, tolerance, label);
}

static void run_mul_mat_id_q4_k_swiglu_sparse_small_prompt_case(ggml_backend_t backend) {
    scoped_env_var expect_provider(
        "GGML_HRX_EXPECT_MUL_MAT_ID_SWIGLU_PROVIDER",
        "hrx_mul_mat_id_q4_k_swiglu_row4_wg64_f32");
    run_mul_mat_id_q4_k_swiglu_fusion_case(
        backend,
        "mul_mat_id_q4_k_swiglu_sparse_small_prompt",
        2048, 64, 8, 9, 256, 8.0e-3f, true);
}

static void run_bf16_mul_mat_swiglu_fusion_case(ggml_backend_t backend, const char * label) {
    scoped_env_var disable_swiglu("GGML_HRX_DISABLE_SWIGLU", "1");
    scoped_env_var disable_wmma("GGML_HRX_DISABLE_BF16_SWIGLU_WMMA16_PROMPT", "1");
    scoped_env_var disable_cols16("GGML_HRX_DISABLE_BF16_SWIGLU_COLS16_PROMPT", "1");

    static constexpr int64_t K = 256;
    static constexpr int64_t ROWS = 16;
    static constexpr int64_t COLS = 512;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, ROWS);
    ggml_tensor * up_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, ROWS);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, COLS);
    ggml_tensor * gate = ggml_mul_mat(ctx.get(), gate_lhs, rhs);
    ggml_tensor * up = ggml_mul_mat(ctx.get(), up_lhs, rhs);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), gate, up);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_f32(static_cast<size_t>(K * ROWS));
    std::vector<float> up_f32(gate_f32.size());
    std::vector<float> rhs_f32(static_cast<size_t>(K * COLS));
    for (size_t i = 0; i < gate_f32.size(); ++i) {
        gate_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 11) % 103) - 51) / 67.0f;
        up_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 5) % 107) - 53) / 71.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 3) % 109) - 54) / 73.0f;
    }

    std::vector<float> gate_reference;
    std::vector<float> up_reference;
    std::vector<uint8_t> gate_storage;
    std::vector<uint8_t> up_storage;
    prepare_mul_mat_lhs(GGML_TYPE_BF16, K, ROWS, gate_f32, gate_reference, gate_storage);
    prepare_mul_mat_lhs(GGML_TYPE_BF16, K, ROWS, up_f32, up_reference, up_storage);
    ggml_backend_tensor_set(gate_lhs, gate_storage.data(), 0, gate_storage.size());
    ggml_backend_tensor_set(up_lhs, up_storage.data(), 0, up_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> gate_expected = reference_mul_mat(gate_reference, rhs_f32, K, ROWS, COLS);
    std::vector<float> up_expected = reference_mul_mat(up_reference, rhs_f32, K, ROWS, COLS);
    for (size_t i = 0; i < gate_expected.size(); ++i) {
        gate_expected[i] = gate_expected[i] / (1.0f + std::exp(-gate_expected[i])) * up_expected[i];
    }
    expect_near(tensor_to_float(out), gate_expected, 3.0e-2f, label);
}

static void run_bf16_mul_mat_set_rows_fusion_case(ggml_backend_t backend, const char * label) {
    scoped_env_var disable_set_rows("GGML_HRX_DISABLE_SET_ROWS", "1");

    static constexpr int64_t K = 256;
    static constexpr int64_t ROWS = 2;
    static constexpr int64_t DST_ROWS = 4;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, ROWS);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, 1);
    ggml_tensor * mul_mat = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * adapter = ggml_reshape_2d(ctx.get(), mul_mat, 1, ROWS);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 1, DST_ROWS);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, ROWS);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, adapter, rows);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(K * ROWS));
    std::vector<float> rhs_f32(static_cast<size_t>(K));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 7) % 97) - 48) / 61.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 3) % 89) - 44) / 59.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_BF16, K, ROWS, lhs_f32, lhs_reference, lhs_storage);
    const std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    const int64_t row_ids[ROWS] = { 2, 0 };

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(rows, row_ids, 0, sizeof(row_ids));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(DST_ROWS, 0.0f);
    const std::vector<float> matmul = reference_mul_mat(lhs_reference, rhs_f32, K, ROWS, 1);
    for (int64_t row = 0; row < ROWS; ++row) {
        expected[static_cast<size_t>(row_ids[row])] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(matmul[row]));
    }
    expect_near(tensor_to_float(out), expected, 0.0f, label);
}

static void run_bf16_mul_mat_vector_set_rows_fusion_case(ggml_backend_t backend, const char * label) {
    scoped_env_var disable_set_rows("GGML_HRX_DISABLE_SET_ROWS", "1");

    static constexpr int64_t K = 2048;
    static constexpr int64_t ROWS = 8;
    static constexpr int64_t DST_ROWS = 3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, ROWS);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, 1);
    ggml_tensor * mul_mat = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * adapter = ggml_reshape_2d(ctx.get(), mul_mat, ROWS, 1);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, ROWS, DST_ROWS);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, adapter, rows);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(K * ROWS));
    std::vector<float> rhs_f32(static_cast<size_t>(K));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 113) - 56) / 83.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 5) % 127) - 63) / 97.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_BF16, K, ROWS, lhs_f32, lhs_reference, lhs_storage);
    const std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    const int64_t row_ids[1] = { 2 };

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(rows, row_ids, 0, sizeof(row_ids));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(ROWS * DST_ROWS), 0.0f);
    const std::vector<float> matmul = reference_mul_mat(lhs_reference, rhs_f32, K, ROWS, 1);
    for (int64_t row = 0; row < ROWS; ++row) {
        expected[static_cast<size_t>(row_ids[0] * ROWS + row)] =
            ggml_fp16_to_fp32(ggml_fp32_to_fp16(matmul[row]));
    }
    expect_near(tensor_to_float(out), expected, 1.0e-3f, label);
}

static void run_q6_k_mul_mat_silu_mul_fusion_case(ggml_backend_t backend, ggml_backend_dev_t dev, const char * label) {
    static constexpr int64_t K = 256;
    static constexpr int64_t ROWS = 8;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q6_K, K, ROWS);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, 1);
    ggml_tensor * mul_mat = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * adapter = ggml_reshape_2d(ctx.get(), mul_mat, 4, ROWS / 4);
    ggml_tensor * silu = ggml_silu(ctx.get(), adapter);
    ggml_tensor * scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, ROWS / 4);
    ggml_tensor * out = ggml_mul(ctx.get(), silu, scale);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(K * ROWS));
    std::vector<float> rhs_f32(static_cast<size_t>(K));
    std::vector<float> scale_f32(static_cast<size_t>(ROWS));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 5) % 101) - 50) / 37.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 89) - 44) / 41.0f;
    }
    for (size_t i = 0; i < scale_f32.size(); ++i) {
        scale_f32[i] = 0.25f + static_cast<float>(i % 7) / 13.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q6_K, K, ROWS, lhs_f32, lhs_reference, lhs_storage);
    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(scale, scale_f32.data(), 0, scale_f32.size() * sizeof(float));

    scoped_env_var disable_silu_mul("GGML_HRX_DISABLE_SILU_MUL_FUSION", "1");
    scoped_env_var disable_silu("GGML_HRX_DISABLE_SILU", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected = reference_mul_mat(lhs_reference, rhs_f32, K, ROWS, 1);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = expected[i] / (1.0f + std::exp(-expected[i])) * scale_f32[i];
    }
    expect_near(tensor_to_float(out), expected, 2.0e-3f, label);
}

static void run_q8_0_mul_mat_add_fusion_case(
        ggml_backend_t backend,
        int64_t k,
        int64_t rows,
        int64_t cols,
        float tolerance,
        const char * label) {
    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(k % ggml_blck_size(GGML_TYPE_Q8_0) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rows, cols);
    ggml_tensor * mm = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * out = ggml_add(ctx.get(), mm, bias);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows));
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols));
    std::vector<float> bias_f32(static_cast<size_t>(rows * cols));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 7) % 101) - 50) / 39.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 13 + 5) % 89) - 44) / 43.0f;
    }
    for (size_t i = 0; i < bias_f32.size(); ++i) {
        bias_f32[i] = static_cast<float>(static_cast<int>((i * 11 + 3) % 31) - 15) / 29.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q8_0, k, rows, lhs_f32, lhs_reference, lhs_storage);
    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(bias, bias_f32.data(), 0, bias_f32.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected = reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] += bias_f32[i];
    }
    expect_near(tensor_to_float(out), expected, tolerance, label);
}

static void expect_flash_attn_ext_prefill_samples(
        const std::vector<float> & actual,
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & mask,
        int64_t d,
        int64_t n,
        int64_t h,
        int64_t h_kv,
        int64_t kv,
        float scale,
        float logit_softcap,
        const char * label) {
    const std::array<int64_t, 5> sample_tokens = { 0, std::min<int64_t>(1, n - 1), n / 3, n / 2, n - 1 };
    const std::array<int64_t, 3> sample_heads = { 0, h / 2, h - 1 };
    const std::array<int64_t, 4> sample_cols = { 0, std::min<int64_t>(31, d - 1), d / 2, d - 1 };
    std::vector<float> logits(static_cast<size_t>(kv));
    std::vector<float> expected(static_cast<size_t>(d));

    for (const int64_t token : sample_tokens) {
        for (const int64_t head : sample_heads) {
            const int64_t kv_head = head / (h / h_kv);
            float max_score = -INFINITY;
            for (int64_t t = 0; t < kv; ++t) {
                float score = 0.0f;
                for (int64_t col = 0; col < d; ++col) {
                    score +=
                        q[index_4d(col, token, head, 0, d, n, h)] *
                        k[index_4d(col, t, kv_head, 0, d, kv, h_kv)];
                }
                score *= scale;
                if (logit_softcap != 0.0f) {
                    score = logit_softcap * std::tanh(score / logit_softcap);
                }
                score += mask[index_4d(t, token, 0, 0, kv, n, 1)];
                logits[static_cast<size_t>(t)] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;
            for (float & logit : logits) {
                logit = std::exp(logit - max_score);
                denom += logit;
            }

            for (int64_t col = 0; col < d; ++col) {
                float value = 0.0f;
                for (int64_t t = 0; t < kv; ++t) {
                    value += logits[static_cast<size_t>(t)] *
                        v[index_4d(col, t, kv_head, 0, d, kv, h_kv)];
                }
                expected[static_cast<size_t>(col)] = value / denom;
            }

            for (const int64_t col : sample_cols) {
                const size_t out_idx = index_4d(col, head, token, 0, d, h, n);
                const float got = actual[out_idx];
                const float want = expected[static_cast<size_t>(col)];
                const float tolerance = logit_softcap != 0.0f ? 6.0e-2f : 2.0e-2f;
                if (std::fabs(got - want) > tolerance) {
                    std::fprintf(stderr,
                        "%s[col=%" PRId64 ",head=%" PRId64 ",token=%" PRId64 "]: got %.9g expected %.9g\n",
                        label, col, head, token, got, want);
                    std::abort();
                }
            }
        }
    }
}

static void run_flash_attn_ext_prefill_f16_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        bool disable_direct,
        bool disable_wmma,
        const char * label,
        float logit_softcap = 0.0f,
        int64_t n = 512,
        int64_t kv = 0) {
    scoped_env_var disable_decode("GGML_HRX_DISABLE_FLASH_ATTN_EXT_DECODE", "1");
    scoped_env_var disable_direct_var(
        "GGML_HRX_DISABLE_F16_PREFILL_FA_DIRECT", disable_direct ? "1" : "0");
    scoped_env_var disable_wmma_var("GGML_HRX_DISABLE_F16_PREFILL_FA_WMMA", disable_wmma ? "1" : "0");
    scoped_env_var disable_tile("GGML_HRX_DISABLE_F16_PREFILL_FA_TILE", "0");

    static constexpr int64_t D = 256;
    static constexpr int64_t H = 16;
    static constexpr int64_t H_KV = 2;
    const int64_t N = n;
    const int64_t KV = kv > 0 ? kv : n;
    static constexpr int64_t S = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, D, N, H, S);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, D, KV, H_KV, S);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, D, KV, H_KV, S);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, KV, N, 1, S);
    ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, scale, 0.0f, logit_softcap);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(D * N * H * S));
    std::vector<float> k_data(static_cast<size_t>(D * KV * H_KV * S));
    std::vector<float> v_data(static_cast<size_t>(D * KV * H_KV * S));
    // Small centered Q/K activations keep scaled attention logits near ordinary
    // model magnitudes, while bounded V values keep reference outputs stable.
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 41) - 20) / 61.0f;
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<float>(static_cast<int>((i * 11 + 5) % 37) - 18) / 67.0f;
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = static_cast<float>(static_cast<int>((i * 13 + 9) % 43) - 21) / 59.0f;
    }

    std::vector<float> k_reference;
    std::vector<float> v_reference;
    std::vector<uint8_t> k_storage;
    std::vector<uint8_t> v_storage;
    prepare_rows(GGML_TYPE_F16, D, KV * H_KV * S, k_data, k_reference, k_storage);
    prepare_rows(GGML_TYPE_F16, D, KV * H_KV * S, v_data, v_reference, v_storage);

    std::vector<float> mask_reference(static_cast<size_t>(KV * N * S));
    std::vector<ggml_fp16_t> mask_storage(mask_reference.size());
    for (int64_t token = 0; token < N; ++token) {
        for (int64_t t = 0; t < KV; ++t) {
            const float value = t > token ? -1000.0f : 0.03125f * static_cast<float>(token - t);
            const size_t idx = index_4d(t, token, 0, 0, KV, N, 1);
            mask_reference[idx] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
            mask_storage[idx] = ggml_fp32_to_fp16(value);
        }
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_storage.data(), 0, k_storage.size());
    ggml_backend_tensor_set(v, v_storage.data(), 0, v_storage.size());
    ggml_backend_tensor_set(mask, mask_storage.data(), 0, mask_storage.size() * sizeof(ggml_fp16_t));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_flash_attn_ext_prefill_samples(
        tensor_to_float(out), q_data, k_reference, v_reference, mask_reference,
        D, N, H, H_KV, KV, scale, logit_softcap, label);
}

struct flash_attn_ext_shape {
    int64_t d;
    int64_t n;
    int64_t h;
    int64_t h_kv;
    int64_t kv;
    int64_t s;
};

enum class flash_attn_ext_phase {
    decode,
    prefill,
};

struct flash_attn_ext_config {
    flash_attn_ext_phase phase;
    flash_attn_ext_shape shape;
    ggml_type k_type;
    ggml_type v_type;
    bool expected_supported;
    bool compute;
    bool mask;
    bool sinks;
    const char * name;
};

static const char * flash_attn_ext_phase_name(flash_attn_ext_phase phase) {
    switch (phase) {
        case flash_attn_ext_phase::decode:
            return "decode";
        case flash_attn_ext_phase::prefill:
            return "prefill";
    }
    return "unknown";
}

static std::string flash_attn_ext_case_label(const flash_attn_ext_config & cfg) {
    const flash_attn_ext_shape & s = cfg.shape;
    std::string label = "flash_attn_ext_";
    label += flash_attn_ext_phase_name(cfg.phase);
    label += "_";
    label += ggml_type_name(cfg.k_type);
    if (cfg.k_type != cfg.v_type) {
        label += "_";
        label += ggml_type_name(cfg.v_type);
    }
    label += "_d" + std::to_string(s.d);
    label += "_n" + std::to_string(s.n);
    label += "_h" + std::to_string(s.h);
    label += "_hkv" + std::to_string(s.h_kv);
    label += "_kv" + std::to_string(s.kv);
    if (!cfg.compute) {
        label += "_support";
    }
    if (!cfg.expected_supported) {
        label += "_unsupported";
    }
    if (cfg.sinks) {
        label += "_sinks";
    }
    if (!cfg.mask) {
        label += "_nomask";
    }
    return label;
}

static void run_flash_attn_ext_config(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const flash_attn_ext_config & cfg) {
    const flash_attn_ext_shape & shape = cfg.shape;
    const int64_t D = shape.d;
    const int64_t N = shape.n;
    const int64_t H = shape.h;
    const int64_t H_KV = shape.h_kv;
    const int64_t KV = shape.kv;
    const int64_t S = shape.s;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const std::string generated_label = cfg.name ? "" : flash_attn_ext_case_label(cfg);
    const char * label = cfg.name ? cfg.name : generated_label.c_str();

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, D, N, H, S);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), cfg.k_type, D, KV, H_KV, S);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), cfg.v_type, D, KV, H_KV, S);
    ggml_tensor * mask = cfg.mask ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, KV, N, 1, S) : nullptr;
    ggml_tensor * sinks = cfg.sinks ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, H) : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, sinks);

    const bool supported = ggml_backend_dev_supports_op(dev, out);
    if (!cfg.expected_supported) {
        if (supported) {
            std::fprintf(stderr,
                "%s: expected unsupported attention shape d=%" PRId64 " n=%" PRId64
                " h=%" PRId64 " h_kv=%" PRId64 " kv=%" PRId64 "\n",
                label, D, N, H, H_KV, KV);
            std::abort();
        }
        return;
    }
    if (!supported) {
        std::fprintf(stderr,
            "%s: expected supported attention shape d=%" PRId64 " n=%" PRId64
            " h=%" PRId64 " h_kv=%" PRId64 " kv=%" PRId64 "\n",
            label, D, N, H, H_KV, KV);
        std::abort();
    }
    if (!cfg.compute) {
        return;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(D * N * H * S));
    std::vector<float> k_data(static_cast<size_t>(D * KV * H_KV * S));
    std::vector<float> v_data(static_cast<size_t>(D * KV * H_KV * S));
    // Small centered Q/K activations keep scaled attention logits near ordinary
    // model magnitudes, while bounded V values keep reference outputs stable.
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 41) - 20) / 61.0f;
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<float>(static_cast<int>((i * 11 + 5) % 37) - 18) / 67.0f;
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = static_cast<float>(static_cast<int>((i * 13 + 9) % 43) - 21) / 59.0f;
    }

    std::vector<float> k_reference;
    std::vector<float> v_reference;
    std::vector<uint8_t> k_storage;
    std::vector<uint8_t> v_storage;
    prepare_rows(cfg.k_type, D, KV * H_KV * S, k_data, k_reference, k_storage);
    prepare_rows(cfg.v_type, D, KV * H_KV * S, v_data, v_reference, v_storage);

    std::vector<float> mask_reference;
    std::vector<ggml_fp16_t> mask_storage;
    if (cfg.mask) {
        mask_reference.resize(static_cast<size_t>(KV * N * S));
        mask_storage.resize(mask_reference.size());
        for (int64_t seq = 0; seq < S; ++seq) {
            for (int64_t token = 0; token < N; ++token) {
                for (int64_t t = 0; t < KV; ++t) {
                    const float value = t > token ? -1000.0f : 0.03125f * static_cast<float>(token - t);
                    const size_t idx = index_4d(t, token, 0, seq, KV, N, 1);
                    mask_reference[idx] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
                    mask_storage[idx] = ggml_fp32_to_fp16(value);
                }
            }
        }
    }

    std::vector<float> sink_data;
    if (cfg.sinks) {
        sink_data.resize(static_cast<size_t>(H));
        for (int64_t head = 0; head < H; ++head) {
            sink_data[static_cast<size_t>(head)] = -0.5f + 0.25f * static_cast<float>(head);
        }
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_storage.data(), 0, k_storage.size());
    ggml_backend_tensor_set(v, v_storage.data(), 0, v_storage.size());
    if (cfg.mask) {
        ggml_backend_tensor_set(mask, mask_storage.data(), 0, mask_storage.size() * sizeof(ggml_fp16_t));
    }
    if (cfg.sinks) {
        ggml_backend_tensor_set(sinks, sink_data.data(), 0, sink_data.size() * sizeof(float));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    if (cfg.phase == flash_attn_ext_phase::decode) {
        expect_near(
            tensor_to_float(out),
            reference_flash_attn_ext_decode(
                q_data, k_reference, v_reference, mask_reference, sink_data,
                D, N, H, H_KV, KV, S, scale, cfg.mask, cfg.sinks),
            5.0e-3f,
            label);
    } else if (cfg.phase == flash_attn_ext_phase::prefill) {
        expect_flash_attn_ext_prefill_samples(
            tensor_to_float(out), q_data, k_reference, v_reference, mask_reference,
            D, N, H, H_KV, KV, scale, 0.0f, label);
    }
}

static void run_flash_attn_ext_cases(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const flash_attn_ext_config * cases,
        size_t ncases) {
    for (size_t i = 0; i < ncases; ++i) {
        run_flash_attn_ext_config(backend, dev, cases[i]);
    }
}

template <size_t N>
static void run_flash_attn_ext_cases(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const std::array<flash_attn_ext_config, N> & cases) {
    run_flash_attn_ext_cases(backend, dev, cases.data(), cases.size());
}

struct flash_attn_ext_decode_kv_limit_case {
    ggml_type k_type;
    ggml_type v_type;
    bool kv8192_supported;
};

static void run_flash_attn_ext_decode_kv_limit_matrix(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        const flash_attn_ext_decode_kv_limit_case * cases,
        size_t ncases) {
    for (size_t i = 0; i < ncases; ++i) {
        const flash_attn_ext_decode_kv_limit_case & c = cases[i];
        run_flash_attn_ext_config(backend, dev, {
            flash_attn_ext_phase::decode,
            { 128, 1, 32, 8, 1024, 1 },
            c.k_type,
            c.v_type,
            true,
            false,
            true,
            false,
            nullptr,
        });
        run_flash_attn_ext_config(backend, dev, {
            flash_attn_ext_phase::decode,
            { 128, 1, 32, 8, 8192, 1 },
            c.k_type,
            c.v_type,
            c.kv8192_supported,
            false,
            true,
            false,
            nullptr,
        });
    }

    if (!env_enabled("GGML_HRX_TEST_FUTURE_FLASH_ATTN_LONG_KV")) {
        return;
    }

    for (size_t i = 0; i < ncases; ++i) {
        const flash_attn_ext_decode_kv_limit_case & c = cases[i];
        run_flash_attn_ext_config(backend, dev, {
            flash_attn_ext_phase::decode,
            { 128, 1, 32, 8, 8192, 1 },
            c.k_type,
            c.v_type,
            true,
            false,
            true,
            false,
            nullptr,
        });
    }
}

static float ssm_conv_update_value(
        const std::vector<float> & conv_state,
        const std::vector<float> & input,
        int64_t state_width,
        int64_t n_tokens,
        int64_t channel,
        int64_t logical_pos) {
    if (logical_pos < state_width) {
        return conv_state[static_cast<size_t>(channel * state_width + logical_pos)];
    }
    return input[static_cast<size_t>(channel * n_tokens + logical_pos - state_width)];
}

static void run_ssm_conv_update_case(
        ggml_backend_t backend,
        int64_t d_conv,
        int64_t d_inner,
        int64_t n_tokens,
        bool apply_silu,
        bool strided_conv_state = false,
        bool force_decode_fusion = false,
        bool non_adjacent_update = false) {
    ggml_context_ptr ctx = make_context();
    const int64_t state_width = d_conv - 1;
    ggml_tensor * conv_state_base = strided_conv_state ?
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_inner, state_width) : nullptr;
    ggml_tensor * conv_state = strided_conv_state ?
        ggml_transpose(ctx.get(), conv_state_base) :
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, state_width, d_inner);
    if (strided_conv_state) {
        GGML_ASSERT(conv_state->nb[0] != sizeof(float));
    }
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_tokens, d_inner);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_conv, d_inner);
    ggml_tensor * concat = ggml_concat(ctx.get(), conv_state, input, 0);
    ggml_tensor * state_view = ggml_view_2d(
        ctx.get(), concat, state_width, d_inner, concat->nb[1], static_cast<size_t>(n_tokens) * sizeof(float));
    ggml_tensor * state_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, state_width, d_inner);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    ggml_tensor * ssm = ggml_ssm_conv(ctx.get(), concat, weight);
    ggml_tensor * out = apply_silu ? ggml_silu(ctx.get(), ssm) : ssm;
    ggml_tensor * interposer = non_adjacent_update ? ggml_scale(ctx.get(), input, 0.5f) : nullptr;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    if (non_adjacent_update) {
        ggml_build_forward_expand(graph, out);
        ggml_build_forward_expand(graph, interposer);
        ggml_build_forward_expand(graph, state_update);
    } else {
        ggml_build_forward_expand(graph, state_update);
        ggml_build_forward_expand(graph, out);
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> conv_state_data(static_cast<size_t>(state_width * d_inner));
    std::vector<float> input_data(static_cast<size_t>(n_tokens * d_inner));
    std::vector<float> weight_data(static_cast<size_t>(d_conv * d_inner));
    for (size_t i = 0; i < conv_state_data.size(); ++i) {
        conv_state_data[i] = static_cast<float>((i * 5) % 17) * 0.07f - 0.45f;
    }
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>((i * 7) % 23) * 0.05f - 0.5f;
    }
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((i * 11) % 19) * 0.04f - 0.35f;
    }
    if (strided_conv_state) {
        std::vector<float> conv_state_base_data(static_cast<size_t>(state_width * d_inner));
        for (int64_t channel = 0; channel < d_inner; ++channel) {
            for (int64_t i = 0; i < state_width; ++i) {
                conv_state_base_data[static_cast<size_t>(i * d_inner + channel)] =
                    conv_state_data[static_cast<size_t>(channel * state_width + i)];
            }
        }
        ggml_backend_tensor_set(
            conv_state_base, conv_state_base_data.data(), 0, conv_state_base_data.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(conv_state, conv_state_data.data(), 0, conv_state_data.size() * sizeof(float));
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_concat("GGML_HRX_DISABLE_CONCAT", force_decode_fusion ? "1" : "0");
    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", force_decode_fusion ? "1" : "0");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected_out(static_cast<size_t>(d_inner * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t channel = 0; channel < d_inner; ++channel) {
            float sum = 0.0f;
            for (int64_t i = 0; i < d_conv; ++i) {
                const float x = ssm_conv_update_value(
                    conv_state_data, input_data, state_width, n_tokens, channel, token + i);
                const float w = weight_data[static_cast<size_t>(channel * d_conv + i)];
                sum += x * w;
            }
            if (apply_silu) {
                sum = sum / (1.0f + std::exp(-sum));
            }
            expected_out[static_cast<size_t>(token * d_inner + channel)] = sum;
        }
    }

    std::vector<float> expected_state(static_cast<size_t>(state_width * d_inner));
    for (int64_t channel = 0; channel < d_inner; ++channel) {
        for (int64_t i = 0; i < state_width; ++i) {
            expected_state[static_cast<size_t>(channel * state_width + i)] = ssm_conv_update_value(
                conv_state_data, input_data, state_width, n_tokens, channel, n_tokens + i);
        }
    }

    const char * out_label = force_decode_fusion ?
        (non_adjacent_update ? "ssm_conv_update_decode_non_adjacent_fusion_out" : "ssm_conv_update_decode_fusion_out") :
        "ssm_conv_update_fallback_out";
    const char * state_label = force_decode_fusion ?
        (non_adjacent_update ? "ssm_conv_update_decode_non_adjacent_fusion_state" :
                               "ssm_conv_update_decode_fusion_state") :
        "ssm_conv_update_fallback_state";
    expect_near(tensor_to_float(out), expected_out, apply_silu ? 2.0e-5f : 2.0e-6f, out_label);
    expect_near(tensor_to_float(state_update), expected_state, 0.0f, state_label);
}

static void run_ssm_conv_update_gather_case(ggml_backend_t backend, bool apply_silu, bool scaled_state) {
    ggml_context_ptr ctx = make_context();
    constexpr int64_t D_CONV = 4;
    constexpr int64_t STATE_WIDTH = D_CONV - 1;
    constexpr int64_t D_INNER = 17;
    constexpr int64_t STATE_ROWS = 5;
    constexpr int64_t N_TOKENS = 1;
    constexpr int64_t FLAT_STATE_COLS = STATE_WIDTH * D_INNER;
    constexpr int32_t SELECTED_ROW = 3;
    constexpr float SCALE = 0.375f;
    constexpr float BIAS = -0.0625f;

    ggml_tensor * state_base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, FLAT_STATE_COLS, STATE_ROWS);
    ggml_tensor * state_idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * state_src = scaled_state ?
        ggml_scale_bias_inplace(ctx.get(), state_base, SCALE, BIAS) : state_base;
    ggml_tensor * state_row = ggml_get_rows(ctx.get(), state_src, state_idx);
    ggml_tensor * conv_state = ggml_reshape_2d(ctx.get(), state_row, STATE_WIDTH, D_INNER);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_TOKENS, D_INNER);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D_CONV, D_INNER);
    ggml_tensor * concat = ggml_concat(ctx.get(), conv_state, input, 0);
    ggml_tensor * state_view = ggml_view_2d(
        ctx.get(), concat, STATE_WIDTH, D_INNER, concat->nb[1], static_cast<size_t>(N_TOKENS) * sizeof(float));
    ggml_tensor * state_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_WIDTH, D_INNER);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    ggml_tensor * ssm = ggml_ssm_conv(ctx.get(), concat, weight);
    ggml_tensor * out = apply_silu ? ggml_silu(ctx.get(), ssm) : ssm;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> state_base_data(static_cast<size_t>(FLAT_STATE_COLS * STATE_ROWS));
    std::vector<float> input_data(static_cast<size_t>(N_TOKENS * D_INNER));
    std::vector<float> weight_data(static_cast<size_t>(D_CONV * D_INNER));
    for (size_t i = 0; i < state_base_data.size(); ++i) {
        state_base_data[i] = static_cast<float>((i * 13) % 41) * 0.025f - 0.5f;
    }
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>((i * 7) % 29) * 0.03125f - 0.375f;
    }
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((i * 11) % 23) * 0.04f - 0.35f;
    }

    ggml_backend_tensor_set(state_base, state_base_data.data(), 0, state_base_data.size() * sizeof(float));
    ggml_backend_tensor_set(state_idx, &SELECTED_ROW, 0, sizeof(SELECTED_ROW));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var expect_provider(
        "GGML_HRX_EXPECT_SSM_CONV_UPDATE_PROVIDER",
        scaled_state ?
            "hrx_ssm_conv_update_gather_decode_f32" :
            "hrx_ssm_conv_update_gather_decode_direct_f32");
    scoped_env_var disable_scale_get_rows("GGML_HRX_DISABLE_SCALE_GET_ROWS_FUSION", "1");
    scoped_env_var disable_concat("GGML_HRX_DISABLE_CONCAT", "1");
    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> gathered_state(static_cast<size_t>(STATE_WIDTH * D_INNER));
    for (int64_t channel = 0; channel < D_INNER; ++channel) {
        for (int64_t i = 0; i < STATE_WIDTH; ++i) {
            const int64_t flat_col = channel * STATE_WIDTH + i;
            const float raw = state_base_data[static_cast<size_t>(SELECTED_ROW * FLAT_STATE_COLS + flat_col)];
            gathered_state[static_cast<size_t>(flat_col)] = scaled_state ? raw * SCALE + BIAS : raw;
        }
    }

    std::vector<float> expected_out(static_cast<size_t>(D_INNER * N_TOKENS));
    for (int64_t channel = 0; channel < D_INNER; ++channel) {
        float sum = 0.0f;
        for (int64_t i = 0; i < D_CONV; ++i) {
            const float x = i < STATE_WIDTH ?
                gathered_state[static_cast<size_t>(channel * STATE_WIDTH + i)] :
                input_data[static_cast<size_t>(channel * N_TOKENS + i - STATE_WIDTH)];
            const float w = weight_data[static_cast<size_t>(channel * D_CONV + i)];
            sum += x * w;
        }
        if (apply_silu) {
            sum = sum / (1.0f + std::exp(-sum));
        }
        expected_out[static_cast<size_t>(channel)] = sum;
    }

    std::vector<float> expected_state(static_cast<size_t>(STATE_WIDTH * D_INNER));
    for (int64_t channel = 0; channel < D_INNER; ++channel) {
        for (int64_t i = 0; i < STATE_WIDTH; ++i) {
            expected_state[static_cast<size_t>(channel * STATE_WIDTH + i)] =
                i + 1 < STATE_WIDTH ?
                    gathered_state[static_cast<size_t>(channel * STATE_WIDTH + i + 1)] :
                    input_data[static_cast<size_t>(channel)];
        }
    }

    expect_near(
        tensor_to_float(out), expected_out, apply_silu ? 2.0e-5f : 2.0e-6f,
        apply_silu ?
            (scaled_state ? "ssm_conv_update_gather_scaled_silu_out" : "ssm_conv_update_gather_direct_silu_out") :
            (scaled_state ? "ssm_conv_update_gather_scaled_out" : "ssm_conv_update_gather_direct_out"));
    expect_near(
        tensor_to_float(state_update),
        expected_state,
        1.0e-6f,
        scaled_state ? "ssm_conv_update_gather_scaled_state" : "ssm_conv_update_gather_direct_state");
}

static void run_ssm_conv_update_negative_state_offset_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    constexpr int64_t D_CONV = 4;
    constexpr int64_t STATE_WIDTH = D_CONV - 1;
    constexpr int64_t D_INNER = 5;
    constexpr int64_t N_TOKENS = 1;
    ggml_tensor * conv_state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_WIDTH, D_INNER);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_TOKENS, D_INNER);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D_CONV, D_INNER);
    ggml_tensor * concat = ggml_concat(ctx.get(), conv_state, input, 0);
    ggml_tensor * state_view = ggml_view_2d(ctx.get(), concat, STATE_WIDTH, D_INNER, concat->nb[1], 0);
    ggml_tensor * state_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_WIDTH, D_INNER);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    ggml_tensor * out = ggml_silu(ctx.get(), ggml_ssm_conv(ctx.get(), concat, weight));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> data(static_cast<size_t>(N_TOKENS * D_INNER), 0.25f);
    std::vector<float> state_data(static_cast<size_t>(STATE_WIDTH * D_INNER), -0.125f);
    std::vector<float> weight_data(static_cast<size_t>(D_CONV * D_INNER), 0.0625f);
    ggml_backend_tensor_set(conv_state, state_data.data(), 0, state_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_concat("GGML_HRX_DISABLE_CONCAT", "1");
    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_ssm_conv_update_negative_state_overlap_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    constexpr int64_t D_CONV = 4;
    constexpr int64_t STATE_WIDTH = D_CONV - 1;
    constexpr int64_t D_INNER = 5;
    constexpr int64_t N_TOKENS = 1;
    ggml_tensor * conv_state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_WIDTH, D_INNER);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_TOKENS, D_INNER);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D_CONV, D_INNER);
    ggml_tensor * concat = ggml_concat(ctx.get(), conv_state, input, 0);
    ggml_tensor * state_view = ggml_view_2d(
        ctx.get(), concat, STATE_WIDTH, D_INNER, concat->nb[1], static_cast<size_t>(N_TOKENS) * sizeof(float));
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, conv_state);
    ggml_tensor * out = ggml_silu(ctx.get(), ggml_ssm_conv(ctx.get(), concat, weight));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> data(static_cast<size_t>(N_TOKENS * D_INNER), 0.25f);
    std::vector<float> state_data(static_cast<size_t>(STATE_WIDTH * D_INNER), -0.125f);
    std::vector<float> weight_data(static_cast<size_t>(D_CONV * D_INNER), 0.0625f);
    ggml_backend_tensor_set(conv_state, state_data.data(), 0, state_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_concat("GGML_HRX_DISABLE_CONCAT", "1");
    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_add8_fusion_case(ggml_backend_t backend, int64_t ncols, int64_t nrows) {
    ggml_context_ptr ctx = make_context();
    static constexpr int ARITY = 8;
    const int64_t base_cols = ncols + 3;
    std::array<ggml_tensor *, ARITY> bases = {};
    std::array<ggml_tensor *, ARITY> views = {};
    for (int i = 0; i < ARITY; ++i) {
        bases[i] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, base_cols, nrows);
        views[i] = ggml_view_2d(
            ctx.get(), bases[i], ncols, nrows, bases[i]->nb[1], static_cast<size_t>(i % 3) * sizeof(float));
    }

    ggml_tensor * sum = ggml_add(ctx.get(), views[0], views[1]);
    for (int i = 2; i < ARITY; ++i) {
        sum = ggml_add(ctx.get(), sum, views[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    for (int i = 0; i < ARITY; ++i) {
        ggml_build_forward_expand(graph, views[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<std::vector<float>> base_data;
    base_data.reserve(ARITY);
    std::vector<float> expected(static_cast<size_t>(ncols * nrows), 0.0f);
    for (int i = 0; i < ARITY; ++i) {
        base_data.emplace_back(static_cast<size_t>(base_cols * nrows));
        for (int64_t row = 0; row < nrows; ++row) {
            for (int64_t col = 0; col < base_cols; ++col) {
                base_data.back()[static_cast<size_t>(col + base_cols * row)] =
                    static_cast<float>(((col + 7 * row + 11 * i) % 29) - 14) / 17.0f;
            }
            for (int64_t col = 0; col < ncols; ++col) {
                expected[static_cast<size_t>(col + ncols * row)] +=
                    base_data.back()[static_cast<size_t>(col + (i % 3) + base_cols * row)];
            }
        }
        ggml_backend_tensor_set(bases[i], base_data.back().data(), 0, base_data.back().size() * sizeof(float));
    }

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(sum), expected, 1.0e-6f, "add8_fusion");
}

static void run_mul_sum8_fusion_case(ggml_backend_t backend, int64_t rows, int64_t tokens) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);

    std::array<ggml_tensor *, 8> slices = {};
    for (int i = 0; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }

    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, product);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> values_data(static_cast<size_t>(rows * 8 * tokens));
    std::vector<float> scales_data(static_cast<size_t>(8 * tokens));
    for (size_t i = 0; i < values_data.size(); ++i) {
        values_data[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) / 19.0f;
    }
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales_data[i] = static_cast<float>(static_cast<int>((i * 7 + 2) % 23) - 11) / 13.0f;
    }

    std::vector<float> expected(static_cast<size_t>(rows * tokens), 0.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t row = 0; row < rows; ++row) {
            float acc = 0.0f;
            for (int64_t expert = 0; expert < 8; ++expert) {
                const size_t value_idx = static_cast<size_t>(row + rows * (expert + 8 * token));
                const size_t scale_idx = static_cast<size_t>(expert + 8 * token);
                acc += values_data[value_idx] * scales_data[scale_idx];
            }
            expected[static_cast<size_t>(row + rows * token)] = acc;
        }
    }

    ggml_backend_tensor_set(values, values_data.data(), 0, values_data.size() * sizeof(float));
    ggml_backend_tensor_set(scales, scales_data.data(), 0, scales_data.size() * sizeof(float));

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(sum), expected, 2.0e-6f, "mul_sum8_fusion");
}

static void set_mul_sum8_inputs(ggml_tensor * values, ggml_tensor * scales, int64_t rows, int64_t tokens) {
    std::vector<float> values_data(static_cast<size_t>(rows * 8 * tokens));
    std::vector<float> scales_data(static_cast<size_t>(8 * tokens));
    for (size_t i = 0; i < values_data.size(); ++i) {
        values_data[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) / 19.0f;
    }
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales_data[i] = static_cast<float>(static_cast<int>((i * 7 + 2) % 23) - 11) / 13.0f;
    }
    ggml_backend_tensor_set(values, values_data.data(), 0, values_data.size() * sizeof(float));
    ggml_backend_tensor_set(scales, scales_data.data(), 0, scales_data.size() * sizeof(float));
}

static void expect_graph_failure(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s unexpectedly succeeded\n", label);
        std::abort();
    }
}

static void run_mul_sum8_negative_intervening_op_case(ggml_backend_t backend) {
    static constexpr int64_t rows = 17;
    static constexpr int64_t tokens = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);
    ggml_tensor * intervening_add = ggml_add(ctx.get(), scales, scales);

    std::array<ggml_tensor *, 8> slices = {};
    for (int i = 0; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }
    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 40, false);
    ggml_build_forward_expand(graph, product);
    ggml_build_forward_expand(graph, intervening_add);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    set_mul_sum8_inputs(values, scales, rows, tokens);

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    expect_graph_failure(backend, graph, "mul_sum8_intervening_op");
}

static void run_mul_sum8_negative_extra_product_consumer_case(ggml_backend_t backend) {
    static constexpr int64_t rows = 17;
    static constexpr int64_t tokens = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);

    std::array<ggml_tensor *, 8> slices = {};
    for (int i = 0; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }
    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }
    ggml_tensor * extra_consumer = ggml_scale_bias(ctx.get(), product, 1.0f, 0.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 40, false);
    ggml_build_forward_expand(graph, product);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);
    ggml_build_forward_expand(graph, extra_consumer);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    set_mul_sum8_inputs(values, scales, rows, tokens);

    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    expect_graph_failure(backend, graph, "mul_sum8_extra_product_consumer");
}

static void run_mul_sum8_negative_duplicate_slice_case(ggml_backend_t backend) {
    static constexpr int64_t rows = 17;
    static constexpr int64_t tokens = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);

    std::array<ggml_tensor *, 8> slices = {};
    slices[0] = ggml_view_2d(ctx.get(), product, rows, tokens, product->nb[2], 0);
    slices[1] = ggml_view_2d(ctx.get(), product, rows, tokens, product->nb[2], 0);
    for (int i = 2; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }

    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 40, false);
    ggml_build_forward_expand(graph, product);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    set_mul_sum8_inputs(values, scales, rows, tokens);

    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    expect_graph_failure(backend, graph, "mul_sum8_duplicate_slice");
}

static void run_mul_add_add_fusion_case(ggml_backend_t backend, int64_t ncols, int64_t nrows) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, nrows);
    ggml_tensor * bias0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, 1);
    ggml_tensor * bias1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * product = ggml_mul(ctx.get(), lhs, scale);
    ggml_tensor * first_add = ggml_add(ctx.get(), product, bias0);
    ggml_tensor * second_add = ggml_add(ctx.get(), first_add, bias1);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, second_add);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(static_cast<size_t>(ncols * nrows));
    std::vector<float> scale_data(static_cast<size_t>(nrows));
    std::vector<float> bias0_data(static_cast<size_t>(ncols));
    std::vector<float> bias1_data(static_cast<size_t>(ncols * nrows));
    for (size_t i = 0; i < lhs_data.size(); ++i) {
        lhs_data[i] = static_cast<float>(static_cast<int>((i * 13 + 1) % 41) - 20) / 23.0f;
    }
    for (size_t i = 0; i < scale_data.size(); ++i) {
        scale_data[i] = static_cast<float>(static_cast<int>((i * 3 + 5) % 17) - 8) / 11.0f;
    }
    for (size_t i = 0; i < bias0_data.size(); ++i) {
        bias0_data[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 19) - 9) / 29.0f;
    }
    for (size_t i = 0; i < bias1_data.size(); ++i) {
        bias1_data[i] = static_cast<float>(static_cast<int>((i * 11 + 4) % 31) - 15) / 31.0f;
    }

    std::vector<float> expected(lhs_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            const size_t idx = static_cast<size_t>(col + ncols * row);
            expected[idx] = lhs_data[idx] * scale_data[static_cast<size_t>(row)] +
                bias0_data[static_cast<size_t>(col)] + bias1_data[idx];
        }
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(scale, scale_data.data(), 0, scale_data.size() * sizeof(float));
    ggml_backend_tensor_set(bias0, bias0_data.data(), 0, bias0_data.size() * sizeof(float));
    ggml_backend_tensor_set(bias1, bias1_data.data(), 0, bias1_data.size() * sizeof(float));

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(second_add), expected, 1.0e-6f, "mul_add_add_fusion");
}

} // namespace

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HRX0");
    if (!dev) {
        std::fprintf(stderr, "HRX0 not available; skipping test-backend-hrx\n");
        return 0;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    GGML_ASSERT(buft != nullptr);
    {
        ggml_context_ptr standalone_ctx = make_context();
        ggml_tensor * standalone = ggml_new_tensor_1d(standalone_ctx.get(), GGML_TYPE_F32, 4);
        ggml_backend_buffer_ptr standalone_buffer(ggml_backend_alloc_ctx_tensors_from_buft(standalone_ctx.get(), buft));
        GGML_ASSERT(standalone_buffer != nullptr);

        const std::vector<float> standalone_input = { 10.0f, 11.0f, 12.0f, 13.0f };
        ggml_backend_tensor_set(standalone, standalone_input.data(), 0, standalone_input.size() * sizeof(float));

        std::vector<float> standalone_output(standalone_input.size(), -1.0f);
        ggml_backend_tensor_get(standalone, standalone_output.data(), 0, standalone_output.size() * sizeof(float));
        expect_eq(standalone_output, standalone_input, "standalone_output");
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend != nullptr);

    if (const char * test_only = std::getenv("GGML_HRX_TEST_ONLY")) {
        if (std::strcmp(test_only, "catalog_mul_mat_q4_k_f32") == 0) {
            run_catalog_mul_mat_q4_k_f32_case(backend.get(), dev);
            return 0;
        }
        if (std::strcmp(test_only, "live_kv_set_rows") == 0) {
            run_live_kv_set_rows_update_case(backend.get(), dev);
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::strcmp(test_only, "live_kv_set_rows_flat") == 0) {
            run_live_kv_set_rows_flat_case(backend.get(), dev);
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::strcmp(test_only, "live_kv_kqv_direct") == 0) {
            run_live_kv_kqv_direct_case(backend.get(), dev);
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::strcmp(test_only, "live_rope_normal_freq") == 0) {
            run_live_rope_normal_freq_case(backend.get(), dev);
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::strcmp(test_only, "live_rope_normal_freq_inplace") == 0) {
            run_live_rope_normal_freq_inplace_case(backend.get(), dev);
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::strcmp(test_only, "live_rope_normal_freq_reused_producer") == 0) {
            run_live_rope_normal_freq_reused_producer_case(backend.get(), dev);
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::strcmp(test_only, "live_rope_normal_freq_reused_producer_sched") == 0) {
            run_live_rope_normal_freq_reused_producer_sched_case(backend.get(), dev);
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        static constexpr const char * prefix = "catalog_";
        if (std::strncmp(test_only, prefix, std::strlen(prefix)) == 0) {
            const std::vector<ggml_backend_hrx_test_case> cases =
                ggml_backend_hrx_test_cases(dev, test_only + std::strlen(prefix));
            GGML_ASSERT(!cases.empty());
            for (const ggml_backend_hrx_test_case & cfg : cases) {
                run_catalog_schedule_case(backend.get(), dev, cfg);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        std::fprintf(stderr, "unknown GGML_HRX_TEST_ONLY=%s\n", test_only);
        return 1;
    }

    const std::vector<ggml_backend_hrx_test_case> cases = ggml_backend_hrx_test_cases(dev, "");
    GGML_ASSERT(!cases.empty());
    run_live_rope_normal_freq_inplace_case(backend.get(), dev);
    for (const ggml_backend_hrx_test_case & cfg : cases) {
        run_catalog_schedule_case(backend.get(), dev, cfg);
    }
    ggml_backend_synchronize(backend.get());
    return 0;

// HRX3 backend coverage is catalog-driven. The legacy hard-coded HRX1/HRX2
// matrix below is intentionally disabled until each case is reintroduced via
// device-specific catalog test schedules emitted by the kernel bench.
#if 0
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor * view = ggml_view_1d(ctx.get(), src, 4, 2 * sizeof(float));
    ggml_tensor * dst  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(src->buffer == buffer.get());
    GGML_ASSERT(view->buffer == buffer.get());
    GGML_ASSERT(dst->buffer == buffer.get());

    const std::vector<float> input = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));

    std::vector<float> view_data(4, -1.0f);
    ggml_backend_tensor_get(view, view_data.data(), 0, view_data.size() * sizeof(float));
    expect_eq(view_data, { 2.0f, 3.0f, 4.0f, 5.0f }, "view_data");

    const std::vector<float> replacement = { 20.0f, 21.0f, 22.0f, 23.0f };
    ggml_backend_tensor_set(view, replacement.data(), 0, replacement.size() * sizeof(float));

    std::vector<float> src_after_view_set(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_view_set.data(), 0, src_after_view_set.size() * sizeof(float));
    expect_eq(src_after_view_set, { 0.0f, 1.0f, 20.0f, 21.0f, 22.0f, 23.0f, 6.0f, 7.0f }, "src_after_view_set");

    ggml_backend_tensor_copy(view, dst);
    std::vector<float> dst_data(4, -1.0f);
    ggml_backend_tensor_get(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));
    expect_eq(dst_data, replacement, "dst_data");

    ggml_backend_tensor_memset(view, 0, 0, ggml_nbytes(view));
    std::vector<float> src_after_memset(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_memset.data(), 0, src_after_memset.size() * sizeof(float));
    expect_eq(src_after_memset, { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 7.0f }, "src_after_memset");

    ggml_backend_buffer_clear(buffer.get(), 0);
    std::vector<float> src_after_clear(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_clear.data(), 0, src_after_clear.size() * sizeof(float));
    expect_eq(src_after_clear, std::vector<float>(8, 0.0f), "src_after_clear");

    ggml_context_ptr graph_ctx = make_context();
    ggml_tensor * graph_base      = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor * graph_view      = ggml_view_1d(graph_ctx.get(), graph_base, 4, 4 * sizeof(float));
    ggml_tensor * graph_reshape   = ggml_reshape_3d(graph_ctx.get(), graph_base, 2, 2, 3);
    ggml_tensor * graph_permute   = ggml_permute(graph_ctx.get(), graph_reshape, 1, 0, 2, 3);
    ggml_tensor * graph_transpose = ggml_transpose(graph_ctx.get(), graph_base);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 16, false);
    ggml_build_forward_expand(graph, graph_view);
    ggml_build_forward_expand(graph, graph_reshape);
    ggml_build_forward_expand(graph, graph_permute);
    ggml_build_forward_expand(graph, graph_transpose);

    ggml_backend_buffer_ptr graph_buffer(ggml_backend_alloc_ctx_tensors(graph_ctx.get(), backend.get()));
    GGML_ASSERT(graph_buffer != nullptr);

    const auto * graph_base_data = static_cast<const uint8_t *>(graph_base->data);
    GGML_ASSERT(graph_view->data == graph_base_data + 4 * sizeof(float));
    GGML_ASSERT(graph_reshape->data == graph_base->data);
    GGML_ASSERT(graph_permute->data == graph_base->data);
    GGML_ASSERT(graph_transpose->data == graph_base->data);

    const std::vector<float> graph_input = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
    };
    ggml_backend_tensor_set(graph_base, graph_input.data(), 0, graph_input.size() * sizeof(float));

    const ggml_status graph_status = ggml_backend_graph_compute(backend.get(), graph);
    if (graph_status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "metadata graph failed: %s\n", ggml_status_to_string(graph_status));
        std::abort();
    }

    std::vector<float> graph_base_after(graph_input.size(), -1.0f);
    ggml_backend_tensor_get(graph_base, graph_base_after.data(), 0, graph_base_after.size() * sizeof(float));
    expect_eq(graph_base_after, graph_input, "graph_base_after");

    std::vector<float> graph_view_data(4, -1.0f);
    ggml_backend_tensor_get(graph_view, graph_view_data.data(), 0, graph_view_data.size() * sizeof(float));
    expect_eq(graph_view_data, { 4.0f, 5.0f, 6.0f, 7.0f }, "graph_view_data");

    std::vector<float> graph_reshape_data(graph_input.size(), -1.0f);
    ggml_backend_tensor_get(graph_reshape, graph_reshape_data.data(), 0, graph_reshape_data.size() * sizeof(float));
    expect_eq(graph_reshape_data, graph_input, "graph_reshape_data");

    run_add_case(backend.get(), 1);
    run_add_case(backend.get(), 255);
    run_add_case(backend.get(), 256);
    run_add_case(backend.get(), 257);
    run_add_case(backend.get(), 1025);
    run_add_activation_boundary_case(dev);
    run_mul_case(backend.get(), 1);
    run_mul_case(backend.get(), 255);
    run_mul_case(backend.get(), 256);
    run_mul_case(backend.get(), 257);
    run_mul_case(backend.get(), 1025);
    run_broadcast_case(backend.get(), GGML_OP_ADD);
    run_broadcast_case(backend.get(), GGML_OP_MUL);
    run_broadcast_case(backend.get(), GGML_OP_DIV);
    run_scale_case(backend.get(), 1);
    run_scale_case(backend.get(), 256);
    run_scale_case(backend.get(), 257);
    run_cpy_strided_case(backend.get());
    run_cpy_f32_f16_case(backend.get());
    run_cont_slice_case(backend.get());
    run_set_rows_case(backend.get(), GGML_TYPE_F32);
    run_set_rows_case(backend.get(), GGML_TYPE_F16);
    run_set_rows_case(backend.get(), GGML_TYPE_Q8_0);
    run_set_rows_q8_0_tie_case(backend.get());
    run_set_rows_case(backend.get(), GGML_TYPE_Q4_0);
    run_get_rows_case(backend.get(), 1);
    run_get_rows_case(backend.get(), 3);
    run_scale_get_rows_fusion_case(backend.get());
    run_get_rows_q5_k_case(backend.get(), ggml_blck_size(GGML_TYPE_Q5_K), 4, 1, 2);
    run_get_rows_q5_k_case(backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q5_K), 5, 3, 2);
    run_concat_case(backend.get());
    run_argsort_case(backend.get(), 1, 3, GGML_SORT_ORDER_ASC);
    run_argsort_case(backend.get(), 255, 2, GGML_SORT_ORDER_ASC);
    run_argsort_case(backend.get(), 256, 2, GGML_SORT_ORDER_DESC);
    run_ssm_conv_case(backend.get(), 1, 3, 4, 2);
    run_ssm_conv_case(backend.get(), 4, 33, 17, 2);
    run_ssm_conv_update_case(backend.get(), 4, 33, 17, false);
    run_ssm_conv_update_case(backend.get(), 4, 33, 1, false, false, true);
    run_ssm_conv_update_case(backend.get(), 4, 33, 1, false, true, true);
    run_ssm_conv_update_case(backend.get(), 4, 33, 1, false, false, true, true);
    if (!env_enabled("GGML_HRX_DISABLE_FAST_APPROX_PROMPT")) {
        run_rms_norm_case(backend.get(), 1, 3, 2);
        run_rms_norm_case(backend.get(), 127, 3, 2);
        run_rms_norm_case(backend.get(), 128, 3, 2);
        run_rms_norm_case(backend.get(), 129, 3, 2);
        run_rms_norm_case(backend.get(), 513, 2, 2);
        run_rms_norm_mul_fusion_case(backend.get(), 64, 3, 2, 1, 1, "rms_norm_mul_fusion_broadcast");
        run_rms_norm_mul_fusion_case(backend.get(), 257, 2, 2, 2, 1, "rms_norm_mul_fusion_rows");
        run_add_rms_norm_mul_fusion_case(backend.get(), dev, "add_rms_norm_mul_fusion");
        run_rms_norm_mul_rope_fusion_case(backend.get(), "rms_norm_mul_rope_fusion");
        run_rms_norm_mul_rope_mode_not_fused_case(backend.get(), rope_test_mode::normal);
        run_rms_norm_mul_rope_mode_not_fused_case(backend.get(), rope_test_mode::neox);
        run_rope_set_rows_fusion_case(backend.get(), false, "rope_set_rows_fusion");
        run_rope_set_rows_fusion_case(backend.get(), true, "rms_norm_mul_rope_set_rows_fusion");
        run_rope_set_rows_mode_not_fused_case(backend.get(), rope_test_mode::normal);
        run_rope_set_rows_mode_not_fused_case(backend.get(), rope_test_mode::neox);
        run_rope_set_rows_negative_partial_view_case(backend.get());
        run_sum_rows_case(backend.get(), 1, 3, 2);
        run_sum_rows_case(backend.get(), 255, 3, 2);
        run_sum_rows_case(backend.get(), 256, 3, 2);
        run_sum_rows_case(backend.get(), 257, 3, 2);
        run_l2_norm_case(backend.get(), 1, 3, 2);
        run_l2_norm_case(backend.get(), 127, 3, 2);
        run_l2_norm_case(backend.get(), 128, 3, 2);
        run_l2_norm_case(backend.get(), 129, 3, 2);
        run_l2_norm_case(backend.get(), 513, 2, 2);
        run_l2_norm_pair_fusion_case(backend.get());
        run_unary_case(backend.get(), GGML_UNARY_OP_SILU, 257);
        run_unary_case(backend.get(), GGML_UNARY_OP_SIGMOID, 257);
        run_unary_case(backend.get(), GGML_UNARY_OP_SOFTPLUS, 257);
        run_swiglu_case(backend.get(), 257);
        run_silu_mul_fusion_case(backend.get(), dev, 257);
        run_add_add_fusion_case(backend.get(), dev, 1, 1);
        run_add_add_fusion_case(backend.get(), dev, 257, 3);
        run_add_softplus_mul_fusion_case(backend.get(), dev);
        run_sigmoid_mul_add_add_fusion_case(backend.get(), dev);
        run_sigmoid_mul_fusion_case(backend.get(), dev);
        run_sigmoid_mul_strided_fusion_case(backend.get(), dev);
        run_sigmoid_mul_gate_strided_fusion_case(backend.get(), dev);
        run_sigmoid_mul_strided_negative_intervening_op_case(backend.get());
        run_add8_fusion_case(backend.get(), 1, 1);
        run_add8_fusion_case(backend.get(), 257, 3);
        run_mul_sum8_fusion_case(backend.get(), 257, 3);
        run_mul_sum8_negative_intervening_op_case(backend.get());
        run_mul_sum8_negative_extra_product_consumer_case(backend.get());
        run_mul_sum8_negative_duplicate_slice_case(backend.get());
        run_mul_add_add_fusion_case(backend.get(), 1, 1);
        run_mul_add_add_fusion_case(backend.get(), 257, 3);
        run_ssm_conv_silu_fusion_case(backend.get());
        run_ssm_conv_update_case(backend.get(), 4, 33, 17, true);
        run_ssm_conv_update_case(backend.get(), 4, 33, 1, true, false, true);
        run_ssm_conv_update_case(backend.get(), 4, 33, 1, true, true, true);
        run_ssm_conv_update_gather_case(backend.get(), false, true);
        run_ssm_conv_update_gather_case(backend.get(), true, true);
        run_ssm_conv_update_gather_case(backend.get(), false, false);
        run_ssm_conv_update_gather_case(backend.get(), true, false);
        run_ssm_conv_update_negative_state_offset_case(backend.get());
        run_ssm_conv_update_negative_state_overlap_case(backend.get());
        run_mul_mat_vec_case(backend.get(), dev, GGML_TYPE_F32, 17, 3, 2, 2.0e-4f, "mul_mat_vec_f32");
        run_mul_mat_vec_case(backend.get(), dev, GGML_TYPE_F16, 257, 3, 2, 2.0e-4f, "mul_mat_vec_f16");
        run_mul_mat_vec_case(backend.get(), dev, GGML_TYPE_BF16, 257, 3, 2, 2.0e-3f, "mul_mat_vec_bf16");
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F32, 17, 3, 2, 2, 1, 4, 3,
            3.0e-4f, "mul_mat_vec_f32_batched");
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F16, 129, 2, 3, 1, 1, 3, 2,
            1.0e-3f, "mul_mat_vec_f16_batched");
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F16, 256, 256, 1, 16, 1, 16, 1,
            2.0e-2f, "mul_mat_vec_f16_batched_rows2_cols1_x8_decode");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q4_K, ggml_blck_size(GGML_TYPE_Q4_K), 3, 1,
            4.0e-4f, "mul_mat_vec_q4_k_one_block");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q4_K, 2 * ggml_blck_size(GGML_TYPE_Q4_K), 2, 3,
            5.0e-4f, "mul_mat_vec_q4_k_two_blocks");
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q4_K, ggml_blck_size(GGML_TYPE_Q4_K), 3, 1,
                8.0e-2f, "mul_mat_vec_q4_k_q8_1_one_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q4_K, 2 * ggml_blck_size(GGML_TYPE_Q4_K), 2, 3,
                1.0e-1f, "mul_mat_vec_q4_k_q8_1_multi_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 3, 1,
                8.0e-2f, "mul_mat_vec_q5_k_q8_1_one_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, 2 * ggml_blck_size(GGML_TYPE_Q5_K), 2, 3,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_multi_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 32, 32,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmq32");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 64, 64,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmq64");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 128, 128,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmql128");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 128, 129,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmql128_tail_cols");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 128, 33,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmql128_tail_cols33");
            {
                scoped_env_var disable_q5_x4_mmql128("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL128", "1");
                scoped_env_var disable_q5_x4_mmq64("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQ64", "1");
                scoped_env_var disable_q5_x4_mmq32("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQ32", "1");
                run_mul_mat_vec_case(
                    backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 32, 32,
                    1.0e-1f, "mul_mat_vec_q5_k_q8_1_mmq32");
            }
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 3, 1,
                8.0e-2f, "mul_mat_vec_q6_k_q8_1_one_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, 2 * ggml_blck_size(GGML_TYPE_Q6_K), 2, 3,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_multi_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 32, 32,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_x4_mmq32");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 128, 64,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_x4_mmql128");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 128, 65,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_x4_mmql128_tail_cols");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 128, 33,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_x4_mmql128_tail_cols33");
        }
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 3, 1,
            4.0e-4f, "mul_mat_vec_q5_k_one_block");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q5_K, 2 * ggml_blck_size(GGML_TYPE_Q5_K), 2, 3,
            5.0e-4f, "mul_mat_vec_q5_k_two_blocks");
        {
            scoped_env_var force_q5_wg64("GGML_HRX_MUL_MAT_VEC_K_WG", "64");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, 2 * ggml_blck_size(GGML_TYPE_Q5_K), 3, 1,
                5.0e-4f, "mul_mat_vec_q5_k_wg64_decode");
        }
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 3, 1,
            4.0e-4f, "mul_mat_vec_q6_k_rows2_cols1_decode");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q6_K, 2 * ggml_blck_size(GGML_TYPE_Q6_K), 2, 3,
            5.0e-4f, "mul_mat_vec_q6_k_two_blocks");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 2, 512,
            5.0e-4f, "mul_mat_vec_q6_k_rows2_cols8_prompt");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 2, 513,
            5.0e-4f, "mul_mat_vec_q6_k_rows2_cols8_tail_cols513");
        {
            scoped_env_var disable_q6_rows2("GGML_HRX_DISABLE_Q6_K_ROWS2_COLS1_DECODE", "1");
            scoped_env_var force_q6_wg64("GGML_HRX_MUL_MAT_VEC_Q6_K_WG", "64");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, 2 * ggml_blck_size(GGML_TYPE_Q6_K), 3, 1,
                5.0e-4f, "mul_mat_vec_q6_k_wg64_decode");
        }
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q8_0, ggml_blck_size(GGML_TYPE_Q8_0), 3, 1,
            4.0e-4f, "mul_mat_vec_q8_0_one_block");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q8_0, 2 * ggml_blck_size(GGML_TYPE_Q8_0), 2, 3,
            5.0e-4f, "mul_mat_vec_q8_0_two_blocks");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q8_0, 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 512,
            5.0e-4f, "mul_mat_vec_q8_0_cols8_prompt");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q8_0, 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 513,
            5.0e-4f, "mul_mat_vec_q8_0_cols8_tail_cols513");
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q8_0, 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 512,
                1.0e-1f, "mul_mat_vec_q8_0_q8_1_x4_mmq128x32_prompt");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q8_0, 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 33,
                1.0e-1f, "mul_mat_vec_q8_0_q8_1_x4_mmq128x32_tail_cols33");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q8_0, 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 511,
                1.0e-1f, "mul_mat_vec_q8_0_q8_1_x4_mmq128x32_tail_cols511");
        }
        run_q8_0_mul_mat_add_fusion_case(
            backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 1,
            5.0e-4f, "mul_mat_vec_q8_0_add_scalar");
        {
            scoped_env_var disable_x4("GGML_HRX_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ128X32_PROMPT", "1");
            scoped_env_var disable_rows4("GGML_HRX_DISABLE_Q8_0_ADD_ROWS4_COLS4_PROMPT", "1");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 512,
                5.0e-4f, "mul_mat_vec_q8_0_add_cols8_prompt");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 513,
                5.0e-4f, "mul_mat_vec_q8_0_add_cols8_tail_cols513");
        }
        {
            scoped_env_var disable_x4("GGML_HRX_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ128X32_PROMPT", "1");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q8_0), 4, 512,
                5.0e-4f, "mul_mat_vec_q8_0_add_rows4_cols4_prompt");
        }
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 512,
                1.0e-1f, "mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_prompt");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 33,
                1.0e-1f, "mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_tail_cols33");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 511,
                1.0e-1f, "mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_tail_cols511");
        }
        {
            scoped_env_var disable_wmma("GGML_HRX_DISABLE_BF16_WMMA16_PROMPT", "1");
            scoped_env_var disable_cols32("GGML_HRX_DISABLE_BF16_COLS32_PROMPT", "1");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_BF16, 256, 16, 512,
                3.0e-2f, "mul_mat_vec_bf16_cols16_prompt");
        }
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_BF16, 256, 16, 512,
            6.0e-2f, "mul_mat_vec_bf16_wmma16_prompt");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_BF16, 256, 16, 31,
            6.0e-2f, "mul_mat_vec_bf16_wmma16_prompt_tail_cols");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_BF16, 512, 2048, 1,
            3.0e-2f, "mul_mat_vec_bf16_rows4_k512_decode");
        {
            scoped_env_var force_bf16_wg128("GGML_HRX_MUL_MAT_VEC_BF16_WORKGROUP_SIZE", "128");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_BF16, 257, 3, 2,
                2.0e-3f, "mul_mat_vec_bf16_wg128");
        }
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F16, 256, 16, 512, 2, 1, 2, 1,
            2.0e-2f, "mul_mat_vec_f16_batched_cols16_prompt");
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F16, 256, 16, 31, 2, 1, 2, 1,
            2.0e-2f, "mul_mat_vec_f16_batched_cols16_prompt_tail_cols");
        {
            scoped_env_var disable_f16_cols16("GGML_HRX_DISABLE_F16_BATCHED_COLS16_PROMPT", "1");
            run_mul_mat_vec_batched_case(
                backend.get(), dev, GGML_TYPE_F16, 256, 16, 31, 2, 1, 2, 1,
                2.0e-2f, "mul_mat_vec_f16_batched_cols8_prompt");
        }
        {
            scoped_env_var disable_f16_cols16("GGML_HRX_DISABLE_F16_BATCHED_COLS16_PROMPT", "1");
            scoped_env_var disable_f16_cols8("GGML_HRX_DISABLE_F16_BATCHED_COLS8_PROMPT", "1");
            run_mul_mat_vec_batched_case(
                backend.get(), dev, GGML_TYPE_F16, 256, 16, 31, 2, 1, 2, 1,
                2.0e-2f, "mul_mat_vec_f16_batched_cols4_prompt");
        }
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F32, 256, 16, 512, 2, 1, 2, 1,
            1.0e-3f, "mul_mat_vec_f32_batched_rows2_cols8_prompt");
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F32, 256, 17, 31, 2, 1, 2, 1,
            1.0e-3f, "mul_mat_vec_f32_batched_rows2_cols8_prompt_tail");
        {
            scoped_env_var disable_f32_rows2("GGML_HRX_DISABLE_F32_BATCHED_ROWS2_COLS8_PROMPT", "1");
            run_mul_mat_vec_batched_case(
                backend.get(), dev, GGML_TYPE_F32, 256, 16, 512, 2, 1, 2, 1,
                1.0e-3f, "mul_mat_vec_f32_batched_cols16_prompt");
        }
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F32, 2048, 4, 1, 1, 1, 1, 1,
            2.0e-3f, "mul_mat_vec_f32_batched_cols1_k2048");
        run_bf16_mul_mat_swiglu_fusion_case(backend.get(), "mul_mat_vec_bf16_swiglu_fusion");
        run_bf16_mul_mat_set_rows_fusion_case(backend.get(), "mul_mat_vec_bf16_set_rows_fusion");
        run_bf16_mul_mat_vector_set_rows_fusion_case(backend.get(), "mul_mat_vec_bf16_vector_set_rows_fusion");
        run_q6_k_mul_mat_silu_mul_fusion_case(backend.get(), dev, "mul_mat_vec_q6_k_silu_mul_fusion");
        run_mul_mat_id_q4_k_case(
            backend.get(), dev, ggml_blck_size(GGML_TYPE_Q4_K), 3, 2, 3, 4, false,
            6.0e-4f, "mul_mat_id_q4_k_wg64");
        run_mul_mat_id_q4_k_case(
            backend.get(), dev, 9 * ggml_blck_size(GGML_TYPE_Q4_K), 2, 3, 2, 5, true,
            3.0e-3f, "mul_mat_id_q4_k_wg256_broadcast");
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            scoped_env_var disable_grouped("GGML_HRX_DISABLE_Q4_K_ID_GROUPED_PROMPT", "1");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 512, 4, 1, 512, 4, false,
                2.0e-3f, "mul_mat_id_q4_k_row4_prompt", true);
        }
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 512, 2, 8, 512, 4, false,
                2.0e-3f, "mul_mat_id_q4_k_grouped_row2_route8_prompt", true);
        }
        run_mul_mat_id_q4_k_case(
            backend.get(), dev, 512, 64, 8, 512, 4, false,
            2.0e-1f, "mul_mat_id_q4_k_grouped_q8_1_x4_prompt", true);
        run_mul_mat_id_q4_k_case(
            backend.get(), dev, 512, 64, 8, 33, 4, false,
            2.0e-1f, "mul_mat_id_q4_k_grouped_q8_1_x4_tail_tokens33", true);
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 512, 64, 8, 512, 4, false,
                2.0e-3f, "mul_mat_id_q4_k_grouped_q8_1_disabled_prompt", true);
        }
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 256, 8, 2, 3, 4, false,
                1.0e-1f, "mul_mat_id_q4_k_q8_1_forced");
        }
        run_mul_mat_id_q4_k_mul_fusion_case(backend.get(), "mul_mat_id_q4_k_mul_fusion");
        run_mul_mat_id_q4_k_mul_fusion_case(
            backend.get(), "mul_mat_id_q4_k_mul_rows2_x16_fusion",
            512, 2048, 8, 1, 4, 2.0e-3f, true);
        {
            scoped_env_var disable_rows2_x16("GGML_HRX_DISABLE_PACKED_Q4_K_MUL_ROWS2_X16", "1");
            run_mul_mat_id_q4_k_mul_fusion_case(
                backend.get(), "mul_mat_id_q4_k_mul_packed_fusion",
                512, 2048, 8, 1, 4, 2.0e-3f, true);
        }
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_id_q4_k_mul_fusion_case(
                backend.get(), "mul_mat_id_q4_k_mul_q8_1_fusion",
                256, 8, 2, 3, 4, 1.0e-1f, false);
        }
        run_mul_mat_id_q4_k_swiglu_fusion_case(backend.get(), "mul_mat_id_q4_k_swiglu_fusion");
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            scoped_env_var disable_grouped("GGML_HRX_DISABLE_Q4_K_SWIGLU_GROUPED_PROMPT", "1");
            run_mul_mat_id_q4_k_swiglu_fusion_case(
                backend.get(), "mul_mat_id_q4_k_swiglu_row4_prompt",
                2048, 4, 8, 512, 4, 8.0e-3f, true);
        }
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_swiglu_fusion_case(
                backend.get(), "mul_mat_id_q4_k_swiglu_grouped_row2_prompt",
                2048, 2, 8, 512, 4, 8.0e-3f, true);
        }
        run_mul_mat_id_q4_k_swiglu_fusion_case(
            backend.get(), "mul_mat_id_q4_k_swiglu_grouped_q8_1_prompt",
            2048, 16, 8, 512, 4, 2.0e-1f, true);
        run_mul_mat_id_q4_k_swiglu_fusion_case(
            backend.get(), "mul_mat_id_q4_k_swiglu_grouped_q8_1_tail_tokens33",
            2048, 16, 8, 33, 4, 2.0e-1f, true);
        run_mul_mat_id_q4_k_swiglu_sparse_small_prompt_case(backend.get());
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_swiglu_fusion_case(
                backend.get(), "mul_mat_id_q4_k_swiglu_grouped_q8_1_disabled_prompt",
                2048, 16, 8, 512, 4, 8.0e-3f, true);
        }
        run_mul_mat_id_q4_k_swiglu_fusion_case(
            backend.get(), "mul_mat_id_q4_k_swiglu_packed_fusion",
            2048, 512, 8, 1, 4, 8.0e-3f, true);
        const std::array<flash_attn_ext_config, 7> decode_feature_cases = {{
            { flash_attn_ext_phase::decode, { 32, 3, 4, 2,    5, 2 }, GGML_TYPE_F16,  GGML_TYPE_F16,  true, true, true,  true,  "flash_attn_ext_f16" },
            { flash_attn_ext_phase::decode, { 32, 1, 4, 2, 2048, 1 }, GGML_TYPE_F16,  GGML_TYPE_F16,  true, true, true,  false, "flash_attn_ext_f16_streaming_kv2048" },
            { flash_attn_ext_phase::decode, { 32, 3, 4, 2,    5, 2 }, GGML_TYPE_BF16, GGML_TYPE_BF16, true, true, false, false, "flash_attn_ext_bf16" },
            { flash_attn_ext_phase::decode, { 32, 3, 4, 2,    5, 2 }, GGML_TYPE_F32,  GGML_TYPE_F32,  true, true, false, false, "flash_attn_ext_f32" },
            { flash_attn_ext_phase::decode, { 32, 3, 4, 2,    5, 2 }, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true, true, false, false, "flash_attn_ext_q4_0" },
            { flash_attn_ext_phase::decode, { 32, 3, 4, 2,    5, 2 }, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true, true, false, false, "flash_attn_ext_q8_0" },
            { flash_attn_ext_phase::decode, { 32, 3, 4, 2,    5, 2 }, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true, true, false, false, "flash_attn_ext_q8_0_q4_0" },
        }};
        run_flash_attn_ext_cases(backend.get(), dev, decode_feature_cases);
        const std::array<flash_attn_ext_config, 7> decode_shape_cases = {{
            { flash_attn_ext_phase::decode, { 128, 1, 32, 8,   512, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true,  true,  true, false, nullptr },
            { flash_attn_ext_phase::decode, { 128, 1, 32, 8,  1024, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true,  true,  true, false, nullptr },
            { flash_attn_ext_phase::decode, { 128, 1, 32, 8,  2048, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true,  false, true, false, nullptr },
            { flash_attn_ext_phase::decode, { 128, 1, 32, 8,  4096, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true,  false, true, false, nullptr },
            { flash_attn_ext_phase::decode, { 128, 1, 32, 8,  8192, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true,  false, true, false, nullptr },
            { flash_attn_ext_phase::decode, { 128, 4, 32, 8,  8192, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true,  false, true, false, nullptr },
            { flash_attn_ext_phase::decode, { 128, 1, 32, 8, 16384, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true, false, nullptr },
        }};
        run_flash_attn_ext_cases(backend.get(), dev, decode_shape_cases);

        const std::array<flash_attn_ext_decode_kv_limit_case, 6> decode_kv_limit_cases = {{
            { GGML_TYPE_F16,  GGML_TYPE_F16,  true  },
            { GGML_TYPE_BF16, GGML_TYPE_BF16, false },
            { GGML_TYPE_F32,  GGML_TYPE_F32,  false },
            { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false },
            { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false },
            { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false },
        }};
        run_flash_attn_ext_decode_kv_limit_matrix(backend.get(), dev, decode_kv_limit_cases.data(),
                                                  decode_kv_limit_cases.size());

        const std::array<flash_attn_ext_config, 5> prefill_shape_cases = {{
            { flash_attn_ext_phase::prefill, { 128, 512, 32, 8,  512, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true, true,  true, false, nullptr },
            { flash_attn_ext_phase::prefill, { 128, 512, 32, 8, 1024, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true, true,  true, false, nullptr },
            { flash_attn_ext_phase::prefill, { 128, 512, 32, 8, 1536, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true, false, true, false, nullptr },
            { flash_attn_ext_phase::prefill, { 128, 512, 32, 8, 2048, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true, false, true, false, nullptr },
            { flash_attn_ext_phase::prefill, { 128, 512, 32, 8, 8192, 1 }, GGML_TYPE_F16, GGML_TYPE_F16, true, false, true, false, nullptr },
        }};
        run_flash_attn_ext_cases(backend.get(), dev, prefill_shape_cases);
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, false, false, "flash_attn_ext_f16_prefill_direct");
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, false, false, "flash_attn_ext_f16_prefill_direct_p2", 0.0f, 2);
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, false, false, "flash_attn_ext_f16_prefill_direct_p513", 0.0f, 513);
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, false, false, "flash_attn_ext_f16_prefill_direct_p513_kv1024", 0.0f, 513, 1024);
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, true, false, "flash_attn_ext_f16_prefill_wmma16");
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, true, true, "flash_attn_ext_f16_prefill_tile8_softcap", 4.0f);
        run_soft_max_case(backend.get(), 1, false);
        run_soft_max_case(backend.get(), 257, false);
        run_soft_max_case(backend.get(), 257, true);
        {
            scoped_env_var enable_prompt_topk("GGML_HRX_ENABLE_PROMPT_TOPK_MOE", "1");
            scoped_env_var topk_variant("GGML_HRX_TOPK_MOE_VARIANT", "baseline");
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 16, 2, 4, false, "topk_moe_baseline");
        }
        {
            scoped_env_var enable_prompt_topk("GGML_HRX_ENABLE_PROMPT_TOPK_MOE", "1");
            scoped_env_var topk_variant("GGML_HRX_TOPK_MOE_VARIANT", "shared4");
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 32, 5, 8, true, "topk_moe_shared4_norm");
        }
        {
            scoped_env_var topk_variant("GGML_HRX_TOPK_MOE_VARIANT", "shared8");
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 32, 8, 8, true, "topk_moe_shared8_norm");
        }
        {
            scoped_env_var topk_variant("GGML_HRX_TOPK_MOE_VARIANT", "wave32");
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 256, 1, 32, true, "topk_moe_wave32_norm");
        }
        {
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 32, 1, 8, true, "topk_moe_n32_top8_norm");
            run_topk_moe_case(backend.get(), 64, 1, 8, true, "topk_moe_n64_top8_norm");
            run_topk_moe_case(backend.get(), 128, 1, 8, true, "topk_moe_n128_top8_norm");
            run_topk_moe_case(backend.get(), 256, 1, 8, true, "topk_moe_n256_top8_norm");
        }
        run_rope_support_matrix(dev);
        run_rope_f32_case(backend.get(), rope_test_mode::normal, 12, 2, 3);
        run_rope_f32_case(backend.get(), rope_test_mode::normal, 128, 4, 2);
        run_rope_f32_case(backend.get(), rope_test_mode::normal, 80, 3, 2, 1.4245f, 1.4245f);
        run_rope_f32_case(backend.get(), rope_test_mode::neox, 12, 2, 3);
        run_rope_f32_case(backend.get(), rope_test_mode::neox, 128, 4, 2);
        run_rope_f32_case(backend.get(), rope_test_mode::neox, 80, 3, 2, 1.4245f, 1.4245f);
        run_rope_f32_case(backend.get(), rope_test_mode::imrope, 12, 2, 3);
        run_rope_f32_case(backend.get(), rope_test_mode::imrope, 128, 4, 2);
        run_rope_f32_case(backend.get(), rope_test_mode::imrope, 128, 3, 2, 1.4245f, 1.4245f);
        run_gated_delta_net_case(backend.get(), false);
        run_gated_delta_net_case(backend.get(), true);
        run_gated_delta_net_s128_beta_sigmoid_state_update_fusion_case(backend.get(), dev);
        run_gated_delta_net_s128_beta_sigmoid_state_gather_fusion_case(backend.get(), dev, true);
        run_gated_delta_net_s128_beta_sigmoid_state_gather_fusion_case(backend.get(), dev, false);
        run_gated_delta_net_state_update_negative_truncated_case(backend.get());
        run_gated_delta_net_state_update_negative_intervening_op_case(backend.get());
    }
    run_clamp_case(backend.get(), 1);
    run_clamp_case(backend.get(), 256);
    run_clamp_case(backend.get(), 257);

    ggml_backend_synchronize(backend.get());
    return 0;
#endif
}
