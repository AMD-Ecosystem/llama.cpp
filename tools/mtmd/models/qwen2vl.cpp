#include "models.h"

ggml_tensor * clip_graph_qwen2vl::build_inp_with_temporal_merge() {
    ggml_tensor * inp_raw = build_inp_raw();

    GGML_ASSERT(img.nx() % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny() % (patch_size * 2) == 0);

    const size_t nb1 = ggml_row_size(inp_raw->type, img.nx());
    const size_t nb2 = ggml_row_size(inp_raw->type, img.nx() * img.ny());

    if (n_batch == 1) {
        // still image input
        return ggml_add(ctx0,
            ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
            ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));
    } else if (n_batch == 2) {
        // 2 frames input (video input)
        ggml_tensor * inp_0 = ggml_view_3d(ctx0, inp_raw,
                                    img.nx(), img.ny(), 3, nb1, nb2, 0);
        ggml_tensor * inp_1 = ggml_view_3d(ctx0, inp_raw,
                                    img.nx(), img.ny(), 3, nb1, nb2,
                                    nb2 * 3); // move to the second frame
        return ggml_add(ctx0,
            ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_0, patch_size, patch_size, 0, 0, 1, 1),
            ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_1, patch_size, patch_size, 0, 0, 1, 1));
    } else {
        GGML_ASSERT(false && "n_batch > 2 is not supported");
    }
}

ggml_cgraph * clip_graph_qwen2vl::build() {
    GGML_ASSERT(model.patch_bias == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size       = 1;
    const bool use_window_attn = hparams.n_wa_pattern > 0;
    const bool batch_windows   = use_window_attn && flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED;
    const int n_wa_pattern     = hparams.n_wa_pattern;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    // Keep the input window order and batch adjacent windows of equal size.
    std::vector<std::pair<int, int>> window_runs;
    if (batch_windows) {
        const int merge_ratio = 2;
        const int window_size = hparams.attn_window_size > 0 ? hparams.attn_window_size : 112;
        const int grid_window = window_size / patch_size / merge_ratio;
        GGML_ASSERT(grid_window > 0);
        const int pw = n_patches_x / merge_ratio;
        const int ph = n_patches_y / merge_ratio;
        for (int y = 0; y < ph; y += grid_window) {
            for (int x = 0; x < pw; x += grid_window) {
                const int len = std::min(grid_window, ph - y) * std::min(grid_window, pw - x) * merge_ratio * merge_ratio;
                if (!window_runs.empty() && window_runs.back().first == len) {
                    window_runs.back().second++;
                } else {
                    window_runs.emplace_back(len, 1);
                }
            }
        }
    }

    norm_type norm_t = proj_type == PROJECTOR_TYPE_QWEN25VL
        ? NORM_TYPE_RMS // qwen 2.5 vl
        : NORM_TYPE_NORMAL; // qwen 2 vl

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp_with_temporal_merge();

    // second conv dimension
    {
        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    ggml_tensor * inpL           = inp;
    ggml_tensor * window_mask    = nullptr;
    ggml_tensor * window_idx     = nullptr;
    ggml_tensor * inv_window_idx = nullptr;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    if (use_window_attn) {
        // handle window attention inputs
        inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(inv_window_idx, "inv_window_idx");
        ggml_set_input(inv_window_idx);
        if (!batch_windows) {
            window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
            ggml_set_name(window_mask, "window_mask");
            ggml_set_input(window_mask);
        }

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
        inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool full_attn = use_window_attn ? (il + 1) % n_wa_pattern == 0 : true;

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            ggml_tensor * Qcur = ggml_add(ctx0,
                build_mm(layer.q_w, cur), layer.q_b);
            ggml_tensor * Kcur = ggml_add(ctx0,
                build_mm(layer.k_w, cur), layer.k_b);
            ggml_tensor * Vcur = ggml_add(ctx0,
                build_mm(layer.v_w, cur), layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

            if (batch_windows && !full_attn) {
                std::vector<ggml_tensor *> outputs;
                int offset = 0;
                for (const auto & run : window_runs) {
                    auto view = [&](ggml_tensor * t) {
                        return ggml_view_4d(ctx0, t, d_head, n_head, run.first, run.second,
                            t->nb[1], t->nb[2], t->nb[2] * run.first, t->nb[2] * offset);
                    };
                    outputs.push_back(build_attn(nullptr, nullptr,
                        view(Qcur), view(Kcur), view(Vcur), nullptr, kq_scale, il));
                    offset += run.first * run.second;
                }
                GGML_ASSERT(offset == n_patches);
                // A balanced tree limits copies when the image has many window runs.
                while (outputs.size() > 1) {
                    std::vector<ggml_tensor *> joined;
                    for (size_t i = 0; i < outputs.size(); i += 2) {
                        joined.push_back(i + 1 < outputs.size()
                            ? ggml_concat(ctx0, outputs[i], outputs[i + 1], 1) : outputs[i]);
                    }
                    outputs = std::move(joined);
                }
                cur = build_mm(layer.o_w, outputs.front());
                if (layer.o_b) {
                    cur = ggml_add(ctx0, cur, layer.o_b);
                }
            } else {
                cur = build_attn(layer.o_w, layer.o_b,
                    Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
            }
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // multimodal projection
    ggml_tensor * embeddings = inpL;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
    embeddings = build_ffn(embeddings,
                        model.mm_0_w, model.mm_0_b,
                        nullptr, nullptr,
                        model.mm_1_w, model.mm_1_b,
                        FFN_GELU,
                        -1);

    if (use_window_attn) {
        window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(window_idx, "window_idx");
        ggml_set_input(window_idx);

        // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        embeddings = ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4);
        embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4, batch_size);
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
