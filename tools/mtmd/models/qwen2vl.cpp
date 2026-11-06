#include "models.h"

ggml_cgraph * clip_graph_qwen2vl::build() {
    GGML_ASSERT(model.patch_bias == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size       = 1;
    const bool use_window_attn = hparams.n_wa_pattern > 0;
    const int n_wa_pattern     = hparams.n_wa_pattern;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    norm_type norm_t = proj_type == PROJECTOR_TYPE_QWEN25VL
        ? NORM_TYPE_RMS // qwen 2.5 vl
        : NORM_TYPE_NORMAL; // qwen 2 vl

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp_temporal_merge();

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

    // batched windows: each window is padded to wa_size rows and becomes one batch entry of the attention
    const bool wa_batched = use_window_attn && hparams.wa_batched;
    ggml_tensor * wa_pad_idx   = nullptr;
    ggml_tensor * wa_unpad_idx = nullptr;
    ggml_tensor * wa_mask      = nullptr;
    int wa_size  = 0;
    int wa_count = 0;
    int wa_full  = 0; // wa_split: number of full windows, placed first

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
        if (wa_batched) {
            const int merge       = 2;
            const int grid_window = (hparams.attn_window_size > 0 ? hparams.attn_window_size : 112) / patch_size / merge;
            const int ph          = n_patches_y / merge;
            const int pw          = n_patches_x / merge;
            wa_size  = grid_window * grid_window * merge * merge;
            wa_count = ((ph + grid_window - 1) / grid_window) * ((pw + grid_window - 1) / grid_window);
            if (hparams.wa_split) {
                wa_full   = (ph / grid_window) * (pw / grid_window);
                wa_count -= wa_full;
            }
            const int n_pad_rows = n_pos - wa_full * wa_size;

            if (wa_count > 0) {
                wa_pad_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, wa_size * wa_count);
                ggml_set_name(wa_pad_idx, "wa_pad_idx");
                ggml_set_input(wa_pad_idx);
                wa_unpad_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pad_rows);
                ggml_set_name(wa_unpad_idx, "wa_unpad_idx");
                ggml_set_input(wa_unpad_idx);
            }
            // only partial windows have padded keys to mask out
            if (wa_size * wa_count != n_pad_rows) {
                const ggml_type type = flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED ? GGML_TYPE_F16 : GGML_TYPE_F32;
                wa_mask = ggml_new_tensor_4d(ctx0, type, wa_size, wa_size, 1, wa_count);
                ggml_set_name(wa_mask, "wa_mask");
                ggml_set_input(wa_mask);
            }
        } else {
            // mask for window attention
            window_mask = build_inp_attn_mask(n_pos, n_pos);
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
            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;
            if (layer.fused_qkv_w) {
                ggml_tensor * qkv = ggml_add(ctx0, build_mm(layer.fused_qkv_w, cur), layer.fused_qkv_b);
                cb(qkv, "qkv", il);
                const size_t es = ggml_element_size(qkv);
                Qcur = ggml_view_3d(ctx0, qkv, d_head, n_head, n_patches, d_head*es, qkv->nb[1], 0);
                Kcur = ggml_view_3d(ctx0, qkv, d_head, n_head, n_patches, d_head*es, qkv->nb[1], n_embd*es);
                Vcur = ggml_view_3d(ctx0, qkv, d_head, n_head, n_patches, d_head*es, qkv->nb[1], 2*n_embd*es);
            } else {
                Qcur = ggml_add(ctx0,
                    build_mm(layer.q_w, cur), layer.q_b);
                Kcur = ggml_add(ctx0,
                    build_mm(layer.k_w, cur), layer.k_b);
                Vcur = ggml_add(ctx0,
                    build_mm(layer.v_w, cur), layer.v_b);

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);
            }

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

            // the padded windows are gathered as f32, cast them after the gather
            ggml_tensor * Kcur_f32 = Kcur;
            ggml_tensor * Vcur_f32 = Vcur;
            if (hparams.kv_cast_early && flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
                Kcur = ggml_cast(ctx0, Kcur, GGML_TYPE_F16);
                Vcur = ggml_cast(ctx0, Vcur, GGML_TYPE_F16);
            }

            if (wa_batched && !full_attn) {
                auto pad = [&](ggml_tensor * t) {
                    const bool to_f16 = t->type == GGML_TYPE_F16;
                    t = t == Kcur ? Kcur_f32 : (t == Vcur ? Vcur_f32 : t);
                    t = ggml_view_2d(ctx0, t, d_head * n_head, n_pos, t->nb[2], 0);
                    t = ggml_get_rows(ctx0, t, wa_pad_idx);
                    t = ggml_reshape_4d(ctx0, t, d_head, n_head, wa_size, wa_count);
                    return to_f16 ? ggml_cast(ctx0, t, GGML_TYPE_F16) : t;
                };
                // full windows are contiguous rows, use them in place
                auto full = [&](ggml_tensor * t) {
                    return ggml_view_4d(ctx0, t, d_head, n_head, wa_size, wa_full, t->nb[1], t->nb[2], wa_size*t->nb[2], 0);
                };
                ggml_tensor * cur_full = nullptr;
                if (wa_full > 0) {
                    cur_full = build_attn(nullptr, nullptr,
                        full(Qcur), full(Kcur), full(Vcur), nullptr, kq_scale, il);
                }
                cur = nullptr;
                if (wa_count > 0) {
                    cur = build_attn(nullptr, nullptr,
                        pad(Qcur), pad(Kcur), pad(Vcur), wa_mask, kq_scale, il);
                    cur = ggml_get_rows(ctx0, cur, wa_unpad_idx);
                }
                cur = cur_full && cur ? ggml_concat(ctx0, cur_full, cur, 1) : (cur ? cur : cur_full);
                cur = ggml_add(ctx0, build_mm(layer.o_w, cur), layer.o_b);
            } else {
                ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

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
        if (layer.fused_gate_up_w && layer.ff_down_b && hparams.ffn_op == FFN_SILU && hparams.swiglu_clamp <= 0.0f) {
            cur = ggml_add(ctx0, build_mm(layer.fused_gate_up_w, cur), layer.fused_gate_up_b);
            cb(cur, "ffn_gate_up", il);
            cur = ggml_swiglu(ctx0, cur);
            cb(cur, "ffn_swiglu", il);
            cur = ggml_add(ctx0, build_mm(layer.ff_down_w, cur), layer.ff_down_b);
        } else {
            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                hparams.ffn_op, il);
        }

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
