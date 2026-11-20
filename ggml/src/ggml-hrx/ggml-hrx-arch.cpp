#include "ggml-hrx-arch.h"

#include <cstring>

static constexpr bool ggml_hrx_pow2(int64_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

// The limits follow from the kernel layouts.
static constexpr bool ggml_hrx_tune_valid(const ggml_hrx_tune & t) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_hrx_mul_mat_vec_tune &     mv  = t.mul_mat_vec[i];
        const ggml_hrx_mul_mat_wmma_tune &    mw  = t.mul_mat_wmma[i];
        const ggml_hrx_mul_mat_id_vec_tune &  iv  = t.mul_mat_id_vec[i];
        const ggml_hrx_mul_mat_id_wmma_tune & iw  = t.mul_mat_id_wmma[i];
        const ggml_hrx_mul_mat_f32_tune &     f32 = t.mul_mat_f32[i];
        if (mv.max_cols < 1 || mv.max_cols > 8 || mv.rows < 1 || mv.rows > 32 || mv.read_ahead_cols < 0 ||
            mv.read_ahead_cols > 8 || mv.xs_max_k < 256 || mv.xs_max_k > 4096 || mv.xs_wg_per_cu < 1 ||
            mv.xs_depth < 1 || mv.xs_depth > 4) {
            return false;
        }
        if (mw.wide_min_tiles < 0 || mw.wide_max_work < 0 || (mw.wide_max_work > 0 && i != GGML_TYPE_Q8_0)) {
            return false;
        }
        if (iv.max_pairs < 0 || iv.kq_pairs_per_expert < 0 || !ggml_hrx_pow2(iv.kq_slots) || iv.kq_slots > 8) {
            return false;
        }
        if (iw.chunk_slots < 1 || iw.chunk_slots > 8 || iw.row_groups < 1 || iw.row_groups > 64) {
            return false;
        }
        if (f32.depth < 1 || f32.depth > 4 || !ggml_hrx_pow2(f32.k_split) || f32.k_split > 16) {
            return false;
        }
    }
    const bool wg_sizes  = t.elementwise_wg_size >= 64 && t.elementwise_wg_size <= 1024 && t.elementwise_wg_size % 64 == 0 &&
                           t.norm_wg_size >= 64 && t.norm_wg_size <= 1024 && t.norm_wg_size % 64 == 0;
    const bool concat    = t.concat_tile == 16 || t.concat_tile == 32 || t.concat_tile == 64;
    const bool stream    = t.stream_wg_per_cu >= 0 && t.stream_wg_per_cu <= 16;
    const bool rms_lanes = t.rms_norm_lanes_max == 16 || t.rms_norm_lanes_max == 32 || t.rms_norm_lanes_max == 64;
    const bool gdn       = ggml_hrx_pow2(t.gated_delta_net_lanes) && t.gated_delta_net_lanes >= 4 && t.gated_delta_net_lanes <= 64 &&
                           t.gated_delta_net_chains >= 1 && t.gated_delta_net_chains <= 8;
    const bool argsort   = t.argsort_parts >= 1 && t.argsort_parts <= 16 && t.argsort_max_row >= 1 && t.argsort_max_row <= 4096;
    // The rows kernel has at most 8 vectors of 4 values for each lane.
    const bool rms_rows  = t.rms_norm_rows_max <= 2048 && t.rms_norm_rows_max <= 32 * t.rms_norm_lanes_max;
    return wg_sizes && concat && stream && rms_rows && rms_lanes && gdn && argsort;
}

// Tuned on gfx1151.
static constexpr ggml_hrx_tune ggml_hrx_tune_rdna3_5 = [] {
    ggml_hrx_tune t = {};
    t.nodes_per_submit   = 64;
    t.nodes_first_submit = 8;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        t.mul_mat_vec[i] = {
            /* .max_cols        = */ 8,
            /* .rows            = */ 8,
            /* .read_ahead_cols = */ 2,
            /* .xs_max_k        = */ 4096,
            /* .xs_wg_per_cu    = */ 4,
            /* .xs_depth        = */ 2,
        };
        t.mul_mat_wmma[i] = {
            /* .wide_min_tiles = */ 0,
            /* .wide_max_work  = */ 0,
        };
        t.mul_mat_id_vec[i] = {
            /* .max_pairs           = */ 64,
            /* .kq_pairs_per_expert = */ 2,
            /* .kq_slots            = */ 8,
        };
        t.mul_mat_id_wmma[i] = {
            /* .chunk_slots = */ 2,
            /* .row_groups  = */ 4,
        };
        t.mul_mat_f32[i] = {
            /* .depth   = */ 3,
            /* .k_split = */ 8,
        };
    }
    t.mul_mat_wmma[GGML_TYPE_Q8_0] = {
        /* .wide_min_tiles = */ 32,
        /* .wide_max_work  = */ 256 * 2048,
    };
    t.elementwise_wg_size    = 256;
    t.norm_wg_size           = 256;
    t.concat_tile            = 32;
    t.stream_wg_per_cu       = 0;
    t.rms_norm_rows_max      = 2048;
    t.rms_norm_lanes_max     = 64;
    t.gated_delta_net_lanes  = 4;
    t.gated_delta_net_chains = 4;
    t.argsort_parts          = 16;
    t.argsort_max_row        = 4096;
    t.fa_target_workgroups   = 32;
    t.fa_many_tiles          = 64;
    t.fa_long_kv_subblocks   = 64;
    t.fa_mid_kv_subblocks    = 16;
    t.fa_small_batch         = 8;
    t.fa_split_waves         = 4;
    t.fa_min_split_subblocks = 8;
    t.fa_max_splits          = 32;
    return t;
}();

// Untuned. These families use the RDNA3.5 values.
static constexpr ggml_hrx_tune ggml_hrx_tune_rdna3 = ggml_hrx_tune_rdna3_5;
static constexpr ggml_hrx_tune ggml_hrx_tune_rdna4 = ggml_hrx_tune_rdna3_5;

// Tuned on gfx1201.
static constexpr ggml_hrx_tune ggml_hrx_tune_gfx1201 = [] {
    ggml_hrx_tune t = ggml_hrx_tune_rdna4;
    t.nodes_per_submit = 256;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        t.mul_mat_vec[i].xs_wg_per_cu = 2;
        t.mul_mat_id_vec[i].kq_pairs_per_expert = 1;
    }
    t.mul_mat_id_vec[GGML_TYPE_Q4_K].kq_slots = 4;
    // A dispatch that fills all wave slots of the device costs approximately 30 us more on gfx1201.
    // 4 workgroups of 256 lanes for each CU use half of the slots.
    t.stream_wg_per_cu = 4;
    return t;
}();

static constexpr uint32_t GGML_HRX_WG_32BIT = UINT32_MAX;
// Architected workgroup ids pack y and z into one 32-bit register.
static constexpr uint32_t GGML_HRX_WG_16BIT = 65536;

static constexpr ggml_hrx_arch ggml_hrx_archs[] = {
    // RDNA3. Untuned.
    { "gfx1100", "gfx1100", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX11, 65536, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    { "gfx1101", "gfx1101", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX11, 65536, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    { "gfx1102", "gfx1102", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX11, 65536, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    { "gfx1103", "gfx1103", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX11, 65536, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3 },
    // RDNA3.5. Only gfx1151 is tuned.
    { "gfx1150", "gfx1150", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX11, 65536, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3_5 },
    { "gfx1151", "gfx1151", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX11, 65536, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3_5 },
    { "gfx1152", "gfx1152", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX11, 65536, GGML_HRX_WG_32BIT, &ggml_hrx_tune_rdna3_5 },
    // RDNA4. Only gfx1201 is tuned.
    { "gfx1200", "gfx1200", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX12, 65536, GGML_HRX_WG_16BIT, &ggml_hrx_tune_rdna4 },
    { "gfx1201", "gfx1201", true, true, GGML_HRX_MATRIX_OP_WMMA_GFX12, 65536, GGML_HRX_WG_16BIT, &ggml_hrx_tune_gfx1201 },
};

static constexpr bool ggml_hrx_archs_valid() {
    for (const ggml_hrx_arch & arch : ggml_hrx_archs) {
        if (!ggml_hrx_tune_valid(*arch.tune)) {
            return false;
        }
    }
    return true;
}
static_assert(ggml_hrx_archs_valid(), "ggml_hrx_tune value out of range");

const ggml_hrx_arch * ggml_hrx_arch_find(const char * name) {
    for (const ggml_hrx_arch & arch : ggml_hrx_archs) {
        if (std::strcmp(arch.name, name) == 0) {
            return &arch;
        }
    }
    return nullptr;
}
