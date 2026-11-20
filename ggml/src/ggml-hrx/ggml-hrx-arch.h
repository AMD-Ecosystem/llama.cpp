#pragma once

#include "ggml.h"

#include <cstdint>

enum ggml_hrx_matrix_op {
    GGML_HRX_MATRIX_OP_NONE,
    GGML_HRX_MATRIX_OP_WMMA_GFX11,
    GGML_HRX_MATRIX_OP_WMMA_GFX12, // The operand layout is different from gfx11. The WMMA motifs have gfx12 providers.
};

// ggml_mul_mat_vec and ggml_mul_mat_vec_xs.
struct ggml_hrx_mul_mat_vec_tune {
    int64_t max_cols;        // At most 8. More input columns use the tiled kernels.
    int64_t rows;            // At most 32. Weight rows of a workgroup, one wave each.
    int64_t read_ahead_cols; // The vec kernel reads ahead only for this many input columns or fewer.
    int64_t xs_max_k;        // At most 4096. The vec_xs kernel keeps 16 superblocks of activations in registers.
    int64_t xs_wg_per_cu;
    int64_t xs_depth;        // At most 4. Pipeline depth of the vec_xs row passes.
};

// The Q8_0 matrix-op kernels.
struct ggml_hrx_mul_mat_wmma_tune {
    // The 128x128 tile kernel runs in this range of tiles and tiles x K. Zero disables it.
    int64_t wide_min_tiles;
    int64_t wide_max_work;
};

// ggml_mul_mat_id_vec and ggml_mul_mat_id_vec_kq.
struct ggml_hrx_mul_mat_id_vec_tune {
    int64_t max_pairs;
    int64_t kq_pairs_per_expert;
    int64_t kq_slots; // 1, 2, 4 or 8. Blocks of one row that a step reads.
};

// ggml_mul_mat_id_wmma.
struct ggml_hrx_mul_mat_id_wmma_tune {
    int64_t chunk_slots; // At most 8.
    int64_t row_groups;  // At most 64. Workgroups that share the row tiles of one expert.
};

// ggml_mul_mat_f32.
struct ggml_hrx_mul_mat_f32_tune {
    int64_t depth;   // At most 4. Pipeline depth of the K loop.
    int64_t k_split; // 1, 2, 4, 8 or 16. Waves that split K.
};

// Op dispatch values for one architecture.
struct ggml_hrx_tune {
    int nodes_per_submit;   // Nodes between stream flushes.
    int nodes_first_submit; // The interval starts here and doubles up to nodes_per_submit.

    // Index: the weight tensor type (src0).
    ggml_hrx_mul_mat_vec_tune     mul_mat_vec[GGML_TYPE_COUNT];
    ggml_hrx_mul_mat_wmma_tune    mul_mat_wmma[GGML_TYPE_COUNT];
    ggml_hrx_mul_mat_id_vec_tune  mul_mat_id_vec[GGML_TYPE_COUNT];
    ggml_hrx_mul_mat_id_wmma_tune mul_mat_id_wmma[GGML_TYPE_COUNT];
    ggml_hrx_mul_mat_f32_tune     mul_mat_f32[GGML_TYPE_COUNT];

    int64_t elementwise_wg_size; // Multiple of 64, at most 1024.
    int64_t norm_wg_size;        // Multiple of 64, at most 1024. Also for soft_max and sum_rows.
    int64_t concat_tile;         // 16, 32 or 64. Square tile of the transposed concat.
    // At most 16. Workgroup limit for each CU of the grid-stride copy kernels. 0 disables these kernels.
    int64_t stream_wg_per_cu;

    int64_t rms_norm_rows_max;  // At most 2048.
    int64_t rms_norm_lanes_max; // 16, 32 or 64. Lanes for each row of the rows kernel.

    int64_t gated_delta_net_lanes;  // 4, 8, 16, 32 or 64. Lanes that share one state column.
    int64_t gated_delta_net_chains; // At most 8. Partial dot chains of each lane.

    int64_t argsort_parts;   // At most 16. Waves that split the compares of one block of 64 elements.
    int64_t argsort_max_row; // At most 4096, the row stage of the kernel. The cost is quadratic in the row length.

    int64_t fa_target_workgroups;
    int64_t fa_many_tiles;
    int64_t fa_long_kv_subblocks;
    int64_t fa_mid_kv_subblocks;
    int64_t fa_small_batch;
    int64_t fa_split_waves;
    int64_t fa_min_split_subblocks;
    int64_t fa_max_splits;
};

struct ggml_hrx_arch {
    const char *          name;        // Architecture that the device reports.
    const char *          loom_target; // Loom AMDGPU target of the kernels.
    bool                  wave32;
    bool                  wave64;
    ggml_hrx_matrix_op    matrix_op;
    uint32_t              lds_bytes;         // Local memory limit of one workgroup.
    uint32_t              max_workgroups_yz; // Workgroup count limit in y and z.
    const ggml_hrx_tune * tune;
};

// Returns nullptr if the backend does not support the architecture.
const ggml_hrx_arch * ggml_hrx_arch_find(const char * name);
