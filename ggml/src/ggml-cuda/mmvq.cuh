#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// hc_pre (a gated DSV4_HC_PRE) mixing the streams with the sigmoid of src0 x src1, one token, in one kernel; also
// writes MMVQ's q8_1 copy of hc_pre when write_q8_1 (see ggml_cuda_q8_1_cache). src1_copy, if not nullptr, holds the
// values of src1 where hc_pre cannot overlap them. With inject_dst, the same kernel also writes there the product inject
// (F32 weights, one token) as mul_mat_vec_f computes it with 256 threads per row.
void ggml_cuda_mul_mat_vec_q_hc_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                    const float * src1_copy, ggml_tensor * hc_pre, bool write_q8_1,
                                    const ggml_tensor * inject = nullptr, float * inject_dst = nullptr);

// unary_node = silu(scale_node(src0 x src1)) for one token in one kernel, also written to dst_copy if not nullptr
void ggml_cuda_mul_mat_vec_q_scale_silu(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * scale_node, ggml_tensor * unary_node, float * dst_copy);

// whether ggml_cuda_mul_mat_vec_q_moe_glu_shared supports routed experts of type and a shared expert of type_s
bool ggml_cuda_mmvq_moe_glu_shared_supported(ggml_type type, ggml_type type_s);

// routed (the GLU of the routed experts' gate and up MUL_MAT_IDs) and shared (that of the shared expert, weights of
// another type) for one token in one kernel; also writes MMVQ's q8_1 copies of routed and shared when asked
// With scratch, the kernel also computes there the shared expert's gate shared_gate (F32 weights, as mul_mat_vec_f
// sums a row with 256 threads) and copies the routing weights and ids, see ggml_cuda_mul_mat_vec_q_moe_down_shared.
void ggml_cuda_mul_mat_vec_q_moe_glu_shared(ggml_backend_cuda_context & ctx, const ggml_tensor * routed,
                                            const ggml_tensor * shared, bool write_q8_1, bool write_q8_1_s,
                                            const ggml_tensor * shared_gate = nullptr, const ggml_tensor * weights = nullptr,
                                            float * scratch = nullptr);

// whether ggml_cuda_mul_mat_vec_q_moe_down_shared supports routed down weights of type and shared ones of type_s
// the MMVQ product mm of one token followed by the gated output norm of a gated delta net: dst = SIGMOID(mm) * (RMS_NORM
// (rms, rows of 128) * norm_w), with MMVQ's q8_1 copy of dst in q8_1, in one kernel
bool ggml_cuda_mmvq_norm_gate_supported(ggml_type type);
void ggml_cuda_mul_mat_vec_q_norm_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * rms,
                                       const ggml_tensor * norm_w, ggml_tensor * dst, const ggml_tensor * q8_1_tensor);

bool ggml_cuda_mmvq_moe_down_shared_supported(ggml_type type, ggml_type type_s);

// dst = the routing-weighted sum of the routed experts' down products (down, a MUL_MAT_ID) + the shared expert's down
// product times the sigmoid of its gate, for one token in one kernel: y_q8_1 and y_s_q8_1 are MMVQ's q8_1 copies of the
// inputs, scratch the gate and routing ggml_cuda_mul_mat_vec_q_moe_glu_shared wrote
void ggml_cuda_mul_mat_vec_q_moe_down_shared(ggml_backend_cuda_context & ctx, const ggml_tensor * down,
                                             const ggml_tensor * shared_down, ggml_tensor * dst, const float * scratch,
                                             const char * y_q8_1, const char * y_s_q8_1);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
