#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// dst = SiLU(the convolution of each channel's state, read in place through the row index of state_rows (a GET_ROWS
// of one cache row), and x with conv_w), and state_out = the window's last d_conv - 1 values, for one token
// norm[k], if not null: also RMS_NORM + SCALE (see ggml_cuda_op_rms_norm_scale_fused) of rows of 128 channels of dst,
// read through a view, into norm[k]'s SCALE node
void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_tensor * state_rows, const ggml_tensor * x,
                                const ggml_tensor * conv_w, ggml_tensor * state_out, ggml_tensor * dst,
                                const ggml_tensor * const * norm = nullptr, const ggml_tensor * const * norm_scale = nullptr);
