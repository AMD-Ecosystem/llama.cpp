#include "../../../ggml-hrx/kernels/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp"

extern "C" __global__ void hrx2_mul_mat_vec_q6_k_q8_1_x4_mmql64x128_wg256_u32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        uint32_t k,
        uint32_t rows,
        uint32_t cols) {
    hrx_mul_mat_vec_q6_k_q8_1_x4_mmql_wg256_impl<64, 128>(
        src0,
        src1,
        dst,
        static_cast<long long>(k),
        static_cast<long long>(rows),
        static_cast<long long>(cols));
}
