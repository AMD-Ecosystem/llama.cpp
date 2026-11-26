#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream);

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst = scale_node(the sum of rows rows[0..3] of get_rows) for one token, after writing the CONCAT of k_new and fill
// into the cache row of set_rows (F16), as FILL, CONCAT, SET_ROWS, GET_ROWS, the ADD chain and SCALE compute it
void ggml_cuda_op_qsa_kpool_step(ggml_backend_cuda_context & ctx, const ggml_tensor * k_new, const ggml_tensor * fill,
                                 const ggml_tensor * set_rows, const ggml_tensor * get_rows, const int * rows,
                                 const ggml_tensor * scale_node, ggml_tensor * dst);
