#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define HRX_FA_PREFILL_DIRECT_KERNEL_NAME hrx_flash_attn_ext_f32_f16_prefill_direct_d128
#define HRX_FA_PREFILL_DIRECT_HEAD_DIM 128
#include "flash_attn_ext_f32_f16_prefill_direct_gfx11.inc"
#else
extern "C" __global__ void hrx_flash_attn_ext_f32_f16_prefill_direct_d128() {}
#endif
