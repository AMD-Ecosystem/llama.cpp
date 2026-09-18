#pragma once

#include "common.cuh"

#include <cstdint>

// These mirror the q8_1 vec_dot_*_q8_1 helpers in vecdotq.cuh,
// but dot the dequantized weight against float activations directly (no q8_1 activation pass).
// Included by mmvdq.cu; the kernels, geometry, and dispatch live there.
// Q4_K scale/min extraction, identical to get_scale_min_k4 in convert.cu.
static __device__ __forceinline__ void dq_get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Shared by the plain and gate+up-fused kernels so the dequant math lives once.
static __device__ __forceinline__ float dq_dot_q4_K(
        const block_q4_K * b, int q_offset, int v_im,
        const float4 & by10, const float4 & by132, const float4 & by20, const float4 & by232,
        float sum10, float sum32, float sum20, float sum42) {
    const float dall = __low2float(b->dm);
    const float dmin = __high2float(b->dm);

    uint8_t dx, mx, dy, my, dz, mz, dw, mw;
    dq_get_scale_min_k4(2*v_im + 0, b->scales, dx, mx);
    dq_get_scale_min_k4(2*v_im + 1, b->scales, dy, my);
    dq_get_scale_min_k4(2*v_im + 4, b->scales, dz, mz);
    dq_get_scale_min_k4(2*v_im + 5, b->scales, dw, mw);

    const uint32_t * qs32 = (const uint32_t *) b->qs;
    const uint32_t qs0  = qs32[q_offset/4     ];
    const uint32_t qs64 = qs32[q_offset/4 + 16];

    const float sx =
        by10.x * (float) ((qs0 >>  0) & 0xF) + by10.y * (float) ((qs0 >>  8) & 0xF) +
        by10.z * (float) ((qs0 >> 16) & 0xF) + by10.w * (float) ((qs0 >> 24) & 0xF);
    const float sy =
        by132.x * (float) ((qs0 >>  4) & 0xF) + by132.y * (float) ((qs0 >> 12) & 0xF) +
        by132.z * (float) ((qs0 >> 20) & 0xF) + by132.w * (float) ((qs0 >> 28) & 0xF);
    const float sz =
        by20.x * (float) ((qs64 >>  0) & 0xF) + by20.y * (float) ((qs64 >>  8) & 0xF) +
        by20.z * (float) ((qs64 >> 16) & 0xF) + by20.w * (float) ((qs64 >> 24) & 0xF);
    const float sw =
        by232.x * (float) ((qs64 >>  4) & 0xF) + by232.y * (float) ((qs64 >> 12) & 0xF) +
        by232.z * (float) ((qs64 >> 20) & 0xF) + by232.w * (float) ((qs64 >> 28) & 0xF);

    const float smin = sum10*mx + sum32*my + sum20*mz + sum42*mw;
    return dall * (sx*dx + sy*dy + sz*dz + sw*dw) - dmin*smin;
}

// Scale unpacking (s04l/s04h/s8) and the qh bit-plane merges mirror the canonical
// vec_dot_q5_K_q8_1 in vecdotq.cuh — keep in sync if that changes.
static __device__ __forceinline__ float dq_dot_q5_K(
        const block_q5_K * b, int q_offset, int l0, int v_im,
        const float2 & by10, const float2 & by116, const float2 & by132, const float2 & by148,
        const float2 & by20, const float2 & by216, const float2 & by232, const float2 & by248,
        float smin_x, float smin_y, float smin_z, float smin_w) {
    const float dall = __low2float(b->dm);
    const float dmin = __high2float(b->dm);

    const uint16_t * sc16 = (const uint16_t *) b->scales;
    const uint32_t scale0 = sc16[v_im    ];
    const uint32_t scale4 = sc16[v_im + 2];
    const uint32_t scale8 = sc16[v_im + 4];
    const uint32_t s04l = (scale4 << 16) | scale0;
    const uint32_t s04h = (s04l & 0xC0C0C0C0u) >> 2;
    const uint32_t s04m = s04l & 0x3F3F3F3Fu;
    const uint32_t s8   = (((scale8 << 12) | scale8) & 0x0F0F0F0Fu) | s04h;

    const float sc0 = (float) ((s04m >>  0) & 0xFF);
    const float sc1 = (float) ((s04m >>  8) & 0xFF);
    const float sc2 = (float) ((s04m >> 16) & 0xFF);
    const float sc3 = (float) ((s04m >> 24) & 0xFF);
    const float sc4 = (float) ((s8   >>  0) & 0xFF);
    const float sc5 = (float) ((s8   >>  8) & 0xFF);
    const float sc6 = (float) ((s8   >> 16) & 0xFF);
    const float sc7 = (float) ((s8   >> 24) & 0xFF);

    const uint16_t * qs16 = (const uint16_t *) b->qs;
    const uint32_t qs0  = (uint32_t) qs16[q_offset/2     ] | ((uint32_t) qs16[q_offset/2 +  8] << 16);
    const uint32_t qs64 = (uint32_t) qs16[q_offset/2 + 32] | ((uint32_t) qs16[q_offset/2 + 40] << 16);

    uint32_t qs0_lo  = qs0  & 0x0F0F0F0Fu;
    uint32_t qs0_hi  = (qs0  >> 4) & 0x0F0F0F0Fu;
    uint32_t qs64_lo = qs64 & 0x0F0F0F0Fu;
    uint32_t qs64_hi = (qs64 >> 4) & 0x0F0F0F0Fu;

    const uint16_t * qh16 = (const uint16_t *) b->qh;
    const uint32_t qh = (uint32_t) qh16[l0/2] | ((uint32_t) qh16[l0/2 + 8] << 16);

    qs0_lo  += ((qh >> (2*v_im)) & 0x01010101u) << 4;
    qs0_hi  += ((qh >> (2*v_im)) & 0x02020202u) << 3;
    qs64_lo += ((qh >> (2*v_im)) & 0x10101010u);
    qs64_hi += ((qh >> (2*v_im)) & 0x20202020u) >> 1;

    const float sx =
        by10.x  * (float) ((qs0_lo  >>  0) & 0xFF) + by10.y  * (float) ((qs0_lo  >>  8) & 0xFF) +
        by116.x * (float) ((qs0_lo  >> 16) & 0xFF) + by116.y * (float) ((qs0_lo  >> 24) & 0xFF);
    const float sy =
        by132.x * (float) ((qs0_hi  >>  0) & 0xFF) + by132.y * (float) ((qs0_hi  >>  8) & 0xFF) +
        by148.x * (float) ((qs0_hi  >> 16) & 0xFF) + by148.y * (float) ((qs0_hi  >> 24) & 0xFF);
    const float sz =
        by20.x  * (float) ((qs64_lo >>  0) & 0xFF) + by20.y  * (float) ((qs64_lo >>  8) & 0xFF) +
        by216.x * (float) ((qs64_lo >> 16) & 0xFF) + by216.y * (float) ((qs64_lo >> 24) & 0xFF);
    const float sw =
        by232.x * (float) ((qs64_hi >>  0) & 0xFF) + by232.y * (float) ((qs64_hi >>  8) & 0xFF) +
        by248.x * (float) ((qs64_hi >> 16) & 0xFF) + by248.y * (float) ((qs64_hi >> 24) & 0xFF);

    const float smin = smin_x*sc2 + smin_y*sc3 + smin_z*sc6 + smin_w*sc7;
    return dall * (sx*sc0 + sy*sc1 + sz*sc4 + sw*sc5) - dmin*smin;
}

// The ql/qh bit-plane merges (q0u..q3u) and the -32 bias mirror the canonical
// vec_dot_q6_K_q8_1 in vecdotq.cuh — keep in sync if that changes.
static __device__ __forceinline__ float dq_dot_q6_K(
        const block_q6_K * b, int ql_offset, int qh_offset, int s_offset,
        const float4 & by0, const float4 & by32, const float4 & by64, const float4 & by96) {
    const float d = __half2float(b->d);

    const uint16_t * ql16 = (const uint16_t *) b->ql;
    const uint32_t ql0  = (uint32_t) ql16[ql_offset/2     ] | ((uint32_t) ql16[ql_offset/2 +  1] << 16);
    const uint32_t ql32 = (uint32_t) ql16[ql_offset/2 + 16] | ((uint32_t) ql16[ql_offset/2 + 17] << 16);

    const uint32_t ql0_lo  = ql0  & 0x0F0F0F0Fu;
    const uint32_t ql0_hi  = (ql0  >> 4) & 0x0F0F0F0Fu;
    const uint32_t ql32_lo = ql32 & 0x0F0F0F0Fu;
    const uint32_t ql32_hi = (ql32 >> 4) & 0x0F0F0F0Fu;

    const uint16_t * qh16 = (const uint16_t *) b->qh;
    const uint32_t qh = (uint32_t) qh16[qh_offset/2] | ((uint32_t) qh16[qh_offset/2 + 1] << 16);

    const uint32_t q0u = ql0_lo  | ((qh & 0x03030303u) << 4);
    const uint32_t q1u = ql32_lo | ((qh & 0x0C0C0C0Cu) << 2);
    const uint32_t q2u = ql0_hi  |  (qh & 0x30303030u);
    const uint32_t q3u = ql32_hi | ((qh & 0xC0C0C0C0u) >> 2);

    const int8_t * sc = b->scales + s_offset;
    const float sc0 = (float) sc[0];
    const float sc2 = (float) sc[2];
    const float sc4 = (float) sc[4];
    const float sc6 = (float) sc[6];

    const float sum0 =
        by0.x * (float) ((int) ((q0u >>  0) & 0xFF) - 32) + by0.y * (float) ((int) ((q0u >>  8) & 0xFF) - 32) +
        by0.z * (float) ((int) ((q0u >> 16) & 0xFF) - 32) + by0.w * (float) ((int) ((q0u >> 24) & 0xFF) - 32);
    const float sum1 =
        by32.x * (float) ((int) ((q1u >>  0) & 0xFF) - 32) + by32.y * (float) ((int) ((q1u >>  8) & 0xFF) - 32) +
        by32.z * (float) ((int) ((q1u >> 16) & 0xFF) - 32) + by32.w * (float) ((int) ((q1u >> 24) & 0xFF) - 32);
    const float sum2 =
        by64.x * (float) ((int) ((q2u >>  0) & 0xFF) - 32) + by64.y * (float) ((int) ((q2u >>  8) & 0xFF) - 32) +
        by64.z * (float) ((int) ((q2u >> 16) & 0xFF) - 32) + by64.w * (float) ((int) ((q2u >> 24) & 0xFF) - 32);
    const float sum3 =
        by96.x * (float) ((int) ((q3u >>  0) & 0xFF) - 32) + by96.y * (float) ((int) ((q3u >>  8) & 0xFF) - 32) +
        by96.z * (float) ((int) ((q3u >> 16) & 0xFF) - 32) + by96.w * (float) ((int) ((q3u >> 24) & 0xFF) - 32);

    return d * (sum0*sc0 + sum1*sc2 + sum2*sc4 + sum3*sc6);
}
