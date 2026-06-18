#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import sys


KERNELS = [
    {
        "name": "hrx_rms_norm_f32",
        "source": "rms_norm.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "hrx_rms_norm_mul_f32",
        "source": "rms_norm_mul_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 144,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "hrx_rms_norm_mul_wg128_f32",
        "source": "rms_norm_mul_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 144,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_rms_norm_mul_rope_f32",
        "source": "rms_norm_mul_rope_f32.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 184,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "hrx_rms_norm_mul_rope_set_rows_f32_f16",
        "source": "rms_norm_mul_rope_f32.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 248,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "hrx_add_f32",
        "source": "add_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_f32",
        "source": "mul_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_add_f32_broadcast",
        "source": "add_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_add_add_f32_broadcast",
        "source": "add_add_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 144,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_add_rms_norm_mul_f32_broadcast",
        "source": "add_rms_norm_mul_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 200,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "hrx_mul_f32_broadcast",
        "source": "mul_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_div_f32_broadcast",
        "source": "div_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_add8_f32",
        "source": "add8_f32.hip.cpp",
        "format": None,
        "binding_count": 9,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_sum8_f32",
        "source": "mul_sum8_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 56,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_add_add_f32_broadcast",
        "source": "mul_add_add_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 176,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_add_softplus_mul_f32_broadcast",
        "source": "add_softplus_mul_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 144,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_sigmoid_mul_add_add_f32_broadcast",
        "source": "mul_add_add_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 176,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_scale_f32",
        "source": "scale_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "parameter_count": 5,
        "constants_size": 16,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_f32",
        "source": "set_rows_f32_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_f16",
        "source": "set_rows_f32_f16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_q8_0",
        "source": "set_rows_f32_q8_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_q4_0",
        "source": "set_rows_f32_q4_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_silu_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_sigmoid_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_sigmoid_mul_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_sigmoid_mul_f32_strided",
        "source": "sigmoid_mul_f32_strided.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_softplus_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_swiglu_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_sum_rows_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_l2_norm_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_l2_norm_wg128_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_l2_norm_pair_wg128_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 176,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_soft_max_f32",
        "source": "soft_max_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_soft_max_f32_mask",
        "source": "soft_max_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_argsort_f32_i32",
        "source": "argsort_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_rope_f32",
        "source": "rope_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 120,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_rope_set_rows_f32_f16",
        "source": "rope_set_rows_f32_f16.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 192,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_ssm_conv_f32",
        "source": "ssm_conv_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 88,
        "workgroup_size": (32, 16, 1),
    },
    {
        "name": "hrx_ssm_conv_update_f32",
        "source": "ssm_conv_update_f32.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_ssm_conv_update_gather_f32",
        "source": "ssm_conv_update_f32.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 136,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_ssm_conv_update_gather_decode_f32",
        "source": "ssm_conv_update_f32.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 136,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_ssm_conv_update_gather_decode_direct_f32",
        "source": "ssm_conv_update_f32.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 136,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 8,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_s128_cluster8_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 8,
        "constants_size": 208,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_s128_cluster8_nokda_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 8,
        "constants_size": 208,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_s128_cluster8_nokda_nomod_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 8,
        "constants_size": 176,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 8,
        "constants_size": 16,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_beta_sigmoid_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 8,
        "constants_size": 16,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_beta_sigmoid_gather_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 9,
        "constants_size": 40,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_beta_sigmoid_direct_gather_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 9,
        "constants_size": 40,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_clamp_f32",
        "source": "clamp_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 16,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_get_rows_f32",
        "source": "get_rows_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_get_rows_f32_nr1",
        "source": "get_rows_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_scale_get_rows_f32_nr1_x4",
        "source": "get_rows_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_get_rows_f32_nr1_x4",
        "source": "get_rows_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_get_rows_q5_k_f32",
        "source": "get_rows_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_wg128_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_wg64_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols1_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols2_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols3_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows2_cols1_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows2_cols1_wg32_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows2_cols1_x8_wg32_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows4_k512_cols1_lds_wg256_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows4_k2048_cols1_lds_wg256_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols4_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols5_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols6_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols7_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols8_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols32_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows2_cols16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_wmma16x16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_wg128_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_wg64_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_wmma16x16_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols1_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols2_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols3_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_rows2_cols1_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_lds_wg256_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_rows8_k2048_cols1_lds_wg256_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols4_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols5_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols6_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols7_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols8_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols16_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_rows2_cols8_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_set_rows_f16",
        "source": "mul_mat_vec_bf16_set_rows.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 40,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows4_k2048_cols1_set_rows_f16",
        "source": "mul_mat_vec_bf16_set_rows.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 40,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_f32",
        "source": "mul_mat_vec_f16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols1_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_rows2_cols1_x8_wg32_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols4_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols8_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols16_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_rows2_cols16_wg32_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_cols3_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_cols4_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_cols5_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_cols6_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_cols7_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols1_ne2_1_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols8_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols16_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_rows2_cols8_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_row4_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_rows2_x16_wg32_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_row8_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_clear_u32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 1,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_compact_moe_routes_i32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 48,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 96,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32",
        "source": "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 96,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x16_wg64_f32",
        "source": "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 96,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_q8_1_f32",
        "source": "mul_mat_id_q4_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 96,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_wg64_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_packed_wg64_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_rows2_x16_wg32_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_row2_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_row4_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route4_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32",
        "source": "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_bn16_wg64_f32",
        "source": "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_packed_wg32_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_q8_1_f32",
        "source": "mul_mat_id_q4_k_mul_q8_1.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q4_k_f32",
        "source": "mul_mat_vec_q4_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_quantize_q8_1_f32",
        "source": "quantize_q8_1.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 56,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_quantize_q8_1_x4_f32",
        "source": "quantize_q8_1.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 56,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q4_k_q8_1_f32",
        "source": "mul_mat_vec_q4_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_wg64_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_wg32_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_rows2_cols2_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_rows2_cols3_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_rows2_cols4_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_rows2_cols5_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_rows2_cols6_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_rows2_cols7_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_rows2_cols8_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32",
        "source": "mul_mat_vec_q5_k_q8_1_wave64.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32",
        "source": "mul_mat_vec_q5_k_q8_1_wave64.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_wg64_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_silu_mul_rows2_cols1_wg32_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols2_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols3_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols4_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols5_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols6_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols7_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_rows2_cols8_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_f32",
        "source": "mul_mat_vec_q6_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32",
        "source": "mul_mat_vec_q6_k_q8_1_wave64.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_x4_mmql64x128_wg256_f32",
        "source": "mul_mat_vec_q6_k_q8_1_wave64.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_x4_mmq64x128_wg256_f32",
        "source": "mul_mat_vec_q6_k_q8_1_x4_wave64_direct.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x128_wg256_f32",
        "source": "mul_mat_vec_q6_k_q8_1_x4_mmql128.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q6_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_cols8_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_add_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_add_cols8_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_add_rows4_cols4_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_decode",
        "source": "flash_attn_ext_f32_f16_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_decode_gqa8_split",
        "source": "flash_attn_ext_f32_f16_decode_split.hip.cpp",
        "format": None,
        "binding_count": 7,
        "constants_size": 200,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_decode_gqa4_split",
        "source": "flash_attn_ext_f32_f16_decode_split.hip.cpp",
        "format": None,
        "binding_count": 7,
        "constants_size": 200,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_decode_reduce",
        "source": "flash_attn_ext_f32_f16_decode_reduce.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_prefill_tile8",
        "source": "flash_attn_ext_f32_f16_prefill_tile8.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_prefill_wmma16",
        "source": "flash_attn_ext_f32_f16_prefill_wmma16.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_prefill_direct",
        "source": "flash_attn_ext_f32_f16_prefill_direct.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_flash_attn_ext_f32_bf16_decode",
        "source": "flash_attn_ext_f32_bf16_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f32_decode",
        "source": "flash_attn_ext_f32_f32_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_q8_0_decode",
        "source": "flash_attn_ext_f32_q8_0_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_q4_0_decode",
        "source": "flash_attn_ext_f32_q4_0_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_q8_0_q4_0_decode",
        "source": "flash_attn_ext_f32_q8_0_q4_0_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_topk_moe_f32",
        "source": "topk_moe_f32_shared.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_topk_moe_f32_shared4",
        "source": "topk_moe_f32_shared.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (64, 4, 1),
    },
    {
        "name": "hrx_topk_moe_f32_shared8",
        "source": "topk_moe_f32_shared.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (64, 8, 1),
    },
    {
        "name": "hrx_topk_moe_f32_wave32",
        "source": "topk_moe_f32_wave32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (32, 4, 1),
    },
    {
        "name": "hrx_topk_moe_f32_wave32_n32_top8_norm",
        "source": "topk_moe_f32_wave32_top8_norm.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (32, 4, 1),
    },
    {
        "name": "hrx_topk_moe_f32_wave32_n64_top8_norm",
        "source": "topk_moe_f32_wave32_top8_norm.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (32, 4, 1),
    },
    {
        "name": "hrx_topk_moe_f32_wave32_n128_top8_norm",
        "source": "topk_moe_f32_wave32_top8_norm.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (32, 4, 1),
    },
    {
        "name": "hrx_topk_moe_f32_wave32_n256_top8_norm",
        "source": "topk_moe_f32_wave32_top8_norm.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (32, 4, 1),
    },
    {
        "name": "hrx_concat_f32",
        "source": "concat_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 72,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_copy_strided_f32",
        "source": "copy_strided_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 64,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_copy_f32_f16",
        "source": "copy_f32_f16.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
]

for _cols in range(2, 9):
    KERNELS.append({
        "name": f"hrx_mul_mat_vec_q5_k_rows2_cols{_cols}_wg64_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    })
    KERNELS.append({
        "name": f"hrx_mul_mat_vec_q6_k_rows2_cols{_cols}_wg64_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    })
    KERNELS.append({
        "name": f"hrx_mul_mat_vec_q6_k_rows2_cols{_cols}_wg32_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    })

DOT8_ARCH_PREFIXES = ("gfx11",)
DOT8_SOURCES = {
    "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
    "mul_mat_vec_q5_k_q8_1.hip.cpp",
    "mul_mat_vec_q5_k_q8_1_wave64.hip.cpp",
    "mul_mat_vec_q6_k_q8_1.hip.cpp",
    "mul_mat_vec_q6_k_q8_1_wave64.hip.cpp",
    "mul_mat_vec_q8_0.hip.cpp",
}


def hsaco_name_for_source(source):
    if source.endswith(".hip.cpp"):
        return source[:-len(".hip.cpp")] + ".hsaco"
    return pathlib.Path(source).name + ".hsaco"


def symbol_part(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def unique_sources():
    return sorted({kernel["source"] for kernel in KERNELS})


def unique_sources_for(kernels):
    return sorted({kernel["source"] for kernel in kernels})


def kernel_supports_arch(kernel, arch):
    prefixes = DOT8_ARCH_PREFIXES if kernel["source"] in DOT8_SOURCES else kernel.get("arch_prefixes")
    return not prefixes or any(arch.startswith(prefix) for prefix in prefixes)


def kernel_source_id(source):
    return symbol_part(source.removesuffix(".hip.cpp"))


def family_for_kernel_name(name):
    name = name.removeprefix("hrx_")
    for suffix in ("_f32", "_i32"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    parts = name.split("_")
    if len(parts) > 4 and parts[-2] in ("cols", "rows"):
        return "_".join(parts[:-2])
    return name


def op_for_family(family):
    if family.startswith("mul_mat"):
        return "MUL_MAT_ID" if family.startswith("mul_mat_id") else "MUL_MAT"
    if family.startswith("rms_norm"):
        return "RMS_NORM"
    if family.startswith("soft_max"):
        return "SOFT_MAX"
    if family.startswith("set_rows"):
        return "SET_ROWS"
    if family.startswith("get_rows"):
        return "GET_ROWS"
    if family.startswith("quantize"):
        return "QUANTIZE"
    if family.startswith("topk"):
        return "TOP_K"
    if family.startswith("flash_attn"):
        return "FLASH_ATTN_EXT"
    if family.startswith("rope"):
        return "ROPE"
    if family.startswith("add"):
        return "ADD"
    if family.startswith("mul"):
        return "MUL"
    if family.startswith("div"):
        return "DIV"
    if family.startswith("scale"):
        return "SCALE"
    if family.startswith("clamp"):
        return "CLAMP"
    if family.startswith("argsort"):
        return "ARGSORT"
    if family.startswith("concat"):
        return "CONCAT"
    if family.startswith("copy"):
        return "CPY"
    return "CUSTOM"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def export_split_catalog(catalog_dir):
    sources = {}
    artifacts = {}
    families = {}
    routes = []
    route_index = []

    for source in unique_sources():
        source_id = kernel_source_id(source)
        sources[source_id] = {"path": source}
        artifacts[source_id + "_hsaco"] = {
            "source_id": source_id,
            "path": hsaco_name_for_source(source),
            "format": "amdgpu-hsaco",
            "storage": "embedded",
        }

    seen_route_ids = set()
    for kernel in KERNELS:
        route_id = kernel["name"]
        if route_id in seen_route_ids:
            raise SystemExit(f"duplicate generated route id: {route_id}")
        seen_route_ids.add(route_id)
        source = kernel["source"]
        source_id = kernel_source_id(source)
        family = family_for_kernel_name(kernel["name"])
        families.setdefault(family, {"family": family, "op": op_for_family(family)})
        wx, wy, wz = kernel["workgroup_size"]
        route = {
            "id": route_id,
            "family": family,
            "op": op_for_family(family),
            "target_key": "gfx1100",
            "source_id": source_id,
            "artifact_id": source_id + "_hsaco",
            "export_name": kernel["name"],
            "loader_format": kernel.get("format") or "amdgpu-hsaco",
            "priority": 100,
            "abi": {
                "binding_count": kernel["binding_count"],
                "parameter_count": kernel.get("parameter_count", kernel["binding_count"] + 1),
                "constant_byte_length": kernel["constants_size"],
            },
            "dispatch": {
                "workgroup_size": [wx, wy, wz],
            },
            "shape_domain": {
                "name": "legacy_cxx_default",
            },
            "shape_guards": {},
            "supports": {
                "legacy_export": kernel["name"],
            },
            "evidence_summary": {
                "status": "accepted_gfx1100_legacy",
                "selection": "ported_from_pre_catalog_hrx_v1_hardcoded_route",
            },
        }
        if "arch_prefixes" in kernel:
            route["target_prefixes"] = list(kernel["arch_prefixes"])
        elif source in DOT8_SOURCES:
            route["target_prefixes"] = list(DOT8_ARCH_PREFIXES)
        routes.append(route)
        route_index.append(route_id)

    write_json(catalog_dir / "metadata.json", {
        "schema": "ggml-hrx-catalog-v1",
        "catalog_id": "hrx-v1-legacy-gfx1100",
        "generated_at": "source",
        "targets": [
            {"target_key": "gfx1100", "target_variant": None, "status": "accepted_legacy"},
            {"target_key": "gfx1151", "target_variant": None, "status": "tuning_seed"},
        ],
    })
    write_json(catalog_dir / "sources.json", sources)
    write_json(catalog_dir / "artifacts.json", artifacts)
    write_json(catalog_dir / "families.json", sorted(families.values(), key=lambda item: item["family"]))
    write_json(catalog_dir / "routes" / "legacy.json", routes)
    write_json(catalog_dir / "routes" / "index.json", route_index)
    write_json(catalog_dir / "fusions" / "candidates.json", [])


def load_json(path):
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def kernels_from_catalog(path, exclude_sources):
    catalog = load_json(path)
    sources = catalog.get("sources", {})
    artifacts = catalog.get("artifacts", {})
    kernels = []
    for route in catalog.get("routes", []):
        source_id = route.get("source_id")
        artifact_id = route.get("artifact_id")
        source = sources.get(source_id, {})
        artifact = artifacts.get(artifact_id, {})
        source_path = source.get("path")
        if not source_path:
            raise SystemExit(f"route {route.get('id')} references source without path: {source_id}")
        if source_path in exclude_sources:
            continue
        abi = route.get("abi", {})
        dispatch = route.get("dispatch", {})
        workgroup = dispatch.get("workgroup_size")
        if not isinstance(workgroup, list) or len(workgroup) != 3:
            raise SystemExit(f"route {route.get('id')} missing dispatch.workgroup_size")
        kernel = {
            "name": route["export_name"],
            "route_id": route["id"],
            "family": route["family"],
            "op": route["op"],
            "target_key": route["target_key"],
            "source": source_path,
            "format": None if route.get("loader_format") == "amdgpu-hsaco" else route.get("loader_format"),
            "priority": route.get("priority", 0),
            "binding_count": abi["binding_count"],
            "parameter_count": abi.get("parameter_count", abi["binding_count"] + 1),
            "constants_size": abi["constant_byte_length"],
            "workgroup_size": tuple(workgroup),
        }
        prefixes = route.get("target_prefixes") or artifact.get("target_prefixes")
        if prefixes:
            kernel["arch_prefixes"] = tuple(prefixes)
        kernels.append(kernel)
    return kernels


def c_string(value):
    return json.dumps(str(value))


def write_catalog(output, entries):
    with output.open("w", encoding="utf-8") as f:
        f.write("// Generated by generate_hrx_kernels.py. Do not edit.\n")
        f.write('#include "kernels/hrx_kernel_catalog.h"\n')
        f.write("#include <cstring>\n\n")

        written_symbols = set()
        for entry in entries:
            if entry["data_symbol"] in written_symbols:
                continue
            written_symbols.add(entry["data_symbol"])
            data = entry["data"]
            f.write(f"alignas(16) static const unsigned char {entry['data_symbol']}[] = {{\n")
            for i in range(0, len(data), 12):
                chunk = data[i:i + 12]
                f.write("    ")
                f.write(", ".join(f"0x{value:02x}" for value in chunk))
                f.write(",\n")
            f.write("};\n\n")

        f.write("static const ggml_hrx_kernel_entry k_hrx_kernel_catalog[] = {\n")
        for entry in entries:
            wx, wy, wz = entry["workgroup_size"]
            executable_format = entry["format"]
            executable_format_c = "nullptr" if not executable_format else f'"{executable_format}"'
            route_id = entry.get("route_id", entry["name"])
            family = entry.get("family", family_for_kernel_name(entry["name"]))
            op = entry.get("op", op_for_family(family))
            target_key = entry.get("target_key", "legacy")
            f.write("    {\n")
            f.write(f"        {c_string(entry['name'])},\n")
            f.write(f"        {c_string(route_id)},\n")
            f.write(f"        {c_string(family)},\n")
            f.write(f"        {c_string(op)},\n")
            f.write(f"        {c_string(target_key)},\n")
            f.write(f"        {c_string(entry['gfx_target'])},\n")
            f.write(f"        {entry['data_symbol']},\n")
            f.write(f"        sizeof({entry['data_symbol']}),\n")
            f.write(f"        {executable_format_c},\n")
            f.write(f"        {entry.get('priority', 0)},\n")
            f.write(f"        {entry['binding_count']},\n")
            f.write(f"        {entry['parameter_count']},\n")
            f.write(f"        {entry['constants_size']},\n")
            f.write(f"        {{ {wx}, {wy}, {wz} }},\n")
            f.write("    },\n")
        f.write("};\n\n")

        f.write("const ggml_hrx_kernel_entry * ggml_hrx_kernel_catalog_entries(size_t * count) {\n")
        f.write("    if (count) {\n")
        f.write("        *count = sizeof(k_hrx_kernel_catalog) / sizeof(k_hrx_kernel_catalog[0]);\n")
        f.write("    }\n")
        f.write("    return k_hrx_kernel_catalog;\n")
        f.write("}\n\n")

        f.write("const ggml_hrx_kernel_entry * ggml_hrx_kernel_catalog_find(const char * name, const char * gfx_target) {\n")
        f.write("    if (!name || !gfx_target) {\n")
        f.write("        return nullptr;\n")
        f.write("    }\n")
        f.write("    size_t count = 0;\n")
        f.write("    const ggml_hrx_kernel_entry * entries = ggml_hrx_kernel_catalog_entries(&count);\n")
        f.write("    for (size_t i = 0; i < count; ++i) {\n")
        f.write("        if (std::strcmp(entries[i].name, name) == 0 && std::strcmp(entries[i].gfx_target, gfx_target) == 0) {\n")
        f.write("            return &entries[i];\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("    return nullptr;\n")
        f.write("}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", action="append", default=[])
    parser.add_argument("--source-dir")
    parser.add_argument("--hsaco-root", default="")
    parser.add_argument("--output")
    parser.add_argument("--catalog", help="Assembled ggml-hrx-catalog-v1 JSON to embed.")
    parser.add_argument("--exclude-source", action="append", default=[], help="Catalog source path to omit from this build.")
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--emit-split-catalog", type=pathlib.Path, help="Bootstrap split catalog JSON from the legacy in-script table.")
    args = parser.parse_args()

    if args.emit_split_catalog:
        export_split_catalog(args.emit_split_catalog)
        return 0

    if not args.source_dir:
        sys.stderr.write("error: --source-dir is required\n")
        return 1

    exclude_sources = set(args.exclude_source)
    kernels = kernels_from_catalog(args.catalog, exclude_sources) if args.catalog else [
        kernel for kernel in KERNELS if kernel["source"] not in exclude_sources
    ]
    source_dir = pathlib.Path(args.source_dir)
    if args.list_sources:
        for source in unique_sources_for(kernels):
            print(source_dir / source)
        return 0

    if not args.output:
        sys.stderr.write("error: --output is required\n")
        return 1
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.arch:
        sys.stderr.write("error: at least one --arch is required\n")
        return 1
    if not args.hsaco_root:
        sys.stderr.write("error: --hsaco-root is required\n")
        return 1

    hsaco_root = pathlib.Path(args.hsaco_root)
    entries = []
    hsaco_cache = {}
    for arch in args.arch:
        for kernel in kernels:
            if not kernel_supports_arch(kernel, arch):
                continue
            source = kernel["source"]
            key = (arch, source)
            if key not in hsaco_cache:
                hsaco = hsaco_root / arch / hsaco_name_for_source(source)
                if not hsaco.exists():
                    sys.stderr.write(f"error: expected compiled HRX kernel object not found: {hsaco}\n")
                    return 1
                hsaco_cache[key] = {
                    "symbol": f"k_hrx_kernel_{symbol_part(arch)}_{len(hsaco_cache)}",
                    "data": hsaco.read_bytes(),
                }

            entry = dict(kernel)
            entry["gfx_target"] = arch
            entry["parameter_count"] = kernel.get("parameter_count", kernel["binding_count"] + 1)
            entry["data_symbol"] = hsaco_cache[key]["symbol"]
            entry["data"] = hsaco_cache[key]["data"]
            entries.append(entry)

    write_catalog(output, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
