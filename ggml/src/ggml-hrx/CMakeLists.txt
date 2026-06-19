message(STATUS "Using HRX backend")

find_package(hrx CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if (NOT GGML_HRX_ROCM_PATH)
    if (DEFINED ENV{ROCM_PATH})
        set(GGML_HRX_ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        message(FATAL_ERROR "GGML_HRX_ROCM_PATH or ROCM_PATH is required to build HRX HIP kernels")
    endif()
endif()

if (NOT GGML_HRX_CLANGXX)
    find_program(GGML_HRX_CLANGXX
        NAMES clang++
        HINTS
            "${GGML_HRX_ROCM_PATH}/lib/llvm/bin"
            "${GGML_HRX_ROCM_PATH}/llvm/bin"
    )
endif()

if (NOT GGML_HRX_CLANGXX)
    message(FATAL_ERROR "clang++ is required to build HRX HIP kernels")
endif()

find_path(GGML_HRX_ROCWMMA_INCLUDE_DIR
    NAMES rocwmma/rocwmma.hpp
    HINTS "${GGML_HRX_ROCM_PATH}/include"
    NO_DEFAULT_PATH
)
set(GGML_HRX_CATALOG_EXCLUDE_SOURCE_ARGS)
if (GGML_HRX_ROCWMMA_INCLUDE_DIR)
    message(STATUS "HRX ROCWMMA headers: ${GGML_HRX_ROCWMMA_INCLUDE_DIR}")
else()
    message(WARNING "HRX ROCWMMA headers not found; skipping ROCWMMA-dependent HIP kernels")
    list(APPEND GGML_HRX_CATALOG_EXCLUDE_SOURCE_ARGS
        --exclude-source "flash_attn_ext_f32_f16_prefill_wmma16.hip.cpp")
endif()

set(GGML_HRX_AMDGPU_TARGETS "gfx1100" CACHE STRING "AMDGPU gfx targets to build HRX HSACO kernels for")
string(REPLACE "," ";" GGML_HRX_AMDGPU_TARGET_LIST "${GGML_HRX_AMDGPU_TARGETS}")

set(GGML_HRX_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_HRX_HSACO_ROOT "${GGML_HRX_GENERATED_DIR}/hsaco")
set(GGML_HRX_CATALOG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/catalog")
set(GGML_HRX_ASSEMBLED_CATALOG_JSON "${GGML_HRX_GENERATED_DIR}/hrx_catalog.json")
set(GGML_HRX_KERNEL_CATALOG_CPP "${GGML_HRX_GENERATED_DIR}/hrx_kernel_catalog.cpp")
set(GGML_HRX_KERNEL_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate_hrx_kernels.py")
set(GGML_HRX_CATALOG_ASSEMBLER "${CMAKE_CURRENT_SOURCE_DIR}/tools/assemble_hrx_catalog.py")
set(GGML_HRX_CATALOG_VALIDATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx_catalog.py")
file(GLOB_RECURSE GGML_HRX_CATALOG_INPUTS CONFIGURE_DEPENDS "${GGML_HRX_CATALOG_DIR}/*.json")

set(GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL)
set(GGML_HRX_CATALOG_ARCH_ARGS)
foreach(GGML_HRX_AMDGPU_TARGET IN LISTS GGML_HRX_AMDGPU_TARGET_LIST)
    string(STRIP "${GGML_HRX_AMDGPU_TARGET}" GGML_HRX_AMDGPU_TARGET)
    if (GGML_HRX_AMDGPU_TARGET STREQUAL "")
        continue()
    endif()
    list(APPEND GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL "${GGML_HRX_AMDGPU_TARGET}")
    list(APPEND GGML_HRX_CATALOG_ARCH_ARGS --arch "${GGML_HRX_AMDGPU_TARGET}")
endforeach()

if (NOT GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL)
    message(FATAL_ERROR "GGML_HRX_AMDGPU_TARGETS must contain at least one target")
endif()

file(MAKE_DIRECTORY "${GGML_HRX_GENERATED_DIR}" "${GGML_HRX_HSACO_ROOT}")
set(GGML_HRX_HSACO_OUTPUTS)

add_custom_command(
    OUTPUT "${GGML_HRX_ASSEMBLED_CATALOG_JSON}"
    COMMAND "${Python3_EXECUTABLE}"
        "${GGML_HRX_CATALOG_ASSEMBLER}"
        --catalog-dir "${GGML_HRX_CATALOG_DIR}"
        --out "${GGML_HRX_ASSEMBLED_CATALOG_JSON}"
    COMMAND "${Python3_EXECUTABLE}"
        "${GGML_HRX_CATALOG_VALIDATOR}"
        --catalog "${GGML_HRX_ASSEMBLED_CATALOG_JSON}"
        --source-root "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    DEPENDS
        "${GGML_HRX_CATALOG_ASSEMBLER}"
        "${GGML_HRX_CATALOG_VALIDATOR}"
        ${GGML_HRX_CATALOG_INPUTS}
    VERBATIM
)

function(ggml_hrx_add_kernel_source)
    cmake_parse_arguments(ARG "" "SOURCE;WAVEFRONT_SIZE" "DEPS;FLAGS;ARCH_PREFIXES" ${ARGN})
    if (NOT ARG_SOURCE)
        message(FATAL_ERROR "ggml_hrx_add_kernel_source requires SOURCE")
    endif()
    if (NOT ARG_WAVEFRONT_SIZE)
        message(FATAL_ERROR "ggml_hrx_add_kernel_source requires WAVEFRONT_SIZE")
    endif()
    if (NOT ARG_WAVEFRONT_SIZE STREQUAL "32" AND NOT ARG_WAVEFRONT_SIZE STREQUAL "64")
        message(FATAL_ERROR "unsupported HRX kernel wavefront size: ${ARG_WAVEFRONT_SIZE}")
    endif()

    get_filename_component(GGML_HRX_KERNEL_SOURCE_NAME "${ARG_SOURCE}" NAME)
    string(REPLACE ".hip.cpp" ".hsaco" GGML_HRX_HSACO_NAME "${GGML_HRX_KERNEL_SOURCE_NAME}")

    set(GGML_HRX_KERNEL_COMPILE_FLAGS)
    if (ARG_WAVEFRONT_SIZE STREQUAL "32")
        list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -mno-wavefrontsize64)
    else()
        list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -mwavefrontsize64)
    endif()
    list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS ${ARG_FLAGS})

    foreach(GGML_HRX_AMDGPU_TARGET IN LISTS GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL)
        if (ARG_ARCH_PREFIXES)
            set(GGML_HRX_BUILD_FOR_TARGET OFF)
            foreach(GGML_HRX_ARCH_PREFIX IN LISTS ARG_ARCH_PREFIXES)
                string(FIND "${GGML_HRX_AMDGPU_TARGET}" "${GGML_HRX_ARCH_PREFIX}" GGML_HRX_ARCH_PREFIX_POS)
                if (GGML_HRX_ARCH_PREFIX_POS EQUAL 0)
                    set(GGML_HRX_BUILD_FOR_TARGET ON)
                endif()
            endforeach()
            if (NOT GGML_HRX_BUILD_FOR_TARGET)
                continue()
            endif()
        endif()
        set(GGML_HRX_HSACO_DIR "${GGML_HRX_HSACO_ROOT}/${GGML_HRX_AMDGPU_TARGET}")
        set(GGML_HRX_HSACO_OUTPUT "${GGML_HRX_HSACO_DIR}/${GGML_HRX_HSACO_NAME}")
        file(MAKE_DIRECTORY "${GGML_HRX_HSACO_DIR}")
        add_custom_command(
            OUTPUT "${GGML_HRX_HSACO_OUTPUT}"
            COMMAND "${GGML_HRX_CLANGXX}"
                -x hip
                --offload-device-only
                --no-gpu-bundle-output
                "--offload-arch=${GGML_HRX_AMDGPU_TARGET}"
                -O2
                ${GGML_HRX_KERNEL_COMPILE_FLAGS}
                -c "${ARG_SOURCE}"
                -o "${GGML_HRX_HSACO_OUTPUT}"
            DEPENDS "${ARG_SOURCE}" ${ARG_DEPS}
            VERBATIM
        )
        list(APPEND GGML_HRX_HSACO_OUTPUTS "${GGML_HRX_HSACO_OUTPUT}")
    endforeach()
    set(GGML_HRX_HSACO_OUTPUTS "${GGML_HRX_HSACO_OUTPUTS}" PARENT_SCOPE)
endfunction()

set(GGML_HRX_WAVE32_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add8_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_add_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_rms_norm_mul_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_softplus_mul_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/argsort_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/clamp_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/concat_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/copy_f32_f16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/copy_strided_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/div_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_bf16_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_decode_reduce.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_decode_split.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_tile8.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f32_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_q4_0_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_q8_0_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_q8_0_q4_0_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gated_delta_net_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/get_rows_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/get_rows_q5_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_add_add_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_bf16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_bf16_set_rows.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_bf16_swiglu.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f16_batched.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f32_batched.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_mul.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_mul_q8_1.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_q8_1.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_swiglu.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_sum8_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantize_q8_1.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm_mul_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm_mul_rope_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rope_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rope_set_rows_f32_f16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/row_reduce_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/scale_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_f16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_q4_0.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_q8_0.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sigmoid_mul_f32_strided.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/soft_max_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/split_k_reduce_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ssm_conv_update_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/unary_f32.hip.cpp")

set(GGML_HRX_WAVE64_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ssm_conv_f32.hip.cpp"
)

foreach(GGML_HRX_KERNEL_SOURCE IN LISTS GGML_HRX_WAVE32_KERNEL_SOURCES)
    ggml_hrx_add_kernel_source(SOURCE "${GGML_HRX_KERNEL_SOURCE}" WAVEFRONT_SIZE 32)
endforeach()

foreach(GGML_HRX_KERNEL_SOURCE IN LISTS GGML_HRX_WAVE64_KERNEL_SOURCES)
    ggml_hrx_add_kernel_source(SOURCE "${GGML_HRX_KERNEL_SOURCE}" WAVEFRONT_SIZE 64)
endforeach()

ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_direct.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_direct_generic.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_direct_gfx11.inc"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_direct_d128.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_direct_gfx11.inc"
)
if (GGML_HRX_ROCWMMA_INCLUDE_DIR)
    ggml_hrx_add_kernel_source(
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_prefill_wmma16.hip.cpp"
        WAVEFRONT_SIZE 32
        ARCH_PREFIXES gfx11
        FLAGS "-I${GGML_HRX_ROCWMMA_INCLUDE_DIR}"
    )
endif()
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_wmma16_direct.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q5_k_q8_1_x4_mmq.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q6_k_q8_1_x4_mmq.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q6_k_wmma16_direct.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_wave64.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128_bk2.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql64_bk2.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql64_bk2_bpair.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql64_bk2_bquad.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128_bsplit4.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128_bpair.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128_bquad.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128_bquad_cr.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128_boct.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128x64.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128_padcache.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1_x4_mmql128.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_prefetch_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_pair64_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_w64_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_w64_h4load_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_w64_b64asm_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_w64_b64group_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_padded_w64_hi_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_wave64.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmq64x48.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk1.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_bquad.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_bpair.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_tn4_bquad.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_cr.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bquad.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2_bquad_bhalf.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql64_bk2.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bquad_splitk_part.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bquad.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bhalf.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bquad.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bpair.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bquad.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_cr.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_x4_mmql128_bquad.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_groupk2_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_groupk2_wait_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_combined96_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_batched4_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_batched4_p33_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_wave4row_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_wave4row_batchwait_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_packstage_fast_half_selected_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_store_stage_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_padded_w64_b64group_store_batch8_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_x4_wave64_direct.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_x4_mmql128.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_padded_w64_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_h4load_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_groupk2_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk64_padded44_w64_groupk2_wait_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_padded_w64_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_padded_w64_store_stage_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_padded_w64_b64group_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_padded_w64_b64group_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_padded_w64_b64group_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_padded_w64_b64group_store_stage_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x64.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x64_medium.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x96.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x104.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x112.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x112_splitqsum.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x104_splitqsum.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x128_splitqsum.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x128_splitqsum_wave64.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x128_splitqsum8.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x128_splitqsum_bstripe.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq32x128.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x128.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq32x64.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq128x64.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq128x64_splitqsum.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_mmq64x64_ab.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wave64.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_q8_1_x4_mmql128.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16.hip.cpp"
    WAVEFRONT_SIZE 32
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_hi_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_h4load_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_store_stage_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_store_stage_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_fullpair_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_fast_half_fullpair_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_fast_half_fullpair_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_fast_half_selected_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_fast_half_split_selected_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_fast_half_split_selected_bcopy_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_bcopy_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_abcopy_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_phase96_abcopy_bufferstore_phase0_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_phase96_abcopy_bufferstore_phase1_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_bm128phase96_abcopy_bufferstore_phase0_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_bm128phase96_abcopy_bufferstore_wg256.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_bm128phase96_abcopy_bufferstore_phase1_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_bm128phase96_abcopy_bufferstore_wg256.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_mediumfrag12_combined96_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_namedfrag_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_fulltile_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_dummyhalf_fullpair_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_packstage_fast_half_fullpair_bufferstore_lb2_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_streamcol_packstage_fast_half_fullpair_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_streamrow_packstage_fast_half_fullpair_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_streamrow_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_streamrow_packstage_abcopy_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_preuse_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_preuse_stagedwait_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_nopreuse_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_depwait_k2_raw_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_k2_directwait_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_nowait_namedfrag_stagedwait_packstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_b64group_dualstage_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_stagealloc_fullstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_padded_w64_bufferstore_wg256.hip.cpp"
    WAVEFRONT_SIZE 64
    ARCH_PREFIXES gfx11
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_shared.hip.cpp"
    WAVEFRONT_SIZE 32
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_common.hip.inc"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_wave32.hip.cpp"
    WAVEFRONT_SIZE 32
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_wave.inc"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_wave32_top8_norm.hip.cpp"
    WAVEFRONT_SIZE 32
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_common.hip.inc"
)

add_custom_command(
    OUTPUT "${GGML_HRX_KERNEL_CATALOG_CPP}"
    COMMAND "${Python3_EXECUTABLE}"
        "${GGML_HRX_KERNEL_GENERATOR}"
        --catalog "${GGML_HRX_ASSEMBLED_CATALOG_JSON}"
        ${GGML_HRX_CATALOG_EXCLUDE_SOURCE_ARGS}
        ${GGML_HRX_CATALOG_ARCH_ARGS}
        --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        --hsaco-root "${GGML_HRX_HSACO_ROOT}"
        --output "${GGML_HRX_KERNEL_CATALOG_CPP}"
    DEPENDS
        "${GGML_HRX_KERNEL_GENERATOR}"
        "${GGML_HRX_ASSEMBLED_CATALOG_JSON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hrx_kernel_catalog.h"
        ${GGML_HRX_HSACO_OUTPUTS}
    VERBATIM
)

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         "${GGML_HRX_KERNEL_CATALOG_CPP}"
                         ../../include/ggml-hrx.h
                         kernels/hrx_kernel_catalog.h
                        )

target_link_libraries(ggml-hrx PRIVATE hrx::hrx)
target_include_directories(ggml-hrx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${GGML_HRX_GENERATED_DIR}")

option(GGML_HRX_BUILD_HIP_BENCHES "Build native HIP microbenchmarks for HRX kernels" ON)

if (GGML_HRX_BUILD_HIP_BENCHES)
    list(GET GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL 0 GGML_HRX_HIP_BENCH_AMDGPU_TARGET)
    set(GGML_HRX_HIP_BENCH_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

    function(ggml_hrx_add_hip_bench TARGET_NAME SOURCE WAVEFRONT_SIZE)
        if (NOT WAVEFRONT_SIZE STREQUAL "32" AND NOT WAVEFRONT_SIZE STREQUAL "64")
            message(FATAL_ERROR "unsupported HRX HIP bench wavefront size: ${WAVEFRONT_SIZE}")
        endif()

        if (WAVEFRONT_SIZE STREQUAL "32")
            set(GGML_HRX_HIP_BENCH_WAVE_FLAG -mno-wavefrontsize64)
        else()
            set(GGML_HRX_HIP_BENCH_WAVE_FLAG -mwavefrontsize64)
        endif()

        set(GGML_HRX_HIP_BENCH_OUTPUT "${GGML_HRX_HIP_BENCH_OUTPUT_DIR}/${TARGET_NAME}")
        add_custom_command(
            OUTPUT "${GGML_HRX_HIP_BENCH_OUTPUT}"
            COMMAND "${GGML_HRX_CLANGXX}"
                -x hip
                "--offload-arch=${GGML_HRX_HIP_BENCH_AMDGPU_TARGET}"
                "--hip-path=${GGML_HRX_ROCM_PATH}"
                -O3
                -g
                "${GGML_HRX_HIP_BENCH_WAVE_FLAG}"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
                "${SOURCE}"
                -L "${GGML_HRX_ROCM_PATH}/lib"
                -lamdhip64
                -o "${GGML_HRX_HIP_BENCH_OUTPUT}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_mul.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k_swiglu.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_bf16.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f16_batched.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f32_batched.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk128_wg256.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_wmma16_vk64_padded44_w64_full64_wg256.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0_wmma16_vk128_wg256.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm_mul_f32.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_rms_norm_mul_f32_broadcast.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_common.hip.inc"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_shared.hip.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_wave.inc"
            VERBATIM
        )
        add_custom_target("${TARGET_NAME}" DEPENDS "${GGML_HRX_HIP_BENCH_OUTPUT}")
    endfunction()

    ggml_hrx_add_hip_bench(
        hrx-hip-bench-q4-moe
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/q4_moe_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-dense
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/dense_matvec_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-bf16-swiglu
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/bf16_swiglu_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-quant
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/quant_matvec_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-quant-q8
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/quant_q8_1_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-q8-add
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/q8_add_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-norm
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/norm_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-raw-buffer-store
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/raw_buffer_store_bench.hip.cpp"
        64)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-lds-halfword-stage
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/lds_halfword_stage_bench.hip.cpp"
        64)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-coopmat-store-contract
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/coopmat_store_contract_bench.hip.cpp"
        64)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-wmma-f16-lane-map
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/wmma_f16_lane_map_bench.hip.cpp"
        64)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-q5-wmma-full64-repro
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/q5_wmma_full64_repro_bench.hip.cpp"
        64)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-q8-wmma-repro
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/q8_wmma_repro_bench.hip.cpp"
        64)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-wmma-issue-window
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/wmma_issue_window_bench.hip.cpp"
        64)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-topk-w32
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/topk_moe_bench.hip.cpp"
        32)
    ggml_hrx_add_hip_bench(
        hrx-hip-bench-topk-w64
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip-bench/topk_moe_bench.hip.cpp"
        64)
    add_custom_target(hrx-hip-bench
        DEPENDS
            hrx-hip-bench-q4-moe
            hrx-hip-bench-dense
            hrx-hip-bench-bf16-swiglu
            hrx-hip-bench-quant
            hrx-hip-bench-quant-q8
            hrx-hip-bench-q8-add
            hrx-hip-bench-norm
            hrx-hip-bench-raw-buffer-store
            hrx-hip-bench-lds-halfword-stage
            hrx-hip-bench-coopmat-store-contract
            hrx-hip-bench-wmma-f16-lane-map
            hrx-hip-bench-q5-wmma-full64-repro
            hrx-hip-bench-q8-wmma-repro
            hrx-hip-bench-wmma-issue-window
            hrx-hip-bench-topk-w32
            hrx-hip-bench-topk-w64)
endif()
