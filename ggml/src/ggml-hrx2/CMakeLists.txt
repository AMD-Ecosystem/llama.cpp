message(STATUS "Using HRX2 backend")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(hrx CONFIG REQUIRED)

set(GGML_HRX2_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_HRX2_ARTIFACT_ROOT "${GGML_HRX2_GENERATED_DIR}/catalog")
set(GGML_HRX2_CATALOG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/catalog")
set(GGML_HRX2_ASSEMBLED_CATALOG "${GGML_HRX2_GENERATED_DIR}/catalog.json")
set(GGML_HRX2_EMBEDDED_CPP "${GGML_HRX2_GENERATED_DIR}/hrx2_embedded_catalog.cpp")
set(GGML_HRX2_EMBEDDED_H "${GGML_HRX2_GENERATED_DIR}/hrx2_embedded_catalog.h")
set(GGML_HRX2_ARTIFACTS
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rms_norm_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rms_norm_mul_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/add_rms_norm_mul_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q8_0_f32_contiguous.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q8_0_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q8_0_f32_static_packed.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q8_0_f32_static_packed_scf_unroll.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/quantize_q8_1_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q4_k_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q4_k_swiglu_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_id_q4_k_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q5_k_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_id_q5_k_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q6_k_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_id_q6_k_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_f32_f32_static.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_f16_f32_batched.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/copy_f32_f16.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/cont_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/cont_set_rows_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/swiglu_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/pointwise_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/set_rows_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rope_set_rows_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/sum_rows_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/get_rows_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/get_rows_q4_k_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/get_rows_q5_k_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/get_rows_q8_0_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/get_rows_q6_k_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/get_rows_moe_weights_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/argsort_f32_i32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rope_neox_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rope_normal_f32.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rope_neox_f32_freq.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rope_normal_f32_freq.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/soft_max_f32.loombc"
)
file(GLOB_RECURSE GGML_HRX2_LOOM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.loom"
)
file(GLOB_RECURSE GGML_HRX2_CATALOG_PARTS CONFIGURE_DEPENDS
    "${GGML_HRX2_CATALOG_SOURCE_DIR}/*.json"
)

get_target_property(_GGML_HRX2_HRX_LOCATION hrx::hrx IMPORTED_LOCATION_RELWITHDEBINFO)
if(NOT _GGML_HRX2_HRX_LOCATION)
    get_target_property(_GGML_HRX2_HRX_LOCATION hrx::hrx IMPORTED_LOCATION)
endif()
if(_GGML_HRX2_HRX_LOCATION)
    get_filename_component(_GGML_HRX2_HRX_PREFIX "${_GGML_HRX2_HRX_LOCATION}/../.." ABSOLUTE)
endif()
find_program(GGML_HRX2_LOOM_LINK_EXECUTABLE
    NAMES loom-link
    HINTS "${_GGML_HRX2_HRX_PREFIX}/bin"
    REQUIRED
)

if(NOT GGML_HRX_ROCM_PATH)
    if(DEFINED ENV{GGML_HRX_ROCM_PATH})
        set(GGML_HRX_ROCM_PATH "$ENV{GGML_HRX_ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(GGML_HRX_ROCM_PATH "$ENV{ROCM_PATH}")
    endif()
endif()
if(NOT GGML_HRX2_CLANGXX)
    find_program(GGML_HRX2_CLANGXX
        NAMES clang++
        HINTS
            "${GGML_HRX_ROCM_PATH}/lib/llvm/bin"
            "${GGML_HRX_ROCM_PATH}/llvm/bin"
            "${GGML_HRX_ROCM_PATH}/bin"
    )
endif()
if(NOT GGML_HRX2_CLANGXX)
    message(FATAL_ERROR "clang++ is required to build HRX2 HIP bridge kernels")
endif()

set(GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET "gfx1100" CACHE STRING
    "AMDGPU gfx target for target-specific HRX2 HIP bridge HSACO kernels")
set(GGML_HRX2_Q6_HIP_BRIDGE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hip/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp")
set(GGML_HRX2_Q6_HIP_BRIDGE_DEPS
    "${GGML_HRX2_Q6_HIP_BRIDGE_SOURCE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-hrx/kernels/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-hrx/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
)
set(GGML_HRX2_Q6_HIP_BRIDGE_HSACO
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/${GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET}/mul_mat_vec_q6_k_q8_1_wave64.hsaco")
set(GGML_HRX2_Q5_HIP_BRIDGE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hip/mul_mat_vec_q5_k_q8_1_wave64.hip.cpp")
set(GGML_HRX2_Q5_HIP_BRIDGE_DEPS
    "${GGML_HRX2_Q5_HIP_BRIDGE_SOURCE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-hrx/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
)
set(GGML_HRX2_Q5_HIP_BRIDGE_HSACO
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/${GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET}/mul_mat_vec_q5_k_q8_1_wave64.hsaco")
set(GGML_HRX2_Q4_HIP_BRIDGE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hip/mul_mat_vec_q4_k_q8_1_wave64.hip.cpp")
set(GGML_HRX2_Q4_HIP_BRIDGE_DEPS
    "${GGML_HRX2_Q4_HIP_BRIDGE_SOURCE}"
)
set(GGML_HRX2_Q4_HIP_BRIDGE_HSACO
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/${GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET}/mul_mat_vec_q4_k_q8_1_wave64.hsaco")
list(APPEND GGML_HRX2_ARTIFACTS
    "${GGML_HRX2_Q6_HIP_BRIDGE_HSACO}"
    "${GGML_HRX2_Q5_HIP_BRIDGE_HSACO}"
    "${GGML_HRX2_Q4_HIP_BRIDGE_HSACO}")

if(NOT TARGET loom::binding::c::loomc OR NOT TARGET loom::binding::c::target::amdgpu)
    message(FATAL_ERROR
        "HRX2 requires HRX to export Loom C API CMake targets "
        "loom::binding::c::loomc and loom::binding::c::target::amdgpu")
endif()

add_custom_command(
    OUTPUT ${GGML_HRX2_ASSEMBLED_CATALOG}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/assemble_hrx2_catalog.py
        --catalog-dir ${GGML_HRX2_CATALOG_SOURCE_DIR}
        --out ${GGML_HRX2_ASSEMBLED_CATALOG}
    DEPENDS
        ${GGML_HRX2_CATALOG_PARTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/assemble_hrx2_catalog.py
    VERBATIM
)

add_custom_command(
    OUTPUT ${GGML_HRX2_ARTIFACTS}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx2_catalog.py
        --catalog ${GGML_HRX2_ASSEMBLED_CATALOG}
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/link_hrx2_artifacts.py
        --catalog ${GGML_HRX2_ASSEMBLED_CATALOG}
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
        --loom-link ${GGML_HRX2_LOOM_LINK_EXECUTABLE}
    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/${GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET}"
    COMMAND
        "${GGML_HRX2_CLANGXX}"
        -x hip
        --offload-device-only
        --no-gpu-bundle-output
        "--offload-arch=${GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET}"
        -O3
        -mwavefrontsize64
        -c "${GGML_HRX2_Q6_HIP_BRIDGE_SOURCE}"
        -o "${GGML_HRX2_Q6_HIP_BRIDGE_HSACO}"
    COMMAND
        "${GGML_HRX2_CLANGXX}"
        -x hip
        --offload-device-only
        --no-gpu-bundle-output
        "--offload-arch=${GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET}"
        -O3
        -mwavefrontsize64
        -c "${GGML_HRX2_Q5_HIP_BRIDGE_SOURCE}"
        -o "${GGML_HRX2_Q5_HIP_BRIDGE_HSACO}"
    COMMAND
        "${GGML_HRX2_CLANGXX}"
        -x hip
        --offload-device-only
        --no-gpu-bundle-output
        "--offload-arch=${GGML_HRX2_HIP_BRIDGE_AMDGPU_TARGET}"
        -O3
        -mwavefrontsize64
        -c "${GGML_HRX2_Q4_HIP_BRIDGE_SOURCE}"
        -o "${GGML_HRX2_Q4_HIP_BRIDGE_HSACO}"
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx2_catalog.py
        --catalog ${GGML_HRX2_ASSEMBLED_CATALOG}
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
        --require-artifacts
    DEPENDS
        ${GGML_HRX2_ASSEMBLED_CATALOG}
        ${GGML_HRX2_LOOM_SOURCES}
        ${GGML_HRX2_Q6_HIP_BRIDGE_DEPS}
        ${GGML_HRX2_Q5_HIP_BRIDGE_DEPS}
        ${GGML_HRX2_Q4_HIP_BRIDGE_DEPS}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/link_hrx2_artifacts.py
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx2_catalog.py
    VERBATIM
)

add_custom_command(
    OUTPUT
        ${GGML_HRX2_EMBEDDED_CPP}
        ${GGML_HRX2_EMBEDDED_H}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hrx2_embedded.py
        --catalog ${GGML_HRX2_ASSEMBLED_CATALOG}
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
        --out-cpp ${GGML_HRX2_EMBEDDED_CPP}
        --out-h ${GGML_HRX2_EMBEDDED_H}
    DEPENDS
        ${GGML_HRX2_ASSEMBLED_CATALOG}
        ${GGML_HRX2_LOOM_SOURCES}
        ${GGML_HRX2_ARTIFACTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hrx2_embedded.py
    VERBATIM
)

ggml_add_backend_library(ggml-hrx2
    ggml-hrx2.cpp
    ggml-hrx2-catalog.cpp
    ${GGML_HRX2_EMBEDDED_CPP}
    loom-jit/ggml-hrx2-loom-jit.cpp
)

target_link_libraries(ggml-hrx2 PRIVATE
    hrx::hrx
    loom::binding::c::loomc
    loom::binding::c::target::amdgpu
)
target_include_directories(ggml-hrx2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
    ${GGML_HRX2_GENERATED_DIR}
)
