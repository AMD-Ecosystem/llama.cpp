message(STATUS "Using HRX2 backend")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(hrx CONFIG REQUIRED)

set(GGML_HRX2_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_HRX2_ARTIFACT_ROOT "${GGML_HRX2_GENERATED_DIR}/catalog")
set(GGML_HRX2_EMBEDDED_CPP "${GGML_HRX2_GENERATED_DIR}/hrx2_embedded_catalog.cpp")
set(GGML_HRX2_EMBEDDED_H "${GGML_HRX2_GENERATED_DIR}/hrx2_embedded_catalog.h")
set(GGML_HRX2_ARTIFACTS
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/rms_norm_f32_contiguous.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q8_0_f32_contiguous.loombc"
    "${GGML_HRX2_ARTIFACT_ROOT}/artifacts/mul_mat_q8_0_f32_static.loombc"
)
file(GLOB_RECURSE GGML_HRX2_LOOM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.loom"
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

set(GGML_HRX2_LOOM_JIT_SOURCES)
set(GGML_HRX2_LOOM_JIT_LIBS)
set(GGML_HRX2_LOOM_JIT_DEFINES)
if(TARGET loom::binding::c::loomc AND TARGET loom::binding::c::target::amdgpu)
    list(APPEND GGML_HRX2_LOOM_JIT_SOURCES
        loom-jit/ggml-hrx2-loom-jit.cpp
    )
    list(APPEND GGML_HRX2_LOOM_JIT_LIBS
        loom::binding::c::loomc
        loom::binding::c::target::amdgpu
    )
else()
    if(NOT TARGET hrx::loom_jit)
        message(FATAL_ERROR
            "HRX2 requires either raw Loom C API CMake targets "
            "(loom::binding::c::loomc and loom::binding::c::target::amdgpu) "
            "or the transitional hrx::loom_jit wrapper target")
    endif()
    list(APPEND GGML_HRX2_LOOM_JIT_LIBS hrx::loom_jit)
    list(APPEND GGML_HRX2_LOOM_JIT_DEFINES GGML_HRX2_USE_HRX_LOOM_JIT)
endif()

add_custom_command(
    OUTPUT ${GGML_HRX2_ARTIFACTS}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx2_catalog.py
        --catalog ${CMAKE_CURRENT_SOURCE_DIR}/catalog.json
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/link_hrx2_artifacts.py
        --catalog ${CMAKE_CURRENT_SOURCE_DIR}/catalog.json
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
        --loom-link ${GGML_HRX2_LOOM_LINK_EXECUTABLE}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx2_catalog.py
        --catalog ${CMAKE_CURRENT_SOURCE_DIR}/catalog.json
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
        --require-artifacts
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/catalog.json
        ${GGML_HRX2_LOOM_SOURCES}
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
        --catalog ${CMAKE_CURRENT_SOURCE_DIR}/catalog.json
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX2_ARTIFACT_ROOT}
        --out-cpp ${GGML_HRX2_EMBEDDED_CPP}
        --out-h ${GGML_HRX2_EMBEDDED_H}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/catalog.json
        ${GGML_HRX2_LOOM_SOURCES}
        ${GGML_HRX2_ARTIFACTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hrx2_embedded.py
    VERBATIM
)

ggml_add_backend_library(ggml-hrx2
    ggml-hrx2.cpp
    ggml-hrx2-catalog.cpp
    ${GGML_HRX2_EMBEDDED_CPP}
    ${GGML_HRX2_LOOM_JIT_SOURCES}
)

target_link_libraries(ggml-hrx2 PRIVATE hrx::hrx ${GGML_HRX2_LOOM_JIT_LIBS})
target_compile_definitions(ggml-hrx2 PRIVATE ${GGML_HRX2_LOOM_JIT_DEFINES})
target_include_directories(ggml-hrx2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
    ${GGML_HRX2_GENERATED_DIR}
)
