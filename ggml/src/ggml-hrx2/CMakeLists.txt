message(STATUS "Using HRX2 backend")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(hrx CONFIG REQUIRED)

set(GGML_HRX2_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_HRX2_EMBEDDED_CPP "${GGML_HRX2_GENERATED_DIR}/hrx2_embedded_catalog.cpp")
set(GGML_HRX2_EMBEDDED_H "${GGML_HRX2_GENERATED_DIR}/hrx2_embedded_catalog.h")
file(GLOB_RECURSE GGML_HRX2_LOOM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.loom"
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
        --out-cpp ${GGML_HRX2_EMBEDDED_CPP}
        --out-h ${GGML_HRX2_EMBEDDED_H}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/catalog.json
        ${GGML_HRX2_LOOM_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hrx2_embedded.py
    VERBATIM
)

ggml_add_backend_library(ggml-hrx2
    ggml-hrx2.cpp
    ggml-hrx2-catalog.cpp
    ${GGML_HRX2_EMBEDDED_CPP}
)

target_link_libraries(ggml-hrx2 PRIVATE hrx::hrx hrx::loom_jit)
target_include_directories(ggml-hrx2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
    ${GGML_HRX2_GENERATED_DIR}
)
