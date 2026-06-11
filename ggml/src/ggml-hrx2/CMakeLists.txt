message(STATUS "Using HRX2 backend")

find_package(hrx CONFIG REQUIRED)

ggml_add_backend_library(ggml-hrx2
    ggml-hrx2.cpp
)

target_link_libraries(ggml-hrx2 PRIVATE hrx::hrx hrx::loom_jit)
target_include_directories(ggml-hrx2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor)
