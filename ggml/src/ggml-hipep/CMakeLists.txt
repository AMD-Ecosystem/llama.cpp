if (NOT GGML_HIP)
    message(FATAL_ERROR "GGML_HIPEP runs over ggml-hip: enable GGML_HIP")
endif()

# libhipep_backend (hip-ep, HIPDNN_EP_BUILD_GGUF): its package, <prefix>/lib/cmake/hipep_backend.
find_package(hipep_backend REQUIRED)
find_package(hip REQUIRED)

message(STATUS "Found hipep_backend: ${hipep_backend_DIR}")

ggml_add_backend_library(ggml-hipep
    ggml-hipep.cpp
    split-desc.cpp
    split-desc.h
)

target_link_libraries(ggml-hipep PRIVATE hipep_backend::hipep_backend hip::host)
if (NOT GGML_BACKEND_DL)
    # the ROCm devices it runs over, from ggml-hip's registry
    target_link_libraries(ggml-hipep PRIVATE ggml-hip)
endif()
