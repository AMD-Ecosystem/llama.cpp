message(STATUS "Using HRX backend")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(hrx CONFIG REQUIRED)
find_package(loomc CONFIG REQUIRED)

set(GGML_HRX_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_HRX_CATALOG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/catalog")
set(GGML_HRX_ASSEMBLED_CATALOG "${GGML_HRX_GENERATED_DIR}/catalog.json")
set(GGML_HRX_EMBEDDED_CPP "${GGML_HRX_GENERATED_DIR}/hrx_embedded_catalog.cpp")
set(GGML_HRX_EMBEDDED_H "${GGML_HRX_GENERATED_DIR}/hrx_embedded_catalog.h")
file(GLOB_RECURSE GGML_HRX_CATALOG_PARTS CONFIGURE_DEPENDS
    "${GGML_HRX_CATALOG_SOURCE_DIR}/*.json"
)

get_target_property(_GGML_HRX_HRX_LOCATION hrx::hrx IMPORTED_LOCATION_RELWITHDEBINFO)
if(NOT _GGML_HRX_HRX_LOCATION)
    get_target_property(_GGML_HRX_HRX_LOCATION hrx::hrx IMPORTED_LOCATION_RELEASE)
endif()
if(NOT _GGML_HRX_HRX_LOCATION)
    get_target_property(_GGML_HRX_HRX_LOCATION hrx::hrx IMPORTED_LOCATION)
endif()
if(_GGML_HRX_HRX_LOCATION)
    get_filename_component(_GGML_HRX_HRX_PREFIX "${_GGML_HRX_HRX_LOCATION}/../.." ABSOLUTE)
endif()
if(hrx_DIR)
    get_filename_component(_GGML_HRX_HRX_DIR_PREFIX "${hrx_DIR}/../../../.." ABSOLUTE)
endif()
find_program(GGML_HRX_LOOM_LINK_EXECUTABLE
    NAMES loom-link
    HINTS
        "${_GGML_HRX_HRX_PREFIX}/bin"
        "${_GGML_HRX_HRX_DIR_PREFIX}/bin"
    REQUIRED
)

add_custom_command(
    OUTPUT ${GGML_HRX_ASSEMBLED_CATALOG}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/assemble_hrx_catalog.py
        --catalog-dir ${GGML_HRX_CATALOG_SOURCE_DIR}
        --out ${GGML_HRX_ASSEMBLED_CATALOG}
    DEPENDS
        ${GGML_HRX_CATALOG_PARTS}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/assemble_hrx_catalog.py
    VERBATIM
)

add_custom_command(
    OUTPUT
        ${GGML_HRX_EMBEDDED_CPP}
        ${GGML_HRX_EMBEDDED_H}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx_catalog.py
        --catalog ${GGML_HRX_ASSEMBLED_CATALOG}
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX_GENERATED_DIR}
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hrx_embedded.py
        --catalog ${GGML_HRX_ASSEMBLED_CATALOG}
        --source-root ${CMAKE_CURRENT_SOURCE_DIR}
        --artifact-root ${GGML_HRX_GENERATED_DIR}
        --out-cpp ${GGML_HRX_EMBEDDED_CPP}
        --out-h ${GGML_HRX_EMBEDDED_H}
    DEPENDS
        ${GGML_HRX_ASSEMBLED_CATALOG}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hrx_catalog.py
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hrx_embedded.py
    VERBATIM
)

option(GGML_HRX_EMBED_ROCM_LIBS "Bundle HRX/ROCm runtime libraries next to the HRX backend" OFF)
set(GGML_HRX_EMBED_LIBRARY_DIRS "" CACHE STRING "Library directories to scan when GGML_HRX_EMBED_ROCM_LIBS=ON")

if (NOT GGML_HRX_ROCM_PATH AND DEFINED ENV{ROCM_PATH})
    set(GGML_HRX_ROCM_PATH "$ENV{ROCM_PATH}")
endif()

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         ggml-hrx-catalog.cpp
                         ${GGML_HRX_EMBEDDED_CPP}
                         loom-jit/ggml-hrx-loom-jit.cpp
                         ../../include/ggml-hrx.h
                        )

target_link_libraries(ggml-hrx PRIVATE
    hrx::hrx
    loomc::loomc
)
target_include_directories(ggml-hrx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
    ${GGML_HRX_GENERATED_DIR}
)

if (GGML_HRX_EMBED_ROCM_LIBS)
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
        message(FATAL_ERROR "GGML_HRX_EMBED_ROCM_LIBS is currently implemented for Linux only")
    endif()

    if (GGML_BACKEND_DL)
        if (GGML_BACKEND_DIR)
            set(GGML_HRX_EMBED_INSTALL_DIR "${GGML_BACKEND_DIR}")
        else()
            set(GGML_HRX_EMBED_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
        endif()
    else()
        set(GGML_HRX_EMBED_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    set(GGML_HRX_EMBED_SEARCH_DIRS ${GGML_HRX_EMBED_LIBRARY_DIRS})
    if (GGML_HRX_ROCM_PATH)
        list(APPEND GGML_HRX_EMBED_SEARCH_DIRS "${GGML_HRX_ROCM_PATH}/lib")
    endif()
    if (GGML_HRX_EMBED_SEARCH_DIRS)
        list(REMOVE_DUPLICATES GGML_HRX_EMBED_SEARCH_DIRS)
    endif()

    function(ggml_hrx_collect_bundle_paths OUT_VAR REQUIRED)
        set(GGML_HRX_COLLECTED_PATHS)
        set(GGML_HRX_MISSING_PATTERNS)
        foreach(GGML_HRX_PATTERN IN LISTS ARGN)
            set(GGML_HRX_PATTERN_MATCHES)
            foreach(GGML_HRX_SEARCH_DIR IN LISTS GGML_HRX_EMBED_SEARCH_DIRS)
                file(GLOB GGML_HRX_DIR_MATCHES "${GGML_HRX_SEARCH_DIR}/${GGML_HRX_PATTERN}")
                list(APPEND GGML_HRX_PATTERN_MATCHES ${GGML_HRX_DIR_MATCHES})
            endforeach()
            if (GGML_HRX_PATTERN_MATCHES)
                list(APPEND GGML_HRX_COLLECTED_PATHS ${GGML_HRX_PATTERN_MATCHES})
            elseif(REQUIRED)
                list(APPEND GGML_HRX_MISSING_PATTERNS "${GGML_HRX_PATTERN}")
            endif()
        endforeach()
        if (GGML_HRX_MISSING_PATTERNS)
            message(FATAL_ERROR
                "GGML_HRX_EMBED_ROCM_LIBS could not find required libraries matching: "
                "${GGML_HRX_MISSING_PATTERNS}. Searched: ${GGML_HRX_EMBED_SEARCH_DIRS}")
        endif()
        if (GGML_HRX_COLLECTED_PATHS)
            list(REMOVE_DUPLICATES GGML_HRX_COLLECTED_PATHS)
        endif()
        set(${OUT_VAR} "${GGML_HRX_COLLECTED_PATHS}" PARENT_SCOPE)
    endfunction()

    ggml_hrx_collect_bundle_paths(GGML_HRX_EMBED_MAIN_LIBS TRUE
        "libhrx.so*"
        "libhsa-runtime64.so*"
        "libhsa-amd-aqlprofile64.so*"
        "librocprofiler-register.so*"
        "libomp.so*"
    )
    ggml_hrx_collect_bundle_paths(GGML_HRX_EMBED_SYSDEP_LIBS FALSE
        "rocm_sysdeps/lib/*.so*"
    )

    set_target_properties(ggml-hrx PROPERTIES
        BUILD_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib"
        INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib"
    )

    add_custom_command(TARGET ggml-hrx POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:ggml-hrx>"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${GGML_HRX_EMBED_MAIN_LIBS}
            "$<TARGET_FILE_DIR:ggml-hrx>"
        VERBATIM
    )
    install(FILES ${GGML_HRX_EMBED_MAIN_LIBS}
        DESTINATION "${GGML_HRX_EMBED_INSTALL_DIR}")

    if (GGML_HRX_EMBED_SYSDEP_LIBS)
        add_custom_command(TARGET ggml-hrx POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:ggml-hrx>/rocm_sysdeps/lib"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                ${GGML_HRX_EMBED_SYSDEP_LIBS}
                "$<TARGET_FILE_DIR:ggml-hrx>/rocm_sysdeps/lib"
            VERBATIM
        )
        install(FILES ${GGML_HRX_EMBED_SYSDEP_LIBS}
            DESTINATION "${GGML_HRX_EMBED_INSTALL_DIR}/rocm_sysdeps/lib")
    endif()
endif()
