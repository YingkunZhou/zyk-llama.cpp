cmake_minimum_required(VERSION 3.18)
project(model_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find onnxruntime-genai ────────────────────────────────────────────────────
# Option 1: set OGA_DIR to the root of the onnxruntime-genai installation
#   cmake -DOGA_DIR=/path/to/onnxruntime-genai ..
# Option 2: the headers/libs are in standard system paths

if(DEFINED OGA_DIR)
    set(OGA_INCLUDE_DIR "${OGA_DIR}/include")
    if(WIN32)
        set(OGA_LIB_DIR "${OGA_DIR}/lib")
        set(OGA_LIB "onnxruntime-genai")
    else()
        set(OGA_LIB_DIR "${OGA_DIR}/lib")
        set(OGA_LIB "onnxruntime-genai")
    endif()
else()
    # Try to find via pkg-config or common paths
    find_path(OGA_INCLUDE_DIR NAMES ort_genai.h
        HINTS
            /usr/local/include
            /usr/include
            $ENV{HOME}/.local/include
    )
    find_library(OGA_LIB_PATH NAMES onnxruntime-genai
        HINTS
            /usr/local/lib
            /usr/lib
            $ENV{HOME}/.local/lib
    )
    if(OGA_LIB_PATH)
        get_filename_component(OGA_LIB_DIR "${OGA_LIB_PATH}" DIRECTORY)
        set(OGA_LIB "onnxruntime-genai")
    endif()
endif()

if(NOT OGA_INCLUDE_DIR OR NOT EXISTS "${OGA_INCLUDE_DIR}/ort_genai.h")
    message(FATAL_ERROR
        "Cannot find ort_genai.h. Set -DOGA_DIR=/path/to/onnxruntime-genai")
endif()

message(STATUS "OGA include: ${OGA_INCLUDE_DIR}")
message(STATUS "OGA lib dir: ${OGA_LIB_DIR}")

# ── Target ───────────────────────────────────────────────────────────────────
add_executable(model_bench model_bench.cpp)

target_include_directories(model_bench PRIVATE ${OGA_INCLUDE_DIR})
target_link_directories(model_bench PRIVATE ${OGA_LIB_DIR})
target_link_libraries(model_bench PRIVATE ${OGA_LIB})

# On Linux, embed RPATH so the binary finds the .so at runtime
if(UNIX AND NOT APPLE)
    set_target_properties(model_bench PROPERTIES
        BUILD_RPATH "${OGA_LIB_DIR}"
        INSTALL_RPATH "${OGA_LIB_DIR}"
    )
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(model_bench PRIVATE /W3 /O2)
else()
    target_compile_options(model_bench PRIVATE -Wall -O2)
endif()
