cmake_minimum_required(VERSION 3.18)
project(ov_bench CXX)
set(CMAKE_CXX_STANDARD 17)

# ── 指定头文件目录 ──────────────────────────────────────────────────────────
include_directories(
    ${OV_DIR}/include
)

# ── 指定动态库搜索目录 ──────────────────────────────────────────────────────
link_directories(
    ${OV_DIR}/lib
)

add_executable(ov_bench main.cpp)

# ── 链接需要的动态库（去掉 lib 前缀和 .so/.dll 后缀）────────────────────────
target_link_libraries(ov_bench PRIVATE
    openvino
    openvino_genai
    openvino_tokenizers
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
