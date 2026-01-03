cmake_minimum_required(VERSION 3.18)

project(cottus_runtime LANGUAGES CXX CUDA)

# Disable LTO globally to avoid CUDA 12.0 / GCC 13.1 LTO version mismatch
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_position_independent_code ON)

# Source Directory
add_subdirectory(cottus/csrc)

# Tests (Build by default)
option(BUILD_TESTS "Build C++ tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
