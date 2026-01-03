# Core Library (libcottus_core)
# Contains all C++ logic (BlockAllocator, PageTable, etc.)
# Pure C++, no Python dependencies.

add_library(cottus_core STATIC
    block_allocator.cpp
    page_table.cpp
    engine.cpp
    generic_transformer.cpp
    paged_attention_cpu.cpp
    paged_attention_cuda.cu
    compute_primitives_cpu.cpp
    compute_primitives_cuda.cu
)

set_target_properties(cottus_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Enable debug instrumentation for parity testing
# target_compile_definitions(cottus_core PRIVATE COTTUS_DEBUG_PARITY=1)

# Link cuBLAS for GEMM operations
target_link_libraries(cottus_core PUBLIC cublas)

# PyBind11 Extension (_cottus_C)
# Disable LTO to avoid CUDA/GCC version mismatch
set(PYBIND11_LTO_CXX_FLAGS "" CACHE STRING "" FORCE)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_cottus_C
    pybind_wrapper.cpp
)

# Disable LTO for CUDA compatibility
set_target_properties(_cottus_C PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION FALSE
)

# Enable debug instrumentation for parity testing
# target_compile_definitions(_cottus_C PRIVATE COTTUS_DEBUG_PARITY=1)

# Link core library to extension
target_link_libraries(_cottus_C PRIVATE cottus_core)
