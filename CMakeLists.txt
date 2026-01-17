cmake_minimum_required(VERSION 3.18)
project(CUDA_Benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)  # RTX 20xx, 30xx, 40xx, 50xx (Blackwell)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Source files
set(CORE_SOURCES
    src/core/timer.cu
    src/core/device_info.cu
    src/core/benchmark.cu
)

set(UTILS_SOURCES
    src/utils/matrix.cu
    src/utils/verification.cu
    src/utils/csv_export.cu
)

set(ANALYSIS_SOURCES
    src/analysis/comparison.cu
    src/analysis/roofline.cu
    src/analysis/profiling_cmd.cu
)

set(KERNEL_SOURCES
    src/kernels/kernel_registry.cu
)

# Main executable
add_executable(benchmark
    src/main.cu
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${ANALYSIS_SOURCES}
    ${KERNEL_SOURCES}
)

target_include_directories(benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels
)

target_link_libraries(benchmark
    CUDA::cublas
    CUDA::cudart
)

# Compiler flags
if(MSVC)
    target_compile_options(benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -Xcompiler=/W3
        >
    )
else()
    target_compile_options(benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -Xcompiler=-Wall
        >
    )
endif()

# Debug/Release configurations
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Output directory
set_target_properties(benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
