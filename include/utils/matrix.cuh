#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <fstream>

namespace cudabench {

// Matrix initialization functions
void randomize_matrix(float* mat, size_t N);
void range_init_matrix(float* mat, size_t N);
void zero_init_matrix(float* mat, size_t N);
void constant_init_matrix(float* mat, size_t N, float value);

// Matrix copy
void copy_matrix(const float* src, float* dest, size_t N);

// Matrix print (for debugging small matrices)
void print_matrix(const float* A, int M, int N, const std::string& title = "");
void print_matrix_to_file(const float* A, int M, int N, std::ofstream& fs);

// CUDA error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Helper macro for ceiling division
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

}  // namespace cudabench
