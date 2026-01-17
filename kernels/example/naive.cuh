#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// Naive SGEMM kernel - each thread computes one element of C
__global__ void sgemm_naive_kernel(int M, int N, int K,
                                   float alpha, const float* A, const float* B,
                                   float beta, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Wrapper function
inline void run_sgemm_naive(int M, int N, int K,
                            float alpha, float* A, float* B,
                            float beta, float* C) {
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
