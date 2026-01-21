#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// Naive SGEMM kernel - each thread computes one element of C
// Matrix sizes: MxK * KxN = MxN
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // row: M dimension, col: N dimension
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// Wrapper function
inline void run_sgemm_naive(int M, int N, int K,
                            float alpha, float* A, float* B,
                            float beta, float* C) {
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
