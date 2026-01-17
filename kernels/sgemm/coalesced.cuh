#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// Global memory coalesced access SGEMM
// Thread indexing changed to improve memory access pattern
template <int BLOCKSIZE>
__global__ void sgemm_coalesced_kernel(int M, int N, int K,
                                       float alpha, const float* A, const float* B,
                                       float beta, float* C) {
    // Each thread computes one element in the row-direction
    // Thread (tx, ty) in block computes C[blockRow + tx][blockCol + ty]
    // But we use 1D thread block for better control

    const int tid = threadIdx.x;
    const int row = blockIdx.y * BLOCKSIZE + (tid / BLOCKSIZE);
    const int col = blockIdx.x * BLOCKSIZE + (tid % BLOCKSIZE);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Wrapper function
inline void run_sgemm_coalesced(int M, int N, int K,
                                float alpha, float* A, float* B,
                                float beta, float* C) {
    constexpr int BLOCKSIZE = 32;
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
    sgemm_coalesced_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
