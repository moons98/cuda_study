#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// Global memory coalesced SGEMM
// Uses 1D thread block, threadIdx.x % BLOCKSIZE -> col for coalesced access
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
    // 1D thread block: threadIdx.x in [0, BLOCKSIZE*BLOCKSIZE)
    // row: derived from threadIdx.x / BLOCKSIZE
    // col: derived from threadIdx.x % BLOCKSIZE (consecutive threads -> consecutive cols)
    const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// Wrapper function
template <const uint BLOCKSIZE = 32>
inline void run_sgemm_global_mem_coalesce(int M, int N, int K,
                                          float alpha, float* A, float* B,
                                          float beta, float* C) {
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    sgemm_global_mem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
