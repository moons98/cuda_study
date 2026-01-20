#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// Global memory coalesced SGEMM
// Thread indexing changed to improve memory access pattern
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
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
