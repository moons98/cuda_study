#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// Shared memory tiled SGEMM
// Uses shared memory to cache tiles of A and B
template <int BLOCKSIZE>
__global__ void sgemm_shared_mem_kernel(int M, int N, int K,
                                        float alpha, const float* A, const float* B,
                                        float beta, float* C) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    const int tid = threadIdx.x;
    const int tx = tid % BLOCKSIZE;
    const int ty = tid / BLOCKSIZE;

    const int row = blockIdx.y * BLOCKSIZE + ty;
    const int col = blockIdx.x * BLOCKSIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tileIdx = 0; tileIdx < CEIL_DIV(K, BLOCKSIZE); tileIdx++) {
        // Load tile of A into shared memory
        int aCol = tileIdx * BLOCKSIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = tileIdx * BLOCKSIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < BLOCKSIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Wrapper function
inline void run_sgemm_shared_mem(int M, int N, int K,
                                 float alpha, float* A, float* B,
                                 float beta, float* C) {
    constexpr int BLOCKSIZE = 32;
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));

    // Configure shared memory
    cudaFuncSetAttribute(sgemm_shared_mem_kernel<BLOCKSIZE>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    sgemm_shared_mem_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
