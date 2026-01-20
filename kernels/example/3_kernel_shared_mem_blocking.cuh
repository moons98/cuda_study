#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// Shared memory blocking SGEMM
// Uses shared memory to cache tiles of A and B matrices
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // Advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

// Wrapper function
template <const int BLOCKSIZE = 32>
inline void run_sgemm_shared_mem_block(int M, int N, int K,
                                       float alpha, float* A, float* B,
                                       float beta, float* C) {
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    sgemm_shared_mem_block<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
