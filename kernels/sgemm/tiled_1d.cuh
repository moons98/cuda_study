#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// 1D Block Tiling SGEMM
// Each thread computes TM elements in the M dimension
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_tiled_1d_kernel(int M, int N, int K,
                                      float alpha, const float* A, const float* B,
                                      float beta, float* C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int numThreads = blockDim.x;

    // Thread's position in the output tile
    // Each thread computes TM elements in a column of the output
    const int threadRow = (tid / BN) * TM;
    const int threadCol = tid % BN;

    // Block's position in output matrix
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Registers for the output values
    float threadResults[TM] = {0.0f};

    // Loop over tiles of K dimension
    for (int tileIdx = 0; tileIdx < K; tileIdx += BK) {
        // Collaborative loading of A tile into shared memory
        for (int loadIdx = tid; loadIdx < BM * BK; loadIdx += numThreads) {
            int row = loadIdx / BK;
            int col = loadIdx % BK;
            int globalRow = blockRow + row;
            int globalCol = tileIdx + col;
            if (globalRow < M && globalCol < K) {
                As[row][col] = A[globalRow * K + globalCol];
            } else {
                As[row][col] = 0.0f;
            }
        }

        // Collaborative loading of B tile into shared memory
        for (int loadIdx = tid; loadIdx < BK * BN; loadIdx += numThreads) {
            int row = loadIdx / BN;
            int col = loadIdx % BN;
            int globalRow = tileIdx + row;
            int globalCol = blockCol + col;
            if (globalRow < K && globalCol < N) {
                Bs[row][col] = B[globalRow * N + globalCol];
            } else {
                Bs[row][col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial results
        if (threadRow < BM && threadCol < BN) {
            for (int k = 0; k < BK; k++) {
                float bVal = Bs[k][threadCol];
                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    if (threadRow + tm < BM) {
                        threadResults[tm] += As[threadRow + tm][k] * bVal;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        int globalRow = blockRow + threadRow + tm;
        int globalCol = blockCol + threadCol;
        if (globalRow < M && globalCol < N) {
            int idx = globalRow * N + globalCol;
            C[idx] = alpha * threadResults[tm] + beta * C[idx];
        }
    }
}

// Wrapper function
inline void run_sgemm_tiled_1d(int M, int N, int K,
                               float alpha, float* A, float* B,
                               float beta, float* C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    // Number of threads: BM/TM * BN = 8 * 64 = 512
    constexpr int numThreads = (BM / TM) * BN;

    dim3 blockDim(numThreads);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_tiled_1d_kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
