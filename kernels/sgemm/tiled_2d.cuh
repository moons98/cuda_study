#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// 2D Block Tiling SGEMM
// Each thread computes a TM x TN tile of output
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tiled_2d_kernel(int M, int N, int K,
                                      float alpha, const float* A, const float* B,
                                      float beta, float* C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int numThreads = blockDim.x;

    // Number of threads in each dimension of the thread tile grid
    constexpr int numThreadsN = BN / TN;

    // Thread's tile position in the block output
    const int threadTileRow = (tid / numThreadsN) * TM;
    const int threadTileCol = (tid % numThreadsN) * TN;

    // Block position
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Registers for accumulation
    float threadResults[TM][TN] = {{0.0f}};

    // Registers for A and B fragments
    float regA[TM];
    float regB[TN];

    // Main loop over K
    for (int tileK = 0; tileK < K; tileK += BK) {
        // Collaborative load of A tile to shared memory
        for (int loadIdx = tid; loadIdx < BM * BK; loadIdx += numThreads) {
            int row = loadIdx / BK;
            int col = loadIdx % BK;
            int globalRow = blockRow + row;
            int globalCol = tileK + col;
            As[row][col] = (globalRow < M && globalCol < K) ?
                           A[globalRow * K + globalCol] : 0.0f;
        }

        // Collaborative load of B tile to shared memory
        for (int loadIdx = tid; loadIdx < BK * BN; loadIdx += numThreads) {
            int row = loadIdx / BN;
            int col = loadIdx % BN;
            int globalRow = tileK + row;
            int globalCol = blockCol + col;
            Bs[row][col] = (globalRow < K && globalCol < N) ?
                           B[globalRow * N + globalCol] : 0.0f;
        }

        __syncthreads();

        // Compute TM x TN output tile
        for (int k = 0; k < BK; k++) {
            // Load A fragment into registers
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                int aRow = threadTileRow + tm;
                regA[tm] = (aRow < BM) ? As[aRow][k] : 0.0f;
            }

            // Load B fragment into registers
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                int bCol = threadTileCol + tn;
                regB[tn] = (bCol < BN) ? Bs[k][bCol] : 0.0f;
            }

            // Outer product
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    threadResults[tm][tn] += regA[tm] * regB[tn];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int globalRow = blockRow + threadTileRow + tm;
            int globalCol = blockCol + threadTileCol + tn;
            if (globalRow < M && globalCol < N) {
                int idx = globalRow * N + globalCol;
                C[idx] = alpha * threadResults[tm][tn] + beta * C[idx];
            }
        }
    }
}

// Wrapper function
inline void run_sgemm_tiled_2d(int M, int N, int K,
                               float alpha, float* A, float* B,
                               float beta, float* C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Number of threads: (BM/TM) * (BN/TN) = 8 * 8 = 64
    constexpr int numThreads = (BM / TM) * (BN / TN);

    dim3 blockDim(numThreads);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_tiled_2d_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
