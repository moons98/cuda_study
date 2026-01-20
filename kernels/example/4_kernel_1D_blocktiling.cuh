#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// 1D Block Tiling SGEMM
// Each thread computes TM elements in the M dimension
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

// Wrapper function
inline void run_sgemm1DBlocktiling(int M, int N, int K,
                                   float alpha, float* A, float* B,
                                   float beta, float* C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    dim3 blockDim((BM * BK));
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm1DBlocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
