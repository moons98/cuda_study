#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// 2D Block Tiling SGEMM
// Each thread computes a TM x TN tile of output
// NOTE: Assumes M, N, K are multiples of block tile sizes (no boundary check)
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  // Block tile position
  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  // Number of threads per block tile
  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // Thread position for computing output (each thread computes TM x TN elements)
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Advance pointers to block tile start
  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  // Thread position for loading into shared memory (with striding)
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreadsBlocktile / BN;

  // Per-thread results and register cache
  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  // Main loop over K dimension
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load tiles into shared memory (with striding for multiple loads per thread)
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    // Compute partial results (outer product accumulation)
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // Write results to global memory
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}

// Wrapper function
inline void run_sgemm2DBlocktiling(int M, int N, int K,
                                   float alpha, float* A, float* B,
                                   float beta, float* C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 blockDim((BM * BN) / (TM * TN));
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
