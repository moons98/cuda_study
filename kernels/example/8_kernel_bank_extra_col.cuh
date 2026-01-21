#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

// SGEMM with extra column padding for bank conflict resolution
// Uses extra columns in Bs to avoid bank conflicts
// NOTE: Assumes M, N, K are multiples of block tile sizes (no boundary check)
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankExtraCol(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  // Block tile position
  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  // Thread position for computing output (each thread computes TM x TN elements)
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  // Extra columns padding to avoid bank conflicts (stride != 32)
  const int extraCols = 5;
  __shared__ float Bs[BK * (BN + extraCols)];

  // Advance pointers to block tile start
  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  // Thread position for vectorized loading (float4 = 4 elements)
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // Per-thread results and register cache
  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  // Main loop over K dimension
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Vectorized load A with transpose
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // Vectorized load B with extra column stride for bank conflict avoidance
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
    __syncthreads();

    A += BK;
    B += BK * N;

    // Compute partial results (outer product accumulation)
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
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

  // Write results to global memory (vectorized stores)
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

// Wrapper function
inline void run_sgemmResolveBankExtraCol(int M, int N, int K,
                                         float alpha, float* A, float* B,
                                         float beta, float* C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 blockDim((BM * BN) / (TM * TN));
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemmResolveBankExtraCol<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
