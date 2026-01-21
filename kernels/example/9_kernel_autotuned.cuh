#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

constexpr int K9_NUM_THREADS = 256;

// Autotuned SGEMM with warptile iterations
// NOTE: Assumes M, N, K are multiples of block tile sizes (no boundary check)
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  // Block tile position
  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  // Warptile dimensions (each thread processes multiple warptiles)
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // Number of warptile iterations per block tile
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // Thread position within warptile
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Advance pointers to block tile start
  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  // Thread position for vectorized loading (with striding)
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

  // Per-thread results and register cache
  float threadResults[WMITER * WNITER * TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  // Main loop over K dimension
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Vectorized load A with transpose (strided loading)
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    // Vectorized load B (strided loading)
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    // Compute partial results (iterate over warptiles)
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
    A += BK;
    B += BK * N;
  }

  // Write results to global memory (vectorized stores per warptile)
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
  }
}

// Wrapper function
inline void run_sgemmAutotuned(int M, int N, int K,
                               float alpha, float* A, float* B,
                               float beta, float* C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 blockDim(K9_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemmAutotuned<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
