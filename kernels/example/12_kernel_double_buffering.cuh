#pragma once

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {
namespace kernels {

namespace db2 {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB, typename T>
__device__ void loadFromGmem(int N, int K, float *A, float *B, float *As,
                             float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB, T &barrier) {

  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    cuda::memcpy_async(&As[(innerColA * 4 + 0) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 1) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 1],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 2) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 2],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 3) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 3],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    cuda::memcpy_async(&Bs[(innerRowB + offset) * BN + innerColB * 4],
                       &B[(innerRowB + offset) * N + innerColB * 4],
                       cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                       barrier);
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

}  // namespace db2

// Double buffering SGEMM kernel with async memcpy (version 2)
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  auto block = cooperative_groups::this_thread_block();
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier;
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> backBarrier;
  auto frontBarrierPtr = &frontBarrier;
  auto backBarrierPtr = &backBarrier;
  if (block.thread_rank() == 0) {
    init(&frontBarrier, block.size());
    init(&backBarrier, block.size());
  }
  __syncthreads();

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  int As_offset = 0;
  int Bs_offset = 0;

  // Double-buffering: load first blocktile into SMEM
  db2::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, As + As_offset * BM * BK, Bs + Bs_offset * BK * BN, innerRowA,
      innerColA, innerRowB, innerColB, (*frontBarrierPtr));

  for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
    // Double-buffering: load next blocktile into SMEM
    db2::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A + BK, B + BK * N, As + (1 - As_offset) * BM * BK,
        Bs + (1 - Bs_offset) * BK * BN, innerRowA, innerColA, innerRowB,
        innerColB, (*backBarrierPtr));

    (*frontBarrierPtr).arrive_and_wait();
    db2::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As + As_offset * BM * BK,
        Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += BK;
    B += BK * N;

    As_offset = 1 - As_offset;
    Bs_offset = 1 - Bs_offset;
    auto tmp = frontBarrierPtr;
    frontBarrierPtr = backBarrierPtr;
    backBarrierPtr = tmp;

    __syncthreads();
  }

  // Compute the last blocktile
  (*frontBarrierPtr).arrive_and_wait();
  db2::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      regM, regN, threadResults, As + As_offset * BM * BK,
      Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
      threadColInWarp);

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

// Wrapper function
inline void run_sgemmDoubleBuffering2(int M, int N, int K,
                                      float alpha, float* A, float* B,
                                      float beta, float* C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int WM = 64;
  constexpr int WN = 64;
  constexpr int WNITER = 4;
  constexpr int TM = 8;
  constexpr int TN = 4;
  constexpr int NUM_THREADS = 128;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  runSgemmDoubleBuffering2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
