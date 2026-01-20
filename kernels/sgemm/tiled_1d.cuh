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
    const int blockCol = blockIdx.x * BN; // blockIdx(BM x BN) 기준으로 C(M x N)에서의 좌표

    // Registers for the output values
    float threadResults[TM] = {0.0f};

    // SMEM: 
    // 블록 내 여러 thread가 반복해서 필요로 하는 A_tile, B_tile을 글로벌에서 한 번만 읽어 SMEM에 캐시, 
    // 이후 계산은 SMEM에서 재사용하여 중복 글로벌 load를 타일 크기 수준으로 제거

    // 1D block tiling(TM):
    // 각 thread가 출력 TM개를 담당 (M 방향으로 연속된 TM개)
    // k 루프마다 Bs[k][col] 값을 레지스터에 유지한 채 TM개의 출력 누산에 재사용
    // 결과적으로 SMEM → register 로드 비용이 TM만큼 줄어듦

    // Loop over tiles of K dimension
    // 하나의 output이 만들어지기 위해서는 K번 iteration 돌아야 함 (전체 행/열 순회)
    for (int tileIdx = 0; tileIdx < K; tileIdx += BK) {
        // Collaborative loading of A tile into shared memory
        // As(BM x BK)이므로 총 BM*BK만큼이 load되어야 함
        // 이 케이스에서 numThreads == BM*BK이므로 한 개씩만 load (아닐 경우 복수 갯수 load)
        for (int loadIdx = tid; loadIdx < BM * BK; loadIdx += numThreads) {
            int row = loadIdx / BK;
            int col = loadIdx % BK; // loadIdx는 As를 1차원으로 펼쳤을 때의 idx
            int globalRow = blockRow + row;
            int globalCol = tileIdx + col;
            if (globalRow < M && globalCol < K) {
                As[row][col] = A[globalRow * K + globalCol];
            }
            else {
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
        // As(BM x BK), Bs(BK x BN) 타일에서 BK번 k-loop × TM번 tm-loop 수행
        if (threadRow < BM && threadCol < BN) {
            for (int k = 0; k < BK; k++) {
                float bVal = Bs[k][threadCol];
                #pragma unroll                    // 컴파일러에게 unroll 힌트 제공
                for (int tm = 0; tm < TM; tm++) { // TM번의 연산에서 bVal 값을 재활용
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
    for (int tm = 0; tm < TM; tm++) { // TM개의 output write
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
    // As[BM][BK]에서, BM을 TM씩 잘라서 한 thread가 output 담당하기 때문
    constexpr int numThreads = (BM / TM) * BN;

    dim3 blockDim(numThreads);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_tiled_1d_kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
