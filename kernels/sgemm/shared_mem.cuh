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

    // 각 thread는 C의 하나의 원소를 계산
    // 각 k-타일 반복마다, 블록의 thread들이 협력해서 A_tile(32×32)과 B_tile(32×32)을 채움 (각 thread가 타일의 일부 원소를 담당, GMEM -> SMEM)
    // 이후, SMEM의 타일을 사용해 BLOCKSIZE번의 FMA 연산 수행

    // SMEM 없이: 각 thread는 k 루프 동안 A와 B를 매번 글로벌에서 로드해야 함 (예: K=32면 A 32회, B 32회)
    // SMEM 사용: 동일한 A_tile, B_tile 원소를 블록 내 여러 thread가 공유하므로, 블록 내 중복 글로벌 로드가 타일 차원 수준으로 제거 (즉, “각 원소를 블록당 1번만” 로드)

    // Loop over tiles - K 방향으로 순회
    for (int tileIdx = 0; tileIdx < CEIL_DIV(K, BLOCKSIZE); tileIdx++) {
        // Load tile of A into shared memory
        int aCol = tileIdx * BLOCKSIZE + tx; // 현재 타일에서 이 thread가 담당할 A 형렬의 열 인덱스
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        }
        else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = tileIdx * BLOCKSIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        }
        else
        {
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
    if (row < M  && col < N) {
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
    // cudaFuncSetAttribute( // 커널 함수의 속성을 설정하는 CUDA API
    //     커널 함수,         // 어떤 커널에
    //     속성 종류,         // 무슨 속성을
    //     값                // 어떻게 설정할지
    // );
    cudaFuncSetAttribute(sgemm_shared_mem_kernel<BLOCKSIZE>,
                         cudaFuncAttributePreferredSharedMemoryCarveout, // shared_memory와 L1-cache 비율 지정
                         cudaSharedmemCarveoutMaxShared);                // shared_memory 최대로 설정
    
    sgemm_shared_mem_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
