#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {

namespace kernels {

// Global memory coalesced access SGEMM
// Thread indexing changed to improve memory access pattern
template <int BLOCKSIZE>
__global__ void sgemm_coalesced_kernel(int M, int N, int K,
                                       float alpha, const float* A, const float* B,
                                       float beta, float* C) {
    // Each thread computes one element in the row-direction
    // Thread (tx, ty) in block computes C[blockRow + tx][blockCol + ty]
    // But we use 1D thread block for better control
    
    const int tid = threadIdx.x;
    const int row = blockIdx.y * BLOCKSIZE + (tid / BLOCKSIZE);
    const int col = blockIdx.x * BLOCKSIZE + (tid % BLOCKSIZE); // thread를 1d로 보고 있으므로 이처럼 계산

    // MxK @ KxN = MxN
    if (row < M && col < N) { // MxN matrix에서 block이 딱 안 맞아떨어지는 경우 방지
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }

    // threadIdx.x가 증가하면 col이 따라서 증가함
    // 따라서 B, C 행렬 접근이 coalesced
    // A 접근의 경우, 같은 warp가 동일 주소 접근 -> GPU가 broadcast 처리해서 한 번만 읽음 (최악은 아님)
}

// Wrapper function
inline void run_sgemm_coalesced(int M, int N, int K,
                                float alpha, float* A, float* B,
                                float beta, float* C) {
    constexpr int BLOCKSIZE = 32;
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);           // 블록 당 32x32 = 1024 threads (1D로 설정)
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32)); // 메모리 저장 차원에 맞게 CUDA 차원은 모두 따라감
    sgemm_coalesced_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}  // namespace kernels
}  // namespace cudabench
