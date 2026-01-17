#pragma once

#include <cuda_runtime.h>
#include "utils/matrix.cuh"

namespace cudabench {

namespace kernels {

// ┌─────────────────────────────────────────────────────────────────────────┐
// │  CUDA Thread 좌표                     행렬 인덱싱                         │
// ├─────────────────────────────────────────────────────────────────────────┤
// │                                                                         │
// │  threadIdx.x → 가로 (열 방향)         row → 세로 (행)                     │
// │  threadIdx.y → 세로 (행 방향)         col → 가로 (열)                     │
// │                                                                         │
// │       x →                                  col →                        │
// │     ┌────────────────┐                   ┌────────────────┐             │
// │   y │ (0,0) (1,0) ...│               row │ [0,0] [0,1] ...│             │
// │   ↓ │ (0,1) (1,1) ...│               ↓   │ [1,0] [1,1] ...│             │
// │     │  ...   ...     │                   │  ...   ...     │             │
// │     └────────────────┘                   └────────────────┘             │
// │                                                                         │
// └─────────────────────────────────────────────────────────────────────────┘

// Naive SGEMM kernel - each thread computes one element of C
__global__ void sgemm_naive_kernel(int M, int N, int K,
                                   float alpha, const float* A, const float* B,
                                   float beta, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 전역 y좌표
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 전역 x좌표

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Wrapper function
inline void run_sgemm_naive(int M, int N, int K,
                            float alpha, float* A, float* B,
                            float beta, float* C) {
    dim3 blockDim(32, 32);                          // 블록 당 32x32 = 1024 threads
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32)); // 행렬 크기에 맞게 블록 수 계산

    /*
    <실행 설정>
    kernel<<<gridDim, blockDim>>>(args...);
            └───┬───┘ └───┬───┘
              블록 수  블록당 스레드 수

    // 기본
    kernel<<<gridDim, blockDim>>>(args);

    // Shared Memory 크기 지정
    kernel<<<gridDim, blockDim, sharedMemBytes>>>(args);

    // Stream 지정 (비동기 실행)
    kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);
    */

    sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }

}
}