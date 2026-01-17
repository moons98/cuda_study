#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cudabench {
namespace kernels {

// Global cuBLAS handle for reference implementation
inline cublasHandle_t& get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

// cuBLAS FP32 SGEMM wrapper
inline void run_cublas_sgemm(int M, int N, int K,
                             float alpha, float* A, float* B,
                             float beta, float* C) {
    cublasHandle_t handle = get_cublas_handle();

    // cuBLAS uses column-major order
    // For row-major matrices: C = A * B becomes C^T = B^T * A^T
    // So we call cublasSgemm with swapped A and B, transposed result
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
}

// cuBLAS with TF32 (faster on Ampere+)
inline void run_cublas_sgemm_tf32(int M, int N, int K,
                                  float alpha, float* A, float* B,
                                  float beta, float* C) {
    cublasHandle_t handle = get_cublas_handle();

    // Enable TF32 math mode
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);

    // Reset to default
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

// Cleanup function (call at program end)
inline void cleanup_cublas() {
    cublasHandle_t& handle = get_cublas_handle();
    if (handle != nullptr) {
        cublasDestroy(handle);
        handle = nullptr;
    }
}

}  // namespace kernels
}  // namespace cudabench
