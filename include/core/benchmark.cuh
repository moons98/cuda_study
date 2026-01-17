#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <functional>

// Forward declaration
namespace cudabench {
struct DeviceInfo;
}

namespace cudabench {

//=============================================================================
// Configuration
//=============================================================================

struct BenchmarkConfig {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    int warmup_runs = 5;
    int benchmark_runs = 50;
    bool verify = true;
    bool print_progress = true;
    float alpha = 1.0f;
    float beta = 0.0f;
};

//=============================================================================
// Result structures
//=============================================================================

struct BenchmarkResult {
    std::string kernel_name;
    int M, N, K;

    // Time statistics (in milliseconds)
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;

    // Performance metrics
    double gflops;
    double achieved_bandwidth_gbps;
    double peak_gflops_percent;
    double peak_bandwidth_percent;

    // Roofline analysis
    double arithmetic_intensity;  // FLOPs / Bytes
    std::string bound_type;       // "compute" or "memory"

    // Verification
    bool verified;
    double max_error;

    // Helper methods
    void print() const;
    std::string to_csv_row() const;
    static std::string csv_header();
};

//=============================================================================
// Kernel function signature
//=============================================================================

// Standard GEMM kernel signature: C = alpha * A * B + beta * C
using GemmKernelFunc = std::function<void(int M, int N, int K,
                                          float alpha, float* A, float* B,
                                          float beta, float* C)>;

//=============================================================================
// Kernel information
//=============================================================================

struct KernelInfo {
    std::string name;
    std::string description;
    GemmKernelFunc run;

    // Verification tolerance (default: 0.01f for FP32, use higher for TF32/lower precision)
    float tolerance = 0.01f;

    // Operational profile for roofline analysis
    struct OperationalProfile {
        bool uses_shared_memory = false;
        size_t shared_mem_bytes = 0;
        int tile_size_m = 0;
        int tile_size_n = 0;
        int tile_size_k = 0;
    } op_profile;
};

//=============================================================================
// Benchmark class
//=============================================================================

class Benchmark {
public:
    Benchmark();
    ~Benchmark();

    // Set device
    void set_device(int device_id);

    // Run single kernel benchmark
    BenchmarkResult run(const KernelInfo& kernel, int size, const BenchmarkConfig& config);
    std::vector<BenchmarkResult> run(const KernelInfo& kernel, const BenchmarkConfig& config);

    // Print results
    void print_results(const std::vector<BenchmarkResult>& results) const;

    // Get cuBLAS handle (for reference comparison)
    cublasHandle_t get_cublas_handle() const { return cublas_handle_; }

private:
    int device_id_;
    cublasHandle_t cublas_handle_;

    // Device memory (pre-allocated for max size)
    float* d_A_;
    float* d_B_;
    float* d_C_;
    float* d_C_ref_;
    size_t allocated_size_;

    // Host memory
    float* h_A_;
    float* h_B_;
    float* h_C_;
    float* h_C_ref_;

    // Internal methods
    void allocate_memory(size_t max_elements);
    void free_memory();
    void init_matrices(int size);
    bool verify_result(int size, float tolerance = 0.01f);

    // cuBLAS reference
    void run_cublas_reference(int M, int N, int K, float alpha, float beta);

    // Statistics calculation
    void calculate_statistics(const std::vector<float>& times, BenchmarkResult& result);
    void calculate_performance_metrics(BenchmarkResult& result, const DeviceInfo& device);
};

}  // namespace cudabench
