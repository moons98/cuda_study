#include "core/benchmark.cuh"
#include "core/timer.cuh"
#include "core/device_info.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

namespace cudabench {

//=============================================================================
// BenchmarkResult methods
//=============================================================================

void BenchmarkResult::print() const {
    printf("  %4d x %4d x %4d | %8.4f ms | %8.2f GFLOPS | %6.2f%% | %s | %s\n",
           M, N, K,
           avg_time_ms,
           gflops,
           peak_gflops_percent,
           bound_type.c_str(),
           verified ? "PASS" : "FAIL");
}

std::string BenchmarkResult::csv_header() {
    return "kernel_name,M,N,K,avg_time_ms,min_time_ms,max_time_ms,std_dev_ms,"
           "gflops,bandwidth_gbps,peak_gflops_pct,peak_bw_pct,"
           "arithmetic_intensity,bound_type,verified,max_error";
}

std::string BenchmarkResult::to_csv_row() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << kernel_name << ","
        << M << "," << N << "," << K << ","
        << avg_time_ms << "," << min_time_ms << "," << max_time_ms << "," << std_dev_ms << ","
        << std::setprecision(2)
        << gflops << "," << achieved_bandwidth_gbps << ","
        << peak_gflops_percent << "," << peak_bandwidth_percent << ","
        << std::setprecision(4)
        << arithmetic_intensity << "," << bound_type << ","
        << (verified ? "true" : "false") << "," << max_error;
    return oss.str();
}

//=============================================================================
// Benchmark class implementation
//=============================================================================

Benchmark::Benchmark()
    : device_id_(0),
      cublas_handle_(nullptr),
      d_A_(nullptr), d_B_(nullptr), d_C_(nullptr), d_C_ref_(nullptr),
      h_A_(nullptr), h_B_(nullptr), h_C_(nullptr), h_C_ref_(nullptr),
      allocated_size_(0) {

    cudaSetDevice(device_id_);
    cublasCreate(&cublas_handle_);
}

Benchmark::~Benchmark() {
    free_memory();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

void Benchmark::set_device(int device_id) {
    device_id_ = device_id;
    cudaSetDevice(device_id_);

    // Recreate cuBLAS handle for new device
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    cublasCreate(&cublas_handle_);

    // Free and reallocate memory on new device
    if (allocated_size_ > 0) {
        size_t size = allocated_size_;
        free_memory();
        allocate_memory(size);
    }
}

void Benchmark::allocate_memory(size_t max_elements) {
    if (allocated_size_ >= max_elements) {
        return;
    }

    free_memory();

    size_t bytes = max_elements * sizeof(float);

    // Device memory
    cudaMalloc(&d_A_, bytes);
    cudaMalloc(&d_B_, bytes);
    cudaMalloc(&d_C_, bytes);
    cudaMalloc(&d_C_ref_, bytes);

    // Host memory
    h_A_ = (float*)malloc(bytes);
    h_B_ = (float*)malloc(bytes);
    h_C_ = (float*)malloc(bytes);
    h_C_ref_ = (float*)malloc(bytes);

    allocated_size_ = max_elements;
}

void Benchmark::free_memory() {
    if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
    if (d_B_) { cudaFree(d_B_); d_B_ = nullptr; }
    if (d_C_) { cudaFree(d_C_); d_C_ = nullptr; }
    if (d_C_ref_) { cudaFree(d_C_ref_); d_C_ref_ = nullptr; }

    if (h_A_) { free(h_A_); h_A_ = nullptr; }
    if (h_B_) { free(h_B_); h_B_ = nullptr; }
    if (h_C_) { free(h_C_); h_C_ = nullptr; }
    if (h_C_ref_) { free(h_C_ref_); h_C_ref_ = nullptr; }

    allocated_size_ = 0;
}

void Benchmark::init_matrices(int size) {
    // Use high-precision timer for seeding
#ifdef _WIN32
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    srand(static_cast<unsigned int>(counter.LowPart));
#else
    struct timeval time;
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
#endif

    size_t elements = static_cast<size_t>(size) * size;

    for (size_t i = 0; i < elements; i++) {
        float tmp = static_cast<float>(rand() % 5) + 0.01f * (rand() % 5);
        h_A_[i] = (rand() % 2 == 0) ? tmp : -tmp;

        tmp = static_cast<float>(rand() % 5) + 0.01f * (rand() % 5);
        h_B_[i] = (rand() % 2 == 0) ? tmp : -tmp;

        h_C_[i] = 0.0f;
    }

    // Copy to device
    size_t bytes = elements * sizeof(float);
    cudaMemcpy(d_A_, h_A_, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_, h_B_, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_, h_C_, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref_, h_C_, bytes, cudaMemcpyHostToDevice);
}

void Benchmark::run_cublas_reference(int M, int N, int K, float alpha, float beta) {
    // cuBLAS uses column-major, so we compute B^T * A^T = (A*B)^T
    cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B_, N,
                d_A_, K,
                &beta,
                d_C_ref_, N);
    cudaDeviceSynchronize();
}

bool Benchmark::verify_result(int size, float tolerance) {
    size_t elements = static_cast<size_t>(size) * size;
    size_t bytes = elements * sizeof(float);

    cudaMemcpy(h_C_, d_C_, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref_, d_C_ref_, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < elements; i++) {
        float diff = std::fabs(h_C_[i] - h_C_ref_[i]);
        if (std::isnan(diff) || diff > tolerance) {
            return false;
        }
    }
    return true;
}

void Benchmark::calculate_statistics(const std::vector<float>& times, BenchmarkResult& result) {
    if (times.empty()) return;

    // Average
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    result.avg_time_ms = sum / times.size();

    // Min/Max
    auto minmax = std::minmax_element(times.begin(), times.end());
    result.min_time_ms = *minmax.first;
    result.max_time_ms = *minmax.second;

    // Standard deviation
    double sq_sum = 0.0;
    for (float t : times) {
        double diff = t - result.avg_time_ms;
        sq_sum += diff * diff;
    }
    result.std_dev_ms = std::sqrt(sq_sum / times.size());
}

void Benchmark::calculate_performance_metrics(BenchmarkResult& result, const DeviceInfo& device) {
    // GEMM FLOPs: 2 * M * N * K (multiply-add)
    double flops = 2.0 * result.M * result.N * result.K;
    result.gflops = (flops / (result.avg_time_ms / 1000.0)) / 1e9;

    // Memory: A(M*K) + B(K*N) + C(M*N) reads, C(M*N) writes
    // Assuming no caching: bytes = (M*K + K*N + 2*M*N) * sizeof(float)
    double bytes = (static_cast<double>(result.M) * result.K +
                    static_cast<double>(result.K) * result.N +
                    2.0 * result.M * result.N) * sizeof(float);
    result.achieved_bandwidth_gbps = (bytes / (result.avg_time_ms / 1000.0)) / 1e9;

    // Peak percentages
    result.peak_gflops_percent = (result.gflops / device.peak_gflops_fp32) * 100.0;
    result.peak_bandwidth_percent = (result.achieved_bandwidth_gbps / device.peak_bandwidth_gbps) * 100.0;

    // Arithmetic intensity
    result.arithmetic_intensity = flops / bytes;

    // Determine if memory or compute bound
    if (result.arithmetic_intensity < device.ridge_point) {
        result.bound_type = "memory";
    } else {
        result.bound_type = "compute";
    }
}

BenchmarkResult Benchmark::run(const KernelInfo& kernel, int size, const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.kernel_name = kernel.name;
    result.M = result.N = result.K = size;
    result.verified = false;
    result.max_error = 0.0;

    // Allocate memory for this size
    size_t elements = static_cast<size_t>(size) * size;
    allocate_memory(elements);

    // Initialize matrices
    init_matrices(size);

    // Run cuBLAS reference first (if verification enabled)
    if (config.verify) {
        run_cublas_reference(size, size, size, config.alpha, config.beta);
    }

    // Warmup runs
    for (int i = 0; i < config.warmup_runs; i++) {
        // Reset C for each run
        cudaMemcpy(d_C_, h_C_, elements * sizeof(float), cudaMemcpyHostToDevice);
        kernel.run(size, size, size, config.alpha, d_A_, d_B_, config.beta, d_C_);
        cudaDeviceSynchronize();
    }

    // Check for CUDA errors after warmup
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after warmup: %s\n", cudaGetErrorString(err));
        result.avg_time_ms = -1;
        return result;
    }

    // Verification
    if (config.verify) {
        cudaMemcpy(d_C_, h_C_, elements * sizeof(float), cudaMemcpyHostToDevice);
        kernel.run(size, size, size, config.alpha, d_A_, d_B_, config.beta, d_C_);
        cudaDeviceSynchronize();
        result.verified = verify_result(size, kernel.tolerance);
    } else {
        result.verified = true;
    }

    // Benchmark runs
    std::vector<float> times;
    times.reserve(config.benchmark_runs);

    CudaTimer timer;

    for (int i = 0; i < config.benchmark_runs; i++) {
        // Note: We don't reset C between benchmark runs for speed
        // This is acceptable as we're measuring computation time, not correctness

        timer.start();
        kernel.run(size, size, size, config.alpha, d_A_, d_B_, config.beta, d_C_);
        timer.stop();

        times.push_back(timer.elapsed_ms());
    }

    // Calculate statistics
    calculate_statistics(times, result);

    // Get device info and calculate performance metrics
    DeviceInfo device = get_device_info(device_id_);
    calculate_performance_metrics(result, device);

    return result;
}

std::vector<BenchmarkResult> Benchmark::run(const KernelInfo& kernel, const BenchmarkConfig& config) {
    std::vector<BenchmarkResult> results;
    results.reserve(config.sizes.size());

    if (config.print_progress) {
        printf("--------------------------------------------------------------------------------\n");
        printf("Kernel: %s\n", kernel.name.c_str());
        printf("Description: %s\n", kernel.description.c_str());
        printf("Warmup: %d runs | Benchmark: %d runs | Verification: %s\n",
               config.warmup_runs, config.benchmark_runs, config.verify ? "ON" : "OFF");
        printf("--------------------------------------------------------------------------------\n");
        printf("    Size      |  Time (ms)  |   GFLOPS   | %%Peak  |  Bound  | Status\n");
        printf("--------------|-------------|------------|--------|---------|--------\n");
    }

    for (int size : config.sizes) {
        BenchmarkResult result = run(kernel, size, config);
        results.push_back(result);

        if (config.print_progress) {
            result.print();
        }
    }

    if (config.print_progress) {
        printf("--------------------------------------------------------------------------------\n");
    }

    return results;
}

void Benchmark::print_results(const std::vector<BenchmarkResult>& results) const {
    printf("--------------------------------------------------------------------------------\n");
    printf("    Size      |  Time (ms)  |   GFLOPS   | %%Peak  |  Bound  | Status\n");
    printf("--------------|-------------|------------|--------|---------|--------\n");
    for (const auto& result : results) {
        result.print();
    }
    printf("--------------------------------------------------------------------------------\n");
}

}  // namespace cudabench
