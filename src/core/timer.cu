#include "core/timer.cuh"
#include <stdexcept>

namespace cudabench {

//=============================================================================
// CudaTimer Implementation
//=============================================================================

CudaTimer::CudaTimer() : elapsed_ms_(0.0f) {
    cudaError_t err;
    err = cudaEventCreate(&start_event_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA start event");
    }
    err = cudaEventCreate(&stop_event_);
    if (err != cudaSuccess) {
        cudaEventDestroy(start_event_);
        throw std::runtime_error("Failed to create CUDA stop event");
    }
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void CudaTimer::start() {
    cudaEventRecord(start_event_);
}

void CudaTimer::stop() {
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&elapsed_ms_, start_event_, stop_event_);
}

float CudaTimer::elapsed_ms() const {
    return elapsed_ms_;
}

//=============================================================================
// CpuTimer Implementation
//=============================================================================

CpuTimer::CpuTimer() : elapsed_ms_(0.0) {
#ifdef _WIN32
    QueryPerformanceFrequency(&frequency_);
#endif
}

void CpuTimer::start() {
#ifdef _WIN32
    QueryPerformanceCounter(&start_time_);
#else
    gettimeofday(&start_time_, nullptr);
#endif
}

void CpuTimer::stop() {
#ifdef _WIN32
    QueryPerformanceCounter(&stop_time_);
    elapsed_ms_ = static_cast<double>(stop_time_.QuadPart - start_time_.QuadPart)
                  * 1000.0 / static_cast<double>(frequency_.QuadPart);
#else
    gettimeofday(&stop_time_, nullptr);
    elapsed_ms_ = (stop_time_.tv_sec - start_time_.tv_sec) * 1000.0 +
                  (stop_time_.tv_usec - start_time_.tv_usec) / 1000.0;
#endif
}

double CpuTimer::elapsed_ms() const {
    return elapsed_ms_;
}

double CpuTimer::elapsed_sec() const {
    return elapsed_ms_ / 1000.0;
}

}  // namespace cudabench
