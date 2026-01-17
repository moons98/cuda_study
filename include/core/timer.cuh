#pragma once

#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

namespace cudabench {

// CUDA Event-based timer (most accurate for GPU operations)
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();

    void start();
    void stop();
    float elapsed_ms() const;  // Returns elapsed time in milliseconds

    // Disable copy
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    float elapsed_ms_;
};

// CPU timer (for host-side timing)
class CpuTimer {
public:
    CpuTimer();

    void start();
    void stop();
    double elapsed_ms() const;
    double elapsed_sec() const;

private:
#ifdef _WIN32
    LARGE_INTEGER frequency_;
    LARGE_INTEGER start_time_;
    LARGE_INTEGER stop_time_;
#else
    struct timeval start_time_;
    struct timeval stop_time_;
#endif
    double elapsed_ms_;
};

}  // namespace cudabench
