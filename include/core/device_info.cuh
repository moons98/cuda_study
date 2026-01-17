#pragma once

#include <cuda_runtime.h>
#include <string>

namespace cudabench {

struct DeviceInfo {
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;

    // Memory
    size_t total_global_mem_bytes;
    size_t shared_mem_per_block;
    size_t shared_mem_per_sm;
    size_t total_const_mem;

    // Compute
    int sm_count;
    int max_threads_per_block;
    int max_threads_per_sm;
    int warp_size;
    int regs_per_block;
    int regs_per_sm;

    // Memory bandwidth
    int memory_bus_width;
    int memory_clock_mhz;

    // Clock
    int clock_rate_mhz;

    // Calculated theoretical peaks
    double peak_gflops_fp32;
    double peak_gflops_fp16;
    double peak_bandwidth_gbps;
    double ridge_point;  // peak_gflops / peak_bandwidth

    // Methods
    void query(int device_id = 0);
    void print() const;
    std::string to_csv_header() const;
    std::string to_csv_row() const;

    // Calculate theoretical peak performance
    static double calculate_peak_fp32_gflops(int sm_count, int clock_mhz, int cuda_cores_per_sm);
    static double calculate_peak_bandwidth_gbps(int memory_clock_mhz, int bus_width);

private:
    int get_cuda_cores_per_sm(int major, int minor) const;
};

// Global function to get current device info
DeviceInfo get_device_info(int device_id = 0);

// Print device info
void print_device_info(int device_id = 0);

}  // namespace cudabench
