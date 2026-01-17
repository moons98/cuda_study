#include "core/device_info.cuh"
#include <cstdio>
#include <stdexcept>
#include <sstream>

namespace cudabench {

int DeviceInfo::get_cuda_cores_per_sm(int major, int minor) const {
    // CUDA cores per SM for different architectures
    switch (major) {
        case 2:  // Fermi
            return (minor == 1) ? 48 : 32;
        case 3:  // Kepler
            return 192;
        case 5:  // Maxwell
            return 128;
        case 6:  // Pascal
            return (minor == 0) ? 64 : 128;
        case 7:  // Volta/Turing
            return (minor == 0) ? 64 : 64;
        case 8:  // Ampere
            return (minor == 0) ? 64 : 128;
        case 9:  // Hopper
            return 128;
        default:
            return 128;  // Default assumption
    }
}

double DeviceInfo::calculate_peak_fp32_gflops(int sm_count, int clock_mhz, int cuda_cores_per_sm) {
    // FP32 GFLOPS = SM count * cores per SM * 2 (FMA) * clock (GHz)
    return static_cast<double>(sm_count) * cuda_cores_per_sm * 2.0 * clock_mhz / 1000.0;
}

double DeviceInfo::calculate_peak_bandwidth_gbps(int memory_clock_mhz, int bus_width) {
    // Bandwidth = memory_clock * bus_width * 2 (DDR) / 8 (bits to bytes)
    return static_cast<double>(memory_clock_mhz) * bus_width * 2.0 / 8.0 / 1000.0;
}

void DeviceInfo::query(int dev_id) {
    device_id = dev_id;

    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties");
    }

    name = props.name;
    compute_capability_major = props.major;
    compute_capability_minor = props.minor;

    // Memory
    total_global_mem_bytes = props.totalGlobalMem;
    shared_mem_per_block = props.sharedMemPerBlock;
    shared_mem_per_sm = props.sharedMemPerMultiprocessor;
    total_const_mem = props.totalConstMem;

    // Compute
    sm_count = props.multiProcessorCount;
    max_threads_per_block = props.maxThreadsPerBlock;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    warp_size = props.warpSize;
    regs_per_block = props.regsPerBlock;
    regs_per_sm = props.regsPerMultiprocessor;

    // Memory bandwidth
    memory_bus_width = props.memoryBusWidth;
    memory_clock_mhz = props.memoryClockRate / 1000;  // Convert kHz to MHz

    // Clock
    clock_rate_mhz = props.clockRate / 1000;  // Convert kHz to MHz

    // Calculate theoretical peaks
    int cuda_cores = get_cuda_cores_per_sm(compute_capability_major, compute_capability_minor);
    peak_gflops_fp32 = calculate_peak_fp32_gflops(sm_count, clock_rate_mhz, cuda_cores);
    peak_gflops_fp16 = peak_gflops_fp32 * 2.0;  // FP16 typically 2x FP32
    peak_bandwidth_gbps = calculate_peak_bandwidth_gbps(memory_clock_mhz, memory_bus_width);
    ridge_point = peak_gflops_fp32 / peak_bandwidth_gbps;
}

void DeviceInfo::print() const {
    printf("================================================================================\n");
    printf("CUDA DEVICE INFORMATION\n");
    printf("================================================================================\n");
    printf("Device ID:                %d\n", device_id);
    printf("Name:                     %s\n", name.c_str());
    printf("Compute Capability:       %d.%d\n", compute_capability_major, compute_capability_minor);
    printf("--------------------------------------------------------------------------------\n");
    printf("Memory:\n");
    printf("  Total Global Memory:    %zu MB\n", total_global_mem_bytes / (1024 * 1024));
    printf("  Shared Mem per Block:   %zu KB\n", shared_mem_per_block / 1024);
    printf("  Shared Mem per SM:      %zu KB\n", shared_mem_per_sm / 1024);
    printf("  Constant Memory:        %zu KB\n", total_const_mem / 1024);
    printf("  Memory Bus Width:       %d bit\n", memory_bus_width);
    printf("  Memory Clock:           %d MHz\n", memory_clock_mhz);
    printf("--------------------------------------------------------------------------------\n");
    printf("Compute:\n");
    printf("  SM Count:               %d\n", sm_count);
    printf("  Max Threads/Block:      %d\n", max_threads_per_block);
    printf("  Max Threads/SM:         %d\n", max_threads_per_sm);
    printf("  Warp Size:              %d\n", warp_size);
    printf("  Registers/Block:        %d\n", regs_per_block);
    printf("  Registers/SM:           %d\n", regs_per_sm);
    printf("  GPU Clock:              %d MHz\n", clock_rate_mhz);
    printf("--------------------------------------------------------------------------------\n");
    printf("Theoretical Peaks:\n");
    printf("  Peak FP32 Performance:  %.1f GFLOPS\n", peak_gflops_fp32);
    printf("  Peak FP16 Performance:  %.1f GFLOPS\n", peak_gflops_fp16);
    printf("  Peak Memory Bandwidth:  %.1f GB/s\n", peak_bandwidth_gbps);
    printf("  Ridge Point (AI):       %.2f FLOPs/Byte\n", ridge_point);
    printf("================================================================================\n");
}

std::string DeviceInfo::to_csv_header() const {
    return "device_id,name,compute_cap,sm_count,global_mem_mb,peak_gflops,peak_bw_gbps,ridge_point";
}

std::string DeviceInfo::to_csv_row() const {
    std::ostringstream oss;
    oss << device_id << ","
        << name << ","
        << compute_capability_major << "." << compute_capability_minor << ","
        << sm_count << ","
        << total_global_mem_bytes / (1024 * 1024) << ","
        << peak_gflops_fp32 << ","
        << peak_bandwidth_gbps << ","
        << ridge_point;
    return oss.str();
}

DeviceInfo get_device_info(int device_id) {
    DeviceInfo info;
    info.query(device_id);
    return info;
}

void print_device_info(int device_id) {
    DeviceInfo info = get_device_info(device_id);
    info.print();
}

}  // namespace cudabench
