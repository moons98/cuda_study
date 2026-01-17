#include "kernels/kernel_registry.cuh"
#include "sgemm/naive.cuh"
#include "sgemm/coalesced.cuh"
#include "sgemm/shared_mem.cuh"
#include "sgemm/tiled_1d.cuh"
#include "sgemm/tiled_2d.cuh"
#include "sgemm/vectorized.cuh"
#include "sgemm/cublas_wrapper.cuh"
#include <cstdio>

namespace cudabench {

KernelRegistry::KernelRegistry() {}

void KernelRegistry::register_kernel(const KernelInfo& kernel) {
    if (kernels_.find(kernel.name) == kernels_.end()) {
        order_.push_back(kernel.name);
    }
    kernels_[kernel.name] = kernel;
}

void KernelRegistry::register_builtin_sgemm_kernels() {
    // cuBLAS reference
    {
        KernelInfo info;
        info.name = "cublas";
        info.description = "cuBLAS SGEMM (reference baseline)";
        info.run = kernels::run_cublas_sgemm;
        info.op_profile.uses_shared_memory = true;
        register_kernel(info);
    }

    // cuBLAS TF32
    {
        KernelInfo info;
        info.name = "cublas_tf32";
        info.description = "cuBLAS SGEMM with TF32 (Ampere+)";
        info.run = kernels::run_cublas_sgemm_tf32;
        info.tolerance = 1.0f;  // TF32 has lower precision (19-bit mantissa vs 23-bit FP32)
        info.op_profile.uses_shared_memory = true;
        register_kernel(info);
    }

    // Naive
    {
        KernelInfo info;
        info.name = "naive";
        info.description = "Naive SGEMM - each thread computes one element";
        info.run = kernels::run_sgemm_naive;
        info.op_profile.uses_shared_memory = false;
        register_kernel(info);
    }

    // Coalesced
    {
        KernelInfo info;
        info.name = "coalesced";
        info.description = "Global memory coalesced access pattern";
        info.run = kernels::run_sgemm_coalesced;
        info.op_profile.uses_shared_memory = false;
        register_kernel(info);
    }

    // Shared memory
    {
        KernelInfo info;
        info.name = "shared_mem";
        info.description = "Shared memory tiled SGEMM (32x32 tiles)";
        info.run = kernels::run_sgemm_shared_mem;
        info.op_profile.uses_shared_memory = true;
        info.op_profile.tile_size_m = 32;
        info.op_profile.tile_size_n = 32;
        info.op_profile.tile_size_k = 32;
        info.op_profile.shared_mem_bytes = 2 * 32 * 32 * sizeof(float);
        register_kernel(info);
    }

    // 1D Tiling
    {
        KernelInfo info;
        info.name = "tiled_1d";
        info.description = "1D block tiling - thread computes TM elements";
        info.run = kernels::run_sgemm_tiled_1d;
        info.op_profile.uses_shared_memory = true;
        info.op_profile.tile_size_m = 64;
        info.op_profile.tile_size_n = 64;
        info.op_profile.tile_size_k = 8;
        info.op_profile.shared_mem_bytes = (64 * 8 + 8 * 64) * sizeof(float);
        register_kernel(info);
    }

    // 2D Tiling
    {
        KernelInfo info;
        info.name = "tiled_2d";
        info.description = "2D block tiling - thread computes TMxTN tile";
        info.run = kernels::run_sgemm_tiled_2d;
        info.op_profile.uses_shared_memory = true;
        info.op_profile.tile_size_m = 128;
        info.op_profile.tile_size_n = 128;
        info.op_profile.tile_size_k = 8;
        info.op_profile.shared_mem_bytes = (128 * 8 + 8 * 128) * sizeof(float);
        register_kernel(info);
    }

    // Vectorized
    {
        KernelInfo info;
        info.name = "vectorized";
        info.description = "Vectorized loads using float4";
        info.run = kernels::run_sgemm_vectorized;
        info.op_profile.uses_shared_memory = true;
        info.op_profile.tile_size_m = 128;
        info.op_profile.tile_size_n = 128;
        info.op_profile.tile_size_k = 8;
        info.op_profile.shared_mem_bytes = (128 * 8 + 8 * 128) * sizeof(float);
        register_kernel(info);
    }
}

const KernelInfo* KernelRegistry::get(const std::string& name) const {
    auto it = kernels_.find(name);
    if (it != kernels_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<KernelInfo> KernelRegistry::get_all() const {
    std::vector<KernelInfo> result;
    result.reserve(order_.size());
    for (const auto& name : order_) {
        result.push_back(kernels_.at(name));
    }
    return result;
}

std::vector<std::string> KernelRegistry::get_names() const {
    return order_;
}

bool KernelRegistry::exists(const std::string& name) const {
    return kernels_.find(name) != kernels_.end();
}

void KernelRegistry::print_available() const {
    printf("================================================================================\n");
    printf("AVAILABLE KERNELS\n");
    printf("================================================================================\n");
    printf("  %-15s | %s\n", "Name", "Description");
    printf("----------------|---------------------------------------------------------------\n");
    for (const auto& name : order_) {
        const auto& kernel = kernels_.at(name);
        printf("  %-15s | %s\n", kernel.name.c_str(), kernel.description.c_str());
    }
    printf("================================================================================\n");
}

// Global registry
KernelRegistry& get_global_registry() {
    static KernelRegistry registry;
    static bool initialized = false;
    if (!initialized) {
        registry.register_builtin_sgemm_kernels();
        initialized = true;
    }
    return registry;
}

}  // namespace cudabench
