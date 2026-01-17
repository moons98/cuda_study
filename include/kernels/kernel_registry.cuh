#pragma once

#include "core/benchmark.cuh"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cudabench {

class KernelRegistry {
public:
    KernelRegistry();
    ~KernelRegistry() = default;

    // Register a kernel
    void register_kernel(const KernelInfo& kernel);

    // Register all builtin SGEMM kernels
    void register_builtin_sgemm_kernels();

    // Get kernel by name
    const KernelInfo* get(const std::string& name) const;

    // Get all kernels
    std::vector<KernelInfo> get_all() const;

    // Get kernel names
    std::vector<std::string> get_names() const;

    // Check if kernel exists
    bool exists(const std::string& name) const;

    // Get kernel count
    size_t count() const { return kernels_.size(); }

    // Print available kernels
    void print_available() const;

private:
    std::unordered_map<std::string, KernelInfo> kernels_;
    std::vector<std::string> order_;  // Maintain registration order
};

// Global registry instance
KernelRegistry& get_global_registry();

}  // namespace cudabench
