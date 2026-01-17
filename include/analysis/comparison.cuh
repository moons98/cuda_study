#pragma once

#include "core/benchmark.cuh"
#include "core/device_info.cuh"
#include "kernels/kernel_registry.cuh"
#include <string>
#include <vector>

namespace cudabench {

struct ComparisonResult {
    std::vector<std::string> kernel_names;
    std::vector<std::vector<BenchmarkResult>> results;  // [kernel][size]

    std::string baseline_kernel;
    std::vector<std::vector<double>> speedup_vs_baseline;  // [kernel][size]

    // Rankings per size
    std::vector<std::vector<int>> ranking_by_gflops;  // [size][rank] -> kernel_index
};

class KernelComparator {
public:
    KernelComparator();
    ~KernelComparator();

    // Configuration
    void add_kernel(const KernelInfo& kernel);
    void add_kernel(const std::string& name);  // From global registry
    void add_all_kernels();  // Add all from global registry
    void set_baseline(const std::string& kernel_name);
    void clear_kernels();

    // Run comparison
    ComparisonResult run(const BenchmarkConfig& config);

    // Output
    void print_comparison_table(const ComparisonResult& result) const;
    void print_summary(const ComparisonResult& result) const;
    void export_csv(const ComparisonResult& result, const std::string& filepath) const;

private:
    std::vector<KernelInfo> kernels_;
    std::string baseline_name_;
    DeviceInfo device_;

    void calculate_speedups(ComparisonResult& result) const;
    void calculate_rankings(ComparisonResult& result) const;
};

}  // namespace cudabench
