#include "analysis/comparison.cuh"
#include "utils/csv_export.cuh"
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cudabench {

KernelComparator::KernelComparator() : baseline_name_("cublas") {
    device_.query(0);
}

KernelComparator::~KernelComparator() {}

void KernelComparator::add_kernel(const KernelInfo& kernel) {
    kernels_.push_back(kernel);
}

void KernelComparator::add_kernel(const std::string& name) {
    const KernelInfo* kernel = get_global_registry().get(name);
    if (kernel) {
        kernels_.push_back(*kernel);
    } else {
        fprintf(stderr, "Warning: Kernel '%s' not found in registry\n", name.c_str());
    }
}

void KernelComparator::add_all_kernels() {
    kernels_ = get_global_registry().get_all();
}

void KernelComparator::set_baseline(const std::string& kernel_name) {
    baseline_name_ = kernel_name;
}

void KernelComparator::clear_kernels() {
    kernels_.clear();
}

ComparisonResult KernelComparator::run(const BenchmarkConfig& config) {
    ComparisonResult result;

    if (kernels_.empty()) {
        fprintf(stderr, "No kernels to compare\n");
        return result;
    }

    // Store kernel names
    for (const auto& kernel : kernels_) {
        result.kernel_names.push_back(kernel.name);
    }
    result.baseline_kernel = baseline_name_;

    printf("================================================================================\n");
    printf("KERNEL COMPARISON\n");
    printf("================================================================================\n");
    printf("Device: %s | Peak: %.1f GFLOPS | BW: %.1f GB/s\n",
           device_.name.c_str(), device_.peak_gflops_fp32, device_.peak_bandwidth_gbps);
    printf("Baseline: %s | Sizes: ", baseline_name_.c_str());
    for (size_t i = 0; i < config.sizes.size(); i++) {
        printf("%d%s", config.sizes[i], (i < config.sizes.size() - 1) ? ", " : "\n");
    }
    printf("Warmup: %d | Runs: %d | Verification: %s\n",
           config.warmup_runs, config.benchmark_runs, config.verify ? "ON" : "OFF");
    printf("================================================================================\n\n");

    // Run benchmarks for each kernel
    Benchmark bench;
    result.results.resize(kernels_.size());

    for (size_t k = 0; k < kernels_.size(); k++) {
        BenchmarkConfig silent_config = config;
        silent_config.print_progress = false;

        printf("Benchmarking: %s... ", kernels_[k].name.c_str());
        fflush(stdout);

        result.results[k] = bench.run(kernels_[k], silent_config);

        printf("done\n");
    }

    printf("\n");

    // Calculate speedups and rankings
    calculate_speedups(result);
    calculate_rankings(result);

    return result;
}

void KernelComparator::calculate_speedups(ComparisonResult& result) const {
    // Find baseline index
    int baseline_idx = -1;
    for (size_t i = 0; i < result.kernel_names.size(); i++) {
        if (result.kernel_names[i] == result.baseline_kernel) {
            baseline_idx = static_cast<int>(i);
            break;
        }
    }

    result.speedup_vs_baseline.resize(result.results.size());

    for (size_t k = 0; k < result.results.size(); k++) {
        result.speedup_vs_baseline[k].resize(result.results[k].size());
        for (size_t s = 0; s < result.results[k].size(); s++) {
            if (baseline_idx >= 0 && result.results[baseline_idx][s].avg_time_ms > 0) {
                double baseline_time = result.results[baseline_idx][s].avg_time_ms;
                double kernel_time = result.results[k][s].avg_time_ms;
                result.speedup_vs_baseline[k][s] = baseline_time / kernel_time;
            } else {
                result.speedup_vs_baseline[k][s] = 1.0;
            }
        }
    }
}

void KernelComparator::calculate_rankings(ComparisonResult& result) const {
    if (result.results.empty() || result.results[0].empty()) return;

    size_t num_sizes = result.results[0].size();
    result.ranking_by_gflops.resize(num_sizes);

    for (size_t s = 0; s < num_sizes; s++) {
        // Create index-gflops pairs
        std::vector<std::pair<int, double>> kernel_gflops;
        for (size_t k = 0; k < result.results.size(); k++) {
            kernel_gflops.push_back({static_cast<int>(k), result.results[k][s].gflops});
        }

        // Sort by GFLOPS descending
        std::sort(kernel_gflops.begin(), kernel_gflops.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Store ranking
        result.ranking_by_gflops[s].resize(kernel_gflops.size());
        for (size_t r = 0; r < kernel_gflops.size(); r++) {
            result.ranking_by_gflops[s][r] = kernel_gflops[r].first;
        }
    }
}

void KernelComparator::print_comparison_table(const ComparisonResult& result) const {
    if (result.results.empty()) return;

    for (size_t s = 0; s < result.results[0].size(); s++) {
        int size = result.results[0][s].M;

        printf("--------------------------------------------------------------------------------\n");
        printf("[Size: %d x %d]\n", size, size);
        printf("--------------------------------------------------------------------------------\n");
        printf(" Rank | %-20s | Time (ms) |  GFLOPS  | %%Peak | Speedup | Status\n", "Kernel");
        printf("------|----------------------|-----------|----------|-------|---------|--------\n");

        for (size_t r = 0; r < result.ranking_by_gflops[s].size(); r++) {
            int k = result.ranking_by_gflops[s][r];
            const auto& res = result.results[k][s];
            const auto& name = result.kernel_names[k];

            std::string display_name = name;
            if (name == result.baseline_kernel) {
                display_name += " (baseline)";
            }

            printf(" %3zu  | %-20s | %9.4f | %8.1f | %5.1f%% | %6.2fx | %s\n",
                   r + 1,
                   display_name.c_str(),
                   res.avg_time_ms,
                   res.gflops,
                   res.peak_gflops_percent,
                   result.speedup_vs_baseline[k][s],
                   res.verified ? "PASS" : "FAIL");
        }
        printf("\n");
    }
}

void KernelComparator::print_summary(const ComparisonResult& result) const {
    if (result.results.empty()) return;

    printf("================================================================================\n");
    printf("SUMMARY - Best Kernels by Size\n");
    printf("================================================================================\n");
    printf("   Size   | Best Kernel          | GFLOPS   | vs Baseline\n");
    printf("----------|----------------------|----------|-------------\n");

    for (size_t s = 0; s < result.results[0].size(); s++) {
        // Find best non-baseline kernel
        int best_idx = -1;
        double best_gflops = 0;

        for (size_t k = 0; k < result.results.size(); k++) {
            if (result.kernel_names[k] != result.baseline_kernel) {
                if (result.results[k][s].gflops > best_gflops) {
                    best_gflops = result.results[k][s].gflops;
                    best_idx = static_cast<int>(k);
                }
            }
        }

        if (best_idx >= 0) {
            int size = result.results[0][s].M;
            printf(" %7d  | %-20s | %8.1f | %6.1f%%\n",
                   size,
                   result.kernel_names[best_idx].c_str(),
                   best_gflops,
                   result.speedup_vs_baseline[best_idx][s] * 100.0);
        }
    }
    printf("================================================================================\n");
}

void KernelComparator::export_csv(const ComparisonResult& result, const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filepath.c_str());
        return;
    }

    // Header
    file << "kernel_name,size,avg_time_ms,gflops,peak_pct,speedup,verified\n";

    // Data
    for (size_t k = 0; k < result.results.size(); k++) {
        for (size_t s = 0; s < result.results[k].size(); s++) {
            const auto& res = result.results[k][s];
            file << result.kernel_names[k] << ","
                 << res.M << ","
                 << std::fixed << std::setprecision(6) << res.avg_time_ms << ","
                 << std::setprecision(2) << res.gflops << ","
                 << res.peak_gflops_percent << ","
                 << result.speedup_vs_baseline[k][s] << ","
                 << (res.verified ? "true" : "false") << "\n";
        }
    }

    file.close();
    printf("Comparison exported to: %s\n", filepath.c_str());
}

}  // namespace cudabench
