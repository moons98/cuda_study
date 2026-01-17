#pragma once

#include "core/benchmark.cuh"
#include <string>
#include <vector>

namespace cudabench {

class CsvExporter {
public:
    // Export benchmark results to CSV
    static bool export_results(const std::vector<BenchmarkResult>& results,
                               const std::string& filepath);

    // Export comparison results (multiple kernels)
    static bool export_comparison(const std::vector<std::vector<BenchmarkResult>>& all_results,
                                  const std::string& filepath);

    // Export roofline data
    static bool export_roofline(const std::vector<BenchmarkResult>& results,
                                double peak_gflops, double peak_bandwidth,
                                const std::string& filepath);

    // Append results to existing file
    static bool append_results(const std::vector<BenchmarkResult>& results,
                               const std::string& filepath);
};

}  // namespace cudabench
