#include "utils/csv_export.cuh"
#include <fstream>
#include <cstdio>
#include <sys/stat.h>

namespace cudabench {

static bool file_exists(const std::string& filepath) {
    struct stat buffer;
    return (stat(filepath.c_str(), &buffer) == 0);
}

bool CsvExporter::export_results(const std::vector<BenchmarkResult>& results,
                                 const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath.c_str());
        return false;
    }

    // Write header
    file << BenchmarkResult::csv_header() << "\n";

    // Write data
    for (const auto& result : results) {
        file << result.to_csv_row() << "\n";
    }

    file.close();
    printf("Results exported to: %s\n", filepath.c_str());
    return true;
}

bool CsvExporter::export_comparison(const std::vector<std::vector<BenchmarkResult>>& all_results,
                                    const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath.c_str());
        return false;
    }

    // Write header
    file << BenchmarkResult::csv_header() << "\n";

    // Write data for all kernels
    for (const auto& kernel_results : all_results) {
        for (const auto& result : kernel_results) {
            file << result.to_csv_row() << "\n";
        }
    }

    file.close();
    printf("Comparison results exported to: %s\n", filepath.c_str());
    return true;
}

bool CsvExporter::export_roofline(const std::vector<BenchmarkResult>& results,
                                  double peak_gflops, double peak_bandwidth,
                                  const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath.c_str());
        return false;
    }

    // Write header with metadata
    file << "# Roofline Data\n";
    file << "# Peak GFLOPS: " << peak_gflops << "\n";
    file << "# Peak Bandwidth (GB/s): " << peak_bandwidth << "\n";
    file << "# Ridge Point: " << (peak_gflops / peak_bandwidth) << "\n";
    file << "kernel_name,arithmetic_intensity,gflops,efficiency_pct,bound_type\n";

    // Write data points
    double ridge_point = peak_gflops / peak_bandwidth;
    for (const auto& result : results) {
        double efficiency;
        if (result.arithmetic_intensity < ridge_point) {
            // Memory bound - compare against bandwidth ceiling
            double max_achievable = result.arithmetic_intensity * peak_bandwidth;
            efficiency = (result.gflops / max_achievable) * 100.0;
        } else {
            // Compute bound - compare against compute ceiling
            efficiency = (result.gflops / peak_gflops) * 100.0;
        }

        file << result.kernel_name << ","
             << result.arithmetic_intensity << ","
             << result.gflops << ","
             << efficiency << ","
             << result.bound_type << "\n";
    }

    file.close();
    printf("Roofline data exported to: %s\n", filepath.c_str());
    return true;
}

bool CsvExporter::append_results(const std::vector<BenchmarkResult>& results,
                                 const std::string& filepath) {
    bool needs_header = !file_exists(filepath);

    std::ofstream file(filepath, std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for appending: %s\n", filepath.c_str());
        return false;
    }

    if (needs_header) {
        file << BenchmarkResult::csv_header() << "\n";
    }

    for (const auto& result : results) {
        file << result.to_csv_row() << "\n";
    }

    file.close();
    return true;
}

}  // namespace cudabench
