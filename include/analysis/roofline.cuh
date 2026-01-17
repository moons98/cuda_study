#pragma once

#include "core/benchmark.cuh"
#include "core/device_info.cuh"
#include <string>
#include <vector>

namespace cudabench {

struct RooflineDataPoint {
    std::string kernel_name;
    int size;
    double arithmetic_intensity;  // FLOPs / Byte
    double achieved_gflops;
    double efficiency;            // vs roofline ceiling (%)
    std::string bound_type;       // "memory" or "compute"
    std::string recommendation;   // Optimization suggestion
};

struct RooflineData {
    // Hardware limits
    double peak_gflops;
    double peak_bandwidth_gbps;
    double ridge_point;

    // Data points
    std::vector<RooflineDataPoint> points;
};

class RooflineAnalyzer {
public:
    RooflineAnalyzer();
    explicit RooflineAnalyzer(const DeviceInfo& device);

    // Set hardware parameters manually (for overclocked/custom configs)
    void set_peak_gflops(double gflops);
    void set_peak_bandwidth(double gbps);

    // Add data points
    void add_result(const BenchmarkResult& result);
    void add_results(const std::vector<BenchmarkResult>& results);
    void clear();

    // Get analysis data
    RooflineData get_data() const;

    // Output
    void print_report() const;
    void export_csv(const std::string& filepath) const;
    void generate_gnuplot_script(const std::string& filepath) const;
    void generate_python_script(const std::string& filepath) const;

private:
    double peak_gflops_;
    double peak_bandwidth_gbps_;
    double ridge_point_;

    std::vector<RooflineDataPoint> points_;

    double calculate_efficiency(double ai, double gflops) const;
    std::string determine_bound(double ai) const;
    std::string generate_recommendation(const RooflineDataPoint& point) const;
};

}  // namespace cudabench
