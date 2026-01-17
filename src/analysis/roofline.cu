#include "analysis/roofline.cuh"
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace cudabench {

RooflineAnalyzer::RooflineAnalyzer() {
    DeviceInfo device;
    device.query(0);
    peak_gflops_ = device.peak_gflops_fp32;
    peak_bandwidth_gbps_ = device.peak_bandwidth_gbps;
    ridge_point_ = peak_gflops_ / peak_bandwidth_gbps_;
}

RooflineAnalyzer::RooflineAnalyzer(const DeviceInfo& device) {
    peak_gflops_ = device.peak_gflops_fp32;
    peak_bandwidth_gbps_ = device.peak_bandwidth_gbps;
    ridge_point_ = peak_gflops_ / peak_bandwidth_gbps_;
}

void RooflineAnalyzer::set_peak_gflops(double gflops) {
    peak_gflops_ = gflops;
    ridge_point_ = peak_gflops_ / peak_bandwidth_gbps_;
}

void RooflineAnalyzer::set_peak_bandwidth(double gbps) {
    peak_bandwidth_gbps_ = gbps;
    ridge_point_ = peak_gflops_ / peak_bandwidth_gbps_;
}

double RooflineAnalyzer::calculate_efficiency(double ai, double gflops) const {
    double ceiling;
    if (ai < ridge_point_) {
        // Memory bound - ceiling is bandwidth * AI
        ceiling = peak_bandwidth_gbps_ * ai;
    } else {
        // Compute bound - ceiling is peak GFLOPS
        ceiling = peak_gflops_;
    }
    return (gflops / ceiling) * 100.0;
}

std::string RooflineAnalyzer::determine_bound(double ai) const {
    return (ai < ridge_point_) ? "memory" : "compute";
}

std::string RooflineAnalyzer::generate_recommendation(const RooflineDataPoint& point) const {
    if (point.bound_type == "memory") {
        if (point.arithmetic_intensity < 1.0) {
            return "Very low AI. Increase data reuse with tiling/shared memory.";
        } else if (point.arithmetic_intensity < ridge_point_ * 0.5) {
            return "Memory bound. Consider larger tiles or prefetching.";
        } else {
            return "Near ridge point. Focus on both memory and compute optimizations.";
        }
    } else {
        if (point.efficiency < 30.0) {
            return "Low compute efficiency. Check occupancy, ILP, or bank conflicts.";
        } else if (point.efficiency < 60.0) {
            return "Moderate efficiency. Consider warp-level optimizations.";
        } else {
            return "Good efficiency. Fine-tune register usage and instruction mix.";
        }
    }
}

void RooflineAnalyzer::add_result(const BenchmarkResult& result) {
    RooflineDataPoint point;
    point.kernel_name = result.kernel_name;
    point.size = result.M;
    point.arithmetic_intensity = result.arithmetic_intensity;
    point.achieved_gflops = result.gflops;
    point.bound_type = determine_bound(point.arithmetic_intensity);
    point.efficiency = calculate_efficiency(point.arithmetic_intensity, point.achieved_gflops);
    point.recommendation = generate_recommendation(point);

    points_.push_back(point);
}

void RooflineAnalyzer::add_results(const std::vector<BenchmarkResult>& results) {
    for (const auto& result : results) {
        add_result(result);
    }
}

void RooflineAnalyzer::clear() {
    points_.clear();
}

RooflineData RooflineAnalyzer::get_data() const {
    RooflineData data;
    data.peak_gflops = peak_gflops_;
    data.peak_bandwidth_gbps = peak_bandwidth_gbps_;
    data.ridge_point = ridge_point_;
    data.points = points_;
    return data;
}

void RooflineAnalyzer::print_report() const {
    printf("================================================================================\n");
    printf("ROOFLINE ANALYSIS REPORT\n");
    printf("================================================================================\n");
    printf("Hardware Limits:\n");
    printf("  Peak FP32 Performance:  %.1f GFLOPS\n", peak_gflops_);
    printf("  Peak Memory Bandwidth:  %.1f GB/s\n", peak_bandwidth_gbps_);
    printf("  Ridge Point (AI):       %.2f FLOPs/Byte\n", ridge_point_);
    printf("--------------------------------------------------------------------------------\n");
    printf("\n");

    // ASCII Roofline visualization
    printf("Roofline Model (log scale approximation):\n");
    printf("          |\n");
    printf(" GFLOPS   |                    * Peak (%.0f)\n", peak_gflops_);
    printf(" (log)    |                ****\n");
    printf("          |            ****\n");
    printf("          |        ****\n");
    printf("          |    ****   [Ridge: AI=%.1f]\n", ridge_point_);
    printf("          |  **\n");
    printf("          | *\n");
    printf("          |*\n");
    printf("          +------------------------------------------------\n");
    printf("                  Arithmetic Intensity (FLOPs/Byte)\n\n");

    // Detailed table
    printf("--------------------------------------------------------------------------------\n");
    printf(" %-15s | Size | AI (F/B) | GFLOPS   | Bound   | Eff.  \n", "Kernel");
    printf("-----------------|------|----------|----------|---------|-------\n");

    for (const auto& point : points_) {
        printf(" %-15s | %4d | %8.2f | %8.1f | %-7s | %5.1f%%\n",
               point.kernel_name.c_str(),
               point.size,
               point.arithmetic_intensity,
               point.achieved_gflops,
               point.bound_type.c_str(),
               point.efficiency);
    }

    printf("--------------------------------------------------------------------------------\n\n");

    // Recommendations
    printf("Optimization Recommendations:\n");
    printf("--------------------------------------------------------------------------------\n");

    // Group by kernel name and show recommendations
    std::vector<std::string> seen_kernels;
    for (const auto& point : points_) {
        if (std::find(seen_kernels.begin(), seen_kernels.end(), point.kernel_name) == seen_kernels.end()) {
            seen_kernels.push_back(point.kernel_name);
            printf("  %s: %s\n", point.kernel_name.c_str(), point.recommendation.c_str());
        }
    }

    printf("================================================================================\n");
}

void RooflineAnalyzer::export_csv(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filepath.c_str());
        return;
    }

    // Metadata as comments
    file << "# Roofline Analysis Data\n";
    file << "# Peak GFLOPS: " << peak_gflops_ << "\n";
    file << "# Peak Bandwidth (GB/s): " << peak_bandwidth_gbps_ << "\n";
    file << "# Ridge Point (FLOPs/Byte): " << ridge_point_ << "\n";

    // Header
    file << "kernel_name,size,arithmetic_intensity,gflops,bound_type,efficiency\n";

    // Data
    for (const auto& point : points_) {
        file << point.kernel_name << ","
             << point.size << ","
             << std::fixed << std::setprecision(4) << point.arithmetic_intensity << ","
             << std::setprecision(2) << point.achieved_gflops << ","
             << point.bound_type << ","
             << point.efficiency << "\n";
    }

    file.close();
    printf("Roofline data exported to: %s\n", filepath.c_str());
}

void RooflineAnalyzer::generate_gnuplot_script(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filepath.c_str());
        return;
    }

    std::string data_file = filepath.substr(0, filepath.find_last_of('.')) + "_data.csv";

    file << "# Roofline Model Plot\n";
    file << "# Generated by CUDA Benchmark Framework\n";
    file << "# Usage: gnuplot " << filepath << "\n\n";

    file << "set terminal pngcairo size 1200,800 enhanced font 'Arial,12'\n";
    file << "set output '" << filepath.substr(0, filepath.find_last_of('.')) << ".png'\n\n";

    file << "set title 'Roofline Model'\n";
    file << "set xlabel 'Arithmetic Intensity (FLOPs/Byte)'\n";
    file << "set ylabel 'Performance (GFLOPS)'\n";
    file << "set logscale xy\n";
    file << "set grid\n\n";

    file << "# Hardware parameters\n";
    file << "peak_gflops = " << peak_gflops_ << "\n";
    file << "peak_bw = " << peak_bandwidth_gbps_ << "\n";
    file << "ridge_point = peak_gflops / peak_bw\n\n";

    file << "# Roofline function\n";
    file << "roofline(x) = (x < ridge_point) ? peak_bw * x : peak_gflops\n\n";

    file << "set xrange [0.1:1000]\n";
    file << "set yrange [10:" << peak_gflops_ * 1.5 << "]\n\n";

    file << "# Key settings\n";
    file << "set key top left\n\n";

    file << "plot roofline(x) title 'Roofline' with lines lw 3 lc rgb 'red', \\\n";
    file << "     '" << data_file << "' using 3:4 with points pt 7 ps 2 title 'Kernels', \\\n";
    file << "     '' using 3:4:1 with labels offset 1,1 font ',10' notitle\n";

    file.close();

    // Also export the data file for gnuplot
    std::ofstream data(data_file);
    data << "# kernel_name arithmetic_intensity gflops\n";
    for (const auto& point : points_) {
        data << point.kernel_name << " "
             << point.arithmetic_intensity << " "
             << point.achieved_gflops << "\n";
    }
    data.close();

    printf("Gnuplot script generated: %s\n", filepath.c_str());
    printf("Data file generated: %s\n", data_file.c_str());
    printf("Run: gnuplot %s\n", filepath.c_str());
}

void RooflineAnalyzer::generate_python_script(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filepath.c_str());
        return;
    }

    file << "#!/usr/bin/env python3\n";
    file << "\"\"\"Roofline Model Plot - Generated by CUDA Benchmark Framework\"\"\"\n\n";

    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n\n";

    file << "# Hardware parameters\n";
    file << "peak_gflops = " << peak_gflops_ << "\n";
    file << "peak_bandwidth = " << peak_bandwidth_gbps_ << "  # GB/s\n";
    file << "ridge_point = peak_gflops / peak_bandwidth\n\n";

    file << "# Kernel data: (name, arithmetic_intensity, gflops)\n";
    file << "kernels = [\n";
    for (const auto& point : points_) {
        file << "    ('" << point.kernel_name << "', "
             << point.arithmetic_intensity << ", "
             << point.achieved_gflops << "),\n";
    }
    file << "]\n\n";

    file << "# Create figure\n";
    file << "fig, ax = plt.subplots(figsize=(12, 8))\n\n";

    file << "# Plot roofline\n";
    file << "ai = np.logspace(-1, 3, 1000)\n";
    file << "performance = np.minimum(peak_bandwidth * ai, peak_gflops)\n";
    file << "ax.loglog(ai, performance, 'r-', linewidth=3, label='Roofline')\n\n";

    file << "# Plot kernel points\n";
    file << "for name, ai_val, gflops in kernels:\n";
    file << "    ax.scatter(ai_val, gflops, s=100, zorder=5)\n";
    file << "    ax.annotate(name, (ai_val, gflops), textcoords='offset points',\n";
    file << "                xytext=(5, 5), fontsize=9)\n\n";

    file << "# Add ridge point line\n";
    file << "ax.axvline(x=ridge_point, color='gray', linestyle='--', alpha=0.5,\n";
    file << "           label=f'Ridge Point (AI={ridge_point:.1f})')\n\n";

    file << "# Labels and formatting\n";
    file << "ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)\n";
    file << "ax.set_ylabel('Performance (GFLOPS)', fontsize=12)\n";
    file << "ax.set_title('Roofline Model Analysis', fontsize=14)\n";
    file << "ax.legend(loc='upper left')\n";
    file << "ax.grid(True, which='both', alpha=0.3)\n";
    file << "ax.set_xlim(0.1, 1000)\n";
    file << "ax.set_ylim(10, " << peak_gflops_ * 1.5 << ")\n\n";

    file << "plt.tight_layout()\n";
    file << "plt.savefig('" << filepath.substr(0, filepath.find_last_of('.')) << ".png', dpi=150)\n";
    file << "plt.show()\n";
    file << "print('Plot saved to " << filepath.substr(0, filepath.find_last_of('.')) << ".png')\n";

    file.close();
    printf("Python script generated: %s\n", filepath.c_str());
    printf("Run: python %s\n", filepath.c_str());
}

}  // namespace cudabench
