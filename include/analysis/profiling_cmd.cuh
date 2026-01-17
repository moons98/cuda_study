#pragma once

#include <string>
#include <vector>

namespace cudabench {

struct ProfilingOptions {
    std::string executable = "./benchmark";
    std::string output_dir = "./output/profiles";

    // Nsight Systems options
    bool nsys_timeline = true;
    bool nsys_cuda_api = true;
    bool nsys_gpu_metrics = false;

    // Nsight Compute options
    bool ncu_full = false;
    bool ncu_memory = false;
    bool ncu_roofline = false;

    // General
    int profile_runs = 10;
    std::string extra_args = "";
};

class ProfilingCommandGenerator {
public:
    ProfilingCommandGenerator();

    // Configuration
    void set_executable(const std::string& exe_path);
    void set_output_dir(const std::string& dir);
    void set_options(const ProfilingOptions& opts);

    // Generate commands for single kernel
    std::string generate_nsys_command(const std::string& kernel_name, int size) const;
    std::string generate_ncu_command(const std::string& kernel_name, int size,
                                     const std::string& metric_set = "full") const;

    // Generate commands for multiple kernels
    std::vector<std::string> generate_all_nsys_commands(
        const std::vector<std::string>& kernel_names,
        const std::vector<int>& sizes) const;

    std::vector<std::string> generate_all_ncu_commands(
        const std::vector<std::string>& kernel_names,
        const std::vector<int>& sizes) const;

    // Print all commands
    void print_all_commands(const std::vector<std::string>& kernel_names,
                           const std::vector<int>& sizes) const;

    // Generate batch script
    void generate_batch_script(const std::vector<std::string>& kernel_names,
                              const std::vector<int>& sizes,
                              const std::string& filepath) const;

private:
    ProfilingOptions options_;

    std::string get_line_continuation() const;
    std::string sanitize_filename(const std::string& name) const;
};

}  // namespace cudabench
