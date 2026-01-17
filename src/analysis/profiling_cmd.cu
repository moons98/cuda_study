#include "analysis/profiling_cmd.cuh"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace cudabench {

ProfilingCommandGenerator::ProfilingCommandGenerator() {}

void ProfilingCommandGenerator::set_executable(const std::string& exe_path) {
    options_.executable = exe_path;
}

void ProfilingCommandGenerator::set_output_dir(const std::string& dir) {
    options_.output_dir = dir;
}

void ProfilingCommandGenerator::set_options(const ProfilingOptions& opts) {
    options_ = opts;
}

std::string ProfilingCommandGenerator::get_line_continuation() const {
#ifdef _WIN32
    return " ^";
#else
    return " \\";
#endif
}

std::string ProfilingCommandGenerator::sanitize_filename(const std::string& name) const {
    std::string result = name;
    std::replace(result.begin(), result.end(), ' ', '_');
    std::replace(result.begin(), result.end(), '/', '_');
    std::replace(result.begin(), result.end(), '\\', '_');
    return result;
}

std::string ProfilingCommandGenerator::generate_nsys_command(
    const std::string& kernel_name, int size) const {

    std::ostringstream cmd;
    std::string cont = get_line_continuation();
    std::string output_name = sanitize_filename(kernel_name) + "_" + std::to_string(size);

    cmd << "nsys profile";
    cmd << " --stats=true";
    cmd << " -o " << options_.output_dir << "/" << output_name;

    if (options_.nsys_cuda_api) {
        cmd << " --trace=cuda,nvtx";
    }

    if (options_.nsys_gpu_metrics) {
        cmd << " --gpu-metrics-device=all";
    }

    cmd << cont << "\n    " << options_.executable;
    cmd << " --single";
    cmd << " --kernel=" << kernel_name;
    cmd << " --size=" << size;
    cmd << " --runs=" << options_.profile_runs;
    cmd << " --no-verify";

    if (!options_.extra_args.empty()) {
        cmd << " " << options_.extra_args;
    }

    return cmd.str();
}

std::string ProfilingCommandGenerator::generate_ncu_command(
    const std::string& kernel_name, int size, const std::string& metric_set) const {

    std::ostringstream cmd;
    std::string cont = get_line_continuation();
    std::string output_name = sanitize_filename(kernel_name) + "_" + std::to_string(size);

    cmd << "ncu";

    if (metric_set == "full") {
        cmd << " --set full";
    } else if (metric_set == "memory") {
        cmd << " --set memory";
    } else if (metric_set == "roofline") {
        cmd << " --set roofline";
    } else {
        cmd << " --set " << metric_set;
    }

    cmd << " --export " << options_.output_dir << "/" << output_name << "_" << metric_set;

    cmd << cont << "\n    " << options_.executable;
    cmd << " --single";
    cmd << " --kernel=" << kernel_name;
    cmd << " --size=" << size;
    cmd << " --runs=1";  // NCU overhead is high, usually run once
    cmd << " --no-verify";

    if (!options_.extra_args.empty()) {
        cmd << " " << options_.extra_args;
    }

    return cmd.str();
}

std::vector<std::string> ProfilingCommandGenerator::generate_all_nsys_commands(
    const std::vector<std::string>& kernel_names,
    const std::vector<int>& sizes) const {

    std::vector<std::string> commands;
    for (const auto& kernel : kernel_names) {
        for (int size : sizes) {
            commands.push_back(generate_nsys_command(kernel, size));
        }
    }
    return commands;
}

std::vector<std::string> ProfilingCommandGenerator::generate_all_ncu_commands(
    const std::vector<std::string>& kernel_names,
    const std::vector<int>& sizes) const {

    std::vector<std::string> commands;
    for (const auto& kernel : kernel_names) {
        for (int size : sizes) {
            if (options_.ncu_full) {
                commands.push_back(generate_ncu_command(kernel, size, "full"));
            }
            if (options_.ncu_memory) {
                commands.push_back(generate_ncu_command(kernel, size, "memory"));
            }
            if (options_.ncu_roofline) {
                commands.push_back(generate_ncu_command(kernel, size, "roofline"));
            }
            if (!options_.ncu_full && !options_.ncu_memory && !options_.ncu_roofline) {
                // Default to full if nothing specified
                commands.push_back(generate_ncu_command(kernel, size, "full"));
            }
        }
    }
    return commands;
}

void ProfilingCommandGenerator::print_all_commands(
    const std::vector<std::string>& kernel_names,
    const std::vector<int>& sizes) const {

    printf("================================================================================\n");
    printf("PROFILING COMMANDS\n");
    printf("================================================================================\n");
    printf("Executable: %s\n", options_.executable.c_str());
    printf("Output Directory: %s\n", options_.output_dir.c_str());
    printf("================================================================================\n\n");

    // Nsight Systems commands
    printf("[Nsight Systems - Timeline Analysis]\n");
    printf("Quick overview of kernel execution timeline and CUDA API calls.\n");
    printf("--------------------------------------------------------------------------------\n\n");

    for (const auto& kernel : kernel_names) {
        for (int size : sizes) {
            printf("# %s kernel, size %d\n", kernel.c_str(), size);
            printf("%s\n\n", generate_nsys_command(kernel, size).c_str());
        }
    }

    // Nsight Compute commands
    printf("--------------------------------------------------------------------------------\n\n");
    printf("[Nsight Compute - Detailed Metrics]\n");
    printf("Deep dive into kernel performance (slower, run selectively).\n");
    printf("--------------------------------------------------------------------------------\n\n");

    for (const auto& kernel : kernel_names) {
        // Only show for first size to keep output manageable
        int size = sizes.empty() ? 1024 : sizes[0];

        printf("# %s kernel - full analysis\n", kernel.c_str());
        printf("%s\n\n", generate_ncu_command(kernel, size, "full").c_str());

        printf("# %s kernel - memory analysis\n", kernel.c_str());
        printf("%s\n\n", generate_ncu_command(kernel, size, "memory").c_str());
    }

    // View commands
    printf("--------------------------------------------------------------------------------\n\n");
    printf("[View Results]\n");
    printf("--------------------------------------------------------------------------------\n\n");

    printf("# Nsight Systems (GUI)\n");
    printf("nsys-ui %s/<profile_name>.nsys-rep\n\n", options_.output_dir.c_str());

    printf("# Nsight Compute (GUI)\n");
    printf("ncu-ui %s/<profile_name>.ncu-rep\n\n", options_.output_dir.c_str());

    printf("# Nsight Compute (CLI summary)\n");
    printf("ncu --import %s/<profile_name>.ncu-rep --page raw\n", options_.output_dir.c_str());

    printf("================================================================================\n");
}

void ProfilingCommandGenerator::generate_batch_script(
    const std::vector<std::string>& kernel_names,
    const std::vector<int>& sizes,
    const std::string& filepath) const {

    std::ofstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filepath.c_str());
        return;
    }

#ifdef _WIN32
    file << "@echo off\n";
    file << "REM Profiling batch script - Generated by CUDA Benchmark Framework\n\n";
    file << "REM Create output directory\n";
    file << "if not exist \"" << options_.output_dir << "\" mkdir \"" << options_.output_dir << "\"\n\n";
#else
    file << "#!/bin/bash\n";
    file << "# Profiling batch script - Generated by CUDA Benchmark Framework\n\n";
    file << "# Create output directory\n";
    file << "mkdir -p \"" << options_.output_dir << "\"\n\n";
#endif

    file << "echo Running Nsight Systems profiles...\n\n";

    // Nsight Systems profiles
    for (const auto& kernel : kernel_names) {
        for (int size : sizes) {
            file << "echo Profiling " << kernel << " size=" << size << "\n";
            file << generate_nsys_command(kernel, size) << "\n\n";
        }
    }

    file << "echo.\n";
    file << "echo Profiling complete. Results saved to " << options_.output_dir << "\n";

#ifdef _WIN32
    file << "pause\n";
#endif

    file.close();
    printf("Batch script generated: %s\n", filepath.c_str());
}

}  // namespace cudabench
