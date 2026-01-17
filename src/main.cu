#include "core/benchmark.cuh"
#include "core/device_info.cuh"
#include "kernels/kernel_registry.cuh"
#include "analysis/comparison.cuh"
#include "analysis/roofline.cuh"
#include "analysis/profiling_cmd.cuh"
#include "utils/csv_export.cuh"
#include "sgemm/cublas_wrapper.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

namespace cudabench {

//=============================================================================
// Command line argument parsing
//=============================================================================

struct CliArgs {
    // Mode
    std::string mode = "single";  // single, compare, roofline, profile, all

    // Kernel selection
    std::string kernel = "naive";
    std::vector<std::string> kernels;

    // Sizes
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    // Benchmark config
    int warmup = 5;
    int runs = 50;
    bool verify = true;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Comparison
    std::string baseline = "cublas";

    // Output
    std::string output_dir = "./output";
    std::string format = "console";  // console, csv

    // Misc
    int device = 0;
    bool list_kernels = false;
    bool help = false;
};

void print_usage(const char* prog_name) {
    printf("CUDA Kernel Benchmark Framework\n\n");
    printf("Usage: %s [MODE] [OPTIONS]\n\n", prog_name);
    printf("MODES:\n");
    printf("  --single      Benchmark a single kernel (default)\n");
    printf("  --compare     Compare multiple kernels\n");
    printf("  --roofline    Perform roofline analysis\n");
    printf("  --profile     Generate profiling commands\n");
    printf("  --all         Run all analyses (compare + roofline + profile)\n\n");
    printf("OPTIONS:\n");
    printf("  --kernel=NAME       Kernel name for single mode (default: naive)\n");
    printf("  --kernels=A,B,C     Comma-separated kernel names for compare mode\n");
    printf("  --size=N            Single matrix size\n");
    printf("  --sizes=A,B,C       Comma-separated sizes (default: 128,256,512,1024,2048,4096)\n");
    printf("  --baseline=NAME     Baseline kernel for comparison (default: cublas)\n");
    printf("  --warmup=N          Number of warmup runs (default: 5)\n");
    printf("  --runs=N            Number of benchmark runs (default: 50)\n");
    printf("  --verify            Enable verification (default: on)\n");
    printf("  --no-verify         Disable verification\n");
    printf("  --alpha=F           GEMM alpha parameter (default: 1.0)\n");
    printf("  --beta=F            GEMM beta parameter (default: 0.0)\n");
    printf("  --output=PATH       Output directory (default: ./output)\n");
    printf("  --format=FMT        Output format: console, csv (default: console)\n");
    printf("  --device=N          CUDA device ID (default: 0)\n");
    printf("  --list              List available kernels\n");
    printf("  --help              Show this help message\n\n");
    printf("EXAMPLES:\n");
    printf("  %s --single --kernel=naive --size=1024\n", prog_name);
    printf("  %s --compare --sizes=512,1024,2048\n", prog_name);
    printf("  %s --all --output=./results\n", prog_name);
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<int> parse_sizes(const std::string& str) {
    std::vector<int> sizes;
    auto tokens = split_string(str, ',');
    for (const auto& t : tokens) {
        sizes.push_back(std::stoi(t));
    }
    return sizes;
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // Modes
        if (arg == "--single") {
            args.mode = "single";
        } else if (arg == "--compare") {
            args.mode = "compare";
        } else if (arg == "--roofline") {
            args.mode = "roofline";
        } else if (arg == "--profile") {
            args.mode = "profile";
        } else if (arg == "--all") {
            args.mode = "all";
        }
        // Kernel options
        else if (arg.rfind("--kernel=", 0) == 0) {
            args.kernel = arg.substr(9);
        } else if (arg.rfind("--kernels=", 0) == 0) {
            args.kernels = split_string(arg.substr(10), ',');
        }
        // Size options
        else if (arg.rfind("--size=", 0) == 0) {
            args.sizes = {std::stoi(arg.substr(7))};
        } else if (arg.rfind("--sizes=", 0) == 0) {
            args.sizes = parse_sizes(arg.substr(8));
        }
        // Benchmark options
        else if (arg.rfind("--baseline=", 0) == 0) {
            args.baseline = arg.substr(11);
        } else if (arg.rfind("--warmup=", 0) == 0) {
            args.warmup = std::stoi(arg.substr(9));
        } else if (arg.rfind("--runs=", 0) == 0) {
            args.runs = std::stoi(arg.substr(7));
        } else if (arg == "--verify") {
            args.verify = true;
        } else if (arg == "--no-verify") {
            args.verify = false;
        } else if (arg.rfind("--alpha=", 0) == 0) {
            args.alpha = std::stof(arg.substr(8));
        } else if (arg.rfind("--beta=", 0) == 0) {
            args.beta = std::stof(arg.substr(7));
        }
        // Output options
        else if (arg.rfind("--output=", 0) == 0) {
            args.output_dir = arg.substr(9);
        } else if (arg.rfind("--format=", 0) == 0) {
            args.format = arg.substr(9);
        }
        // Device
        else if (arg.rfind("--device=", 0) == 0) {
            args.device = std::stoi(arg.substr(9));
        }
        // Misc
        else if (arg == "--list") {
            args.list_kernels = true;
        } else if (arg == "--help" || arg == "-h") {
            args.help = true;
        }
    }

    return args;
}

BenchmarkConfig create_benchmark_config(const CliArgs& args) {
    BenchmarkConfig config;
    config.sizes = args.sizes;
    config.warmup_runs = args.warmup;
    config.benchmark_runs = args.runs;
    config.verify = args.verify;
    config.alpha = args.alpha;
    config.beta = args.beta;
    config.print_progress = true;
    return config;
}

//=============================================================================
// Mode handlers
//=============================================================================

void run_single_mode(const CliArgs& args) {
    KernelRegistry& registry = get_global_registry();
    const KernelInfo* kernel = registry.get(args.kernel);

    if (!kernel) {
        fprintf(stderr, "Error: Kernel '%s' not found\n", args.kernel.c_str());
        fprintf(stderr, "Use --list to see available kernels\n");
        return;
    }

    DeviceInfo device = get_device_info(args.device);
    device.print();

    BenchmarkConfig config = create_benchmark_config(args);
    Benchmark bench;
    bench.set_device(args.device);

    auto results = bench.run(*kernel, config);

    if (args.format == "csv") {
        std::string filepath = args.output_dir + "/single_" + args.kernel + ".csv";
        CsvExporter::export_results(results, filepath);
    }
}

void run_compare_mode(const CliArgs& args) {
    KernelComparator comparator;

    if (args.kernels.empty()) {
        comparator.add_all_kernels();
    } else {
        for (const auto& name : args.kernels) {
            comparator.add_kernel(name);
        }
    }

    comparator.set_baseline(args.baseline);

    BenchmarkConfig config = create_benchmark_config(args);
    config.print_progress = false;

    auto result = comparator.run(config);
    comparator.print_comparison_table(result);
    comparator.print_summary(result);

    if (args.format == "csv") {
        std::string filepath = args.output_dir + "/csv/comparison.csv";
        comparator.export_csv(result, filepath);
    }
}

void run_roofline_mode(const CliArgs& args) {
    DeviceInfo device = get_device_info(args.device);
    RooflineAnalyzer roofline(device);

    KernelRegistry& registry = get_global_registry();
    Benchmark bench;
    bench.set_device(args.device);

    BenchmarkConfig config = create_benchmark_config(args);
    config.print_progress = false;

    std::vector<std::string> kernel_names;
    if (args.kernels.empty()) {
        kernel_names = registry.get_names();
    } else {
        kernel_names = args.kernels;
    }

    printf("Running roofline analysis...\n");

    for (const auto& name : kernel_names) {
        const KernelInfo* kernel = registry.get(name);
        if (kernel) {
            printf("  Benchmarking: %s... ", name.c_str());
            fflush(stdout);

            auto results = bench.run(*kernel, config);
            roofline.add_results(results);

            printf("done\n");
        }
    }

    printf("\n");
    roofline.print_report();

    // Export
    roofline.export_csv(args.output_dir + "/roofline/roofline_data.csv");
    roofline.generate_gnuplot_script(args.output_dir + "/roofline/roofline_plot.gp");
    roofline.generate_python_script(args.output_dir + "/roofline/roofline_plot.py");
}

void run_profile_mode(const CliArgs& args) {
    ProfilingCommandGenerator profiler;
    profiler.set_executable(args.output_dir + "/../build/bin/benchmark");
    profiler.set_output_dir(args.output_dir + "/profiles");

    KernelRegistry& registry = get_global_registry();
    std::vector<std::string> kernel_names;

    if (args.kernels.empty()) {
        kernel_names = registry.get_names();
    } else {
        kernel_names = args.kernels;
    }

    profiler.print_all_commands(kernel_names, args.sizes);

    // Generate batch script
    std::string script_path = args.output_dir + "/profiles/run_profiles";
#ifdef _WIN32
    script_path += ".bat";
#else
    script_path += ".sh";
#endif
    profiler.generate_batch_script(kernel_names, args.sizes, script_path);
}

void run_all_mode(const CliArgs& args) {
    DeviceInfo device = get_device_info(args.device);
    device.print();

    printf("\n");
    printf("================================================================================\n");
    printf("[1/3] KERNEL COMPARISON\n");
    printf("================================================================================\n\n");
    run_compare_mode(args);

    printf("\n");
    printf("================================================================================\n");
    printf("[2/3] ROOFLINE ANALYSIS\n");
    printf("================================================================================\n\n");
    run_roofline_mode(args);

    printf("\n");
    printf("================================================================================\n");
    printf("[3/3] PROFILING COMMANDS\n");
    printf("================================================================================\n\n");
    run_profile_mode(args);

    printf("\n");
    printf("================================================================================\n");
    printf("ALL ANALYSES COMPLETE\n");
    printf("================================================================================\n");
    printf("Results saved to: %s\n", args.output_dir.c_str());
    printf("  - Comparison CSV:  %s/csv/comparison.csv\n", args.output_dir.c_str());
    printf("  - Roofline data:   %s/roofline/\n", args.output_dir.c_str());
    printf("  - Profile scripts: %s/profiles/\n", args.output_dir.c_str());
    printf("================================================================================\n");
}

}  // namespace cudabench

//=============================================================================
// Main entry point
//=============================================================================

int main(int argc, char** argv) {
    using namespace cudabench;

    CliArgs args = parse_args(argc, argv);

    // Handle help
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    // Handle list kernels
    if (args.list_kernels) {
        get_global_registry().print_available();
        return 0;
    }

    // Set device
    cudaError_t err = cudaSetDevice(args.device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to set CUDA device %d: %s\n",
                args.device, cudaGetErrorString(err));
        return 1;
    }

    // Run selected mode
    if (args.mode == "single") {
        run_single_mode(args);
    } else if (args.mode == "compare") {
        run_compare_mode(args);
    } else if (args.mode == "roofline") {
        run_roofline_mode(args);
    } else if (args.mode == "profile") {
        run_profile_mode(args);
    } else if (args.mode == "all") {
        run_all_mode(args);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", args.mode.c_str());
        print_usage(argv[0]);
        return 1;
    }

    // Cleanup cuBLAS
    kernels::cleanup_cublas();

    return 0;
}
