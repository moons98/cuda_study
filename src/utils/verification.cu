#include "utils/verification.cuh"
#include <cmath>
#include <cstdio>

namespace cudabench {

VerificationResult verify_matrix(const float* ref, const float* test, size_t N, float tolerance) {
    VerificationResult result;
    result.passed = true;
    result.max_error = 0.0;
    result.avg_error = 0.0;
    result.error_count = 0;
    result.first_error_index = 0;

    double total_error = 0.0;

    for (size_t i = 0; i < N; i++) {
        double diff = std::fabs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
        total_error += diff;

        if (diff > result.max_error) {
            result.max_error = diff;
        }

        if (std::isnan(diff) || diff > tolerance) {
            if (result.error_count == 0) {
                result.first_error_index = i;
            }
            result.error_count++;
            result.passed = false;
        }
    }

    result.avg_error = total_error / N;

    return result;
}

VerificationResult verify_matrix_detailed(const float* ref, const float* test,
                                          int M, int N, float tolerance,
                                          bool print_errors, int max_errors_to_print) {
    VerificationResult result;
    result.passed = true;
    result.max_error = 0.0;
    result.avg_error = 0.0;
    result.error_count = 0;
    result.first_error_index = 0;

    double total_error = 0.0;
    size_t total_elements = static_cast<size_t>(M) * N;
    int errors_printed = 0;

    for (size_t i = 0; i < total_elements; i++) {
        double diff = std::fabs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
        total_error += diff;

        if (diff > result.max_error) {
            result.max_error = diff;
        }

        if (std::isnan(diff) || diff > tolerance) {
            if (result.error_count == 0) {
                result.first_error_index = i;
            }
            result.error_count++;
            result.passed = false;

            if (print_errors && errors_printed < max_errors_to_print) {
                int row = static_cast<int>(i / N);
                int col = static_cast<int>(i % N);
                printf("  Error at [%d,%d]: ref=%f, test=%f, diff=%f\n",
                       row, col, ref[i], test[i], diff);
                errors_printed++;
            }
        }
    }

    result.avg_error = total_error / total_elements;

    if (print_errors && result.error_count > max_errors_to_print) {
        printf("  ... and %zu more errors\n", result.error_count - max_errors_to_print);
    }

    return result;
}

}  // namespace cudabench
