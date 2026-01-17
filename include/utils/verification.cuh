#pragma once

#include <cstddef>

namespace cudabench {

struct VerificationResult {
    bool passed;
    double max_error;
    double avg_error;
    size_t error_count;
    size_t first_error_index;
};

// Verify two matrices are equal within tolerance
VerificationResult verify_matrix(const float* ref, const float* test, size_t N, float tolerance = 0.01f);

// Verify with detailed reporting
VerificationResult verify_matrix_detailed(const float* ref, const float* test,
                                          int M, int N, float tolerance = 0.01f,
                                          bool print_errors = false, int max_errors_to_print = 10);

}  // namespace cudabench
