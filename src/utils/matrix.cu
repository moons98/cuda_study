#include "utils/matrix.cuh"
#include <cstdio>
#include <cstdlib>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

namespace cudabench {

void randomize_matrix(float* mat, size_t N) {
    // Use high-precision timer for seeding
#ifdef _WIN32
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    srand(static_cast<unsigned int>(counter.LowPart));
#else
    struct timeval time;
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
#endif

    for (size_t i = 0; i < N; i++) {
        float tmp = static_cast<float>(rand() % 5) + 0.01f * (rand() % 5);
        mat[i] = (rand() % 2 == 0) ? tmp : -tmp;
    }
}

void range_init_matrix(float* mat, size_t N) {
    for (size_t i = 0; i < N; i++) {
        mat[i] = static_cast<float>(i);
    }
}

void zero_init_matrix(float* mat, size_t N) {
    for (size_t i = 0; i < N; i++) {
        mat[i] = 0.0f;
    }
}

void constant_init_matrix(float* mat, size_t N, float value) {
    for (size_t i = 0; i < N; i++) {
        mat[i] = value;
    }
}

void copy_matrix(const float* src, float* dest, size_t N) {
    for (size_t i = 0; i < N; i++) {
        dest[i] = src[i];
    }
}

void print_matrix(const float* A, int M, int N, const std::string& title) {
    if (!title.empty()) {
        printf("%s:\n", title.c_str());
    }
    printf("[");
    for (int i = 0; i < M; i++) {
        if (i > 0) printf(" ");
        for (int j = 0; j < N; j++) {
            printf("%8.4f", A[i * N + j]);
            if (j < N - 1) printf(", ");
        }
        if (i < M - 1) printf(";\n");
    }
    printf("]\n");
}

void print_matrix_to_file(const float* A, int M, int N, std::ofstream& fs) {
    fs << std::setprecision(4) << std::fixed;
    fs << "[";
    for (int i = 0; i < M; i++) {
        if (i > 0) fs << " ";
        for (int j = 0; j < N; j++) {
            fs << std::setw(8) << A[i * N + j];
            if (j < N - 1) fs << ", ";
        }
        if (i < M - 1) fs << ";\n";
    }
    fs << "]\n";
}

}  // namespace cudabench
