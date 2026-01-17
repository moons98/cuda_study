#!/bin/bash
# CUDA Benchmark Framework - Linux/macOS Build Script
# Requirements: CMake 3.18+, CUDA Toolkit, GCC/Clang

set -e

# Configuration
BUILD_TYPE="Release"
BUILD_DIR="build"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        release)
            BUILD_TYPE="Release"
            shift
            ;;
        clean)
            echo "Cleaning build directory..."
            rm -rf "$BUILD_DIR"
            echo "Done."
            exit 0
            ;;
        rebuild)
            rm -rf "$BUILD_DIR"
            shift
            ;;
        help|--help|-h)
            echo ""
            echo "CUDA Benchmark Framework - Build Script"
            echo ""
            echo "Usage: ./build.sh [options]"
            echo ""
            echo "Options:"
            echo "  debug     Build in Debug mode"
            echo "  release   Build in Release mode (default)"
            echo "  clean     Remove build directory"
            echo "  rebuild   Clean and rebuild"
            echo "  help      Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found"
    echo "Please install CMake 3.18 or later"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "Error: NVCC not found"
    echo "Please install CUDA Toolkit and add to PATH"
    exit 1
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Configure
echo ""
echo "============================================================"
echo "Configuring CUDA Benchmark Framework ($BUILD_TYPE)"
echo "============================================================"
echo ""

cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..

# Build
echo ""
echo "============================================================"
echo "Building..."
echo "============================================================"
echo ""

cmake --build . --config "$BUILD_TYPE" --parallel $(nproc)

cd ..

echo ""
echo "============================================================"
echo "Build successful!"
echo "============================================================"
echo ""
echo "Executable: $BUILD_DIR/bin/benchmark"
echo ""
echo "Usage examples:"
echo "  $BUILD_DIR/bin/benchmark --list"
echo "  $BUILD_DIR/bin/benchmark --single --kernel=naive --size=1024"
echo "  $BUILD_DIR/bin/benchmark --compare --sizes=512,1024,2048"
echo "  $BUILD_DIR/bin/benchmark --all"
echo ""
