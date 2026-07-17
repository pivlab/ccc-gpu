#!/bin/bash

# Run this script from the root of the repository:
# bash ./scripts/run_tests.sh [test_suite...]
# Examples:
#   bash ./scripts/run_tests.sh all
#   bash ./scripts/run_tests.sh cpu gpu cpp
#   bash ./scripts/run_tests.sh cpu
#
# Test suites map to the pytest markers registered in pyproject.toml
# (gpu / slow / network) and to the CUDA C++ gtests (cpp):
#   cpu  -> marker expression "not gpu and not slow and not network" (what CI runs)
#   gpu  -> marker expression "gpu" (requires a CUDA device + ccc_cuda_ext)
#   slow -> marker expression "slow" (1M-element / heavy CPU cases)
#   cpp  -> the CUDA C++ gtests via CMake (-DCCC_BUILD_TESTS=ON) + ctest

# Available test suites
declare -A TEST_SUITES=(
    ["cpu"]="CPU test subset (not gpu/slow/network)"
    ["gpu"]="GPU tests (marker: gpu)"
    ["slow"]="Slow CPU tests (marker: slow)"
    ["cpp"]="CUDA C++ gtests (ctest)"
)

# Research-only modules import undeclared heavy deps (requests/minepy/IPython)
# and are excluded from the published wheel; ignore them here too.
RESEARCH_IGNORES=(
    --ignore=tests/test_giant.py
    --ignore=tests/test_methods.py
    --ignore=tests/test_plots.py
    --ignore=tests/test_corr.py
)

# Function to display usage
usage() {
    echo "Usage: $0 [test_suite...]"
    echo "Available test suites:"
    echo "  all - Run all test suites"
    for suite in "${!TEST_SUITES[@]}"; do
        echo "  $suite - Run ${TEST_SUITES[$suite]}"
    done
    exit 1
}

# CPU subset (the CI target).
run_cpu_tests() {
    echo -e "\033[34mRunning CPU test subset (not gpu/slow/network)...\033[0m"
    pytest -rs --color=yes tests/ \
        "${RESEARCH_IGNORES[@]}" \
        -m "not gpu and not slow and not network" -o addopts=""
}

# GPU tests (needs a CUDA device + the built extension).
run_gpu_tests() {
    echo -e "\033[34mRunning GPU tests (marker: gpu)...\033[0m"
    pytest -rs --color=yes tests/ -m gpu -o addopts=""
}

# Slow CPU tests (1M-element / heavy parallel cases).
run_slow_tests() {
    echo -e "\033[34mRunning slow CPU tests (marker: slow)...\033[0m"
    pytest -rs --color=yes tests/ "${RESEARCH_IGNORES[@]}" -m slow -o addopts=""
}

# CUDA C++ gtests via CMake + ctest.
run_cpp_tests() {
    echo -e "\033[34mBuilding and running CUDA C++ gtests (ctest)...\033[0m"
    rm -rf build-tests
    cmake -S . -B build-tests -DCCC_BUILD_TESTS=ON -GNinja
    cmake --build build-tests
    ctest --test-dir build-tests --output-on-failure
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    usage
fi

# Exit immediately if a command exits with a non-zero status
set -e

# Setup environment
source ./scripts/setup_dev.sh

# Check if cccgpu is installed and uninstall if it exists
echo -e "\033[34mChecking for existing cccgpu installation...\033[0m"
if pip show cccgpu > /dev/null 2>&1; then
    echo -e "\033[33mUninstalling existing cccgpu...\033[0m"
    pip uninstall -y cccgpu
fi

# Install cccgpu with the cuda extension module
echo -e "\033[34mInstalling cccgpu with the cuda extension module...\033[0m"
if ! pip install .; then
    echo -e "\033[31mFailed to install cccgpu. Exiting...\033[0m"
    exit 1
fi

# Process arguments
for arg in "$@"; do
    case $arg in
        "all")
            for suite in "${!TEST_SUITES[@]}"; do
                run_${suite}_tests
            done
            ;;
        "cpu"|"gpu"|"slow"|"cpp")
            run_${arg}_tests
            ;;
        *)
            echo "Error: Unknown test suite '$arg'"
            usage
            ;;
    esac
done
