#!/bin/bash

# Run this script from the root of the repository:
# bash ./scripts/run_tests.sh [test_suite...]
# Examples:
#   bash ./scripts/run_tests.sh all
#   bash ./scripts/run_tests.sh python cpp
#   bash ./scripts/run_tests.sh python

# Available test suites
declare -A TEST_SUITES=(
    ["python"]="Python tests"
    ["cpp"]="C++ tests"
    # Add new test suites here in the future
    # ["rust"]="Rust tests"
    # ["cuda"]="CUDA-specific tests"
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

# Function to run Python tests
run_python_tests() {
    echo -e "\033[34mRunning Python tests...\033[0m"
    pytest -rs --color=yes ./tests/ --ignore ./tests/gpu/excluded
}

# Function to run C++ tests
run_cpp_tests() {
    echo -e "\033[34mBuilding C++ tests...\033[0m"
    # Clean up build directory
    rm -rf build
    # Build the CUDA extension module
    cmake -S . -B build
    cmake --build build -j $(nproc)

    echo -e "\033[34mRunning C++ tests...\033[0m"
    for test in ./build/test_*; do
        echo "Running $test..."
        ./$test
    done
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
            # Run all test suites
            for suite in "${!TEST_SUITES[@]}"; do
                run_${suite}_tests
            done
            ;;
        "python"|"cpp")
            # Run specific test suite
            run_${arg}_tests
            ;;
        *)
            echo "Error: Unknown test suite '$arg'"
            usage
            ;;
    esac
done
