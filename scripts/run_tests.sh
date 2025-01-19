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
    cmake --build build

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

# Setup environment
source ./scripts/setup_dev.sh

# Install cccgpu with the cuda extension module
echo -e "\033[34mInstalling cccgpu with the cuda extension module...\033[0m"
pip install .

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

# Uninstall cccgpu
echo -e "\033[34mUninstalling cccgpu...\033[0m"
pip uninstall cccgpu -y
