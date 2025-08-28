#!/bin/bash
# Script to build cccgpu package for PyPI distribution
# This script builds both source distribution and wheels

set -e  # Exit on error

echo "====================================="
echo "Building cccgpu package for PyPI"
echo "====================================="

# Activate conda environment
echo "Activating conda environment..."
# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || eval "$(mamba shell.bash hook)" 2>/dev/null || true
conda activate ccc-gpu 2>/dev/null || mamba activate ccc-gpu

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info libs/ccc/*.egg-info

# Install build dependencies
echo "Installing build dependencies..."
python -m pip install --upgrade pip
python -m pip install --upgrade build twine setuptools wheel auditwheel

# Build the package
echo "Building source distribution and wheel..."
python -m build

# List built packages
echo "====================================="
echo "Built packages:"
ls -la dist/
echo "====================================="

echo "Build complete! Packages are in the dist/ directory."
echo "To upload to test PyPI, run: ./scripts/pypi/10-upload_to_test_pypi.sh"