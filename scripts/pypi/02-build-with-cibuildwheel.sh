#!/bin/bash
# Script to build cccgpu wheels using cibuildwheel
# This produces manylinux wheels compatible with most Linux distributions

set -e  # Exit on error

echo "========================================================================="
echo "Building cccgpu wheels with cibuildwheel"
echo "========================================================================="
echo ""

# Project root directory (assumes script is in scripts/pypi/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if Docker is installed and running
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "✗ Error: Docker is not installed"
    echo "  Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker ps &> /dev/null; then
    echo "✗ Error: Docker daemon is not running"
    echo "  Please start Docker and try again"
    echo ""
    echo "  On Ubuntu/Debian:"
    echo "    sudo systemctl start docker"
    echo ""
    echo "  Or start Docker Desktop if using that"
    exit 1
fi

echo "✓ Docker is installed and running"
echo ""

# Check if cibuildwheel is installed
echo "Checking for cibuildwheel..."
if ! command -v cibuildwheel &> /dev/null; then
    echo "cibuildwheel not found. Installing..."

    # Try to use current Python environment
    if command -v pip &> /dev/null; then
        pip install cibuildwheel
    elif command -v pip3 &> /dev/null; then
        pip3 install cibuildwheel
    else
        echo "✗ Error: pip not found"
        echo "  Please install cibuildwheel manually:"
        echo "    pip install cibuildwheel"
        exit 1
    fi
fi

CIBW_VERSION=$(cibuildwheel --version 2>&1 || echo "unknown")
echo "✓ cibuildwheel installed: ${CIBW_VERSION}"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info libs/ccc/*.egg-info
mkdir -p dist
echo "✓ Build directories cleaned"
echo ""

# Display cibuildwheel configuration
echo "========================================================================="
echo "Build Configuration"
echo "========================================================================="
echo "Platform: Linux"
echo "Python versions: 3.10, 3.11, 3.12, 3.13, 3.14, 3.15"
echo "Image: manylinux_2_28"
echo "CUDA: 12.6 (will be installed in container)"
echo ""
echo "Note: First build may take 1-2 hours due to CUDA download"
echo "      Subsequent builds will be faster due to Docker caching"
echo ""

# Ask for confirmation
echo "Proceed with build? (y/N)"
read -t 10 -r CONFIRM || CONFIRM="n"
if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "Build cancelled"
    exit 0
fi

echo ""
echo "========================================================================="
echo "Starting cibuildwheel build"
echo "========================================================================="
echo ""

# Run cibuildwheel
# All configuration is in pyproject.toml under [tool.cibuildwheel]
if cibuildwheel --platform linux --output-dir dist .; then
    echo ""
    echo "========================================================================="
    echo "Build completed successfully!"
    echo "========================================================================="
    echo ""

    # List built wheels
    echo "Built wheels:"
    ls -lh dist/*.whl 2>/dev/null || echo "  (no wheels found)"
    echo ""

    # Count wheels
    WHEEL_COUNT=$(ls dist/*.whl 2>/dev/null | wc -l)
    echo "Total wheels built: ${WHEEL_COUNT}"
    echo ""

    echo "========================================================================="
    echo "Next steps:"
    echo "  1. List wheels: ./scripts/pypi/03-list-built-wheels.sh"
    echo "  2. Test wheels manually (see scripts/pypi/README.md)"
    echo "  3. Upload: ./scripts/pypi/10-upload_to_test_pypi.sh"
    echo "========================================================================="
    echo ""
    echo "✓ Build complete!"
else
    echo ""
    echo "========================================================================="
    echo "Build failed!"
    echo "========================================================================="
    echo ""
    echo "Common issues:"
    echo "  1. Docker out of disk space: Run 'docker system prune -a'"
    echo "  2. CUDA download failed: Check internet connection"
    echo "  3. Python version not available: Check cibuildwheel output"
    echo ""
    echo "For more troubleshooting, see scripts/pypi/README.md"
    exit 1
fi