#!/bin/bash
# Script to upload cccgpu package to PyPI or TestPyPI
# Requires TWINE_USERNAME and TWINE_PASSWORD environment variables or ~/.pypirc
#
# Usage:
#   ./10-upload_to_pypi.sh [testpypi|pypi]
#
# Examples:
#   ./10-upload_to_pypi.sh              # uploads to TestPyPI (default)
#   ./10-upload_to_pypi.sh testpypi     # uploads to TestPyPI
#   ./10-upload_to_pypi.sh pypi         # uploads to PyPI

set -e  # Exit on error

# Parse command-line argument
REPOSITORY="${1:-testpypi}"

if [ "$REPOSITORY" = "pypi" ]; then
    REPOSITORY_NAME="PyPI"
    REPOSITORY_URL="https://pypi.org/project/cccgpu/"
    INSTALL_CMD="pip install cccgpu"
elif [ "$REPOSITORY" = "testpypi" ]; then
    REPOSITORY_NAME="TestPyPI"
    REPOSITORY_URL="https://test.pypi.org/project/cccgpu/"
    INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cccgpu"
else
    echo "Error: Invalid repository '$REPOSITORY'"
    echo ""
    echo "Usage: $0 [testpypi|pypi]"
    echo ""
    echo "  testpypi - Upload to TestPyPI (test.pypi.org) [default]"
    echo "  pypi     - Upload to PyPI (pypi.org)"
    exit 1
fi

echo "======================================="
echo "Uploading to $REPOSITORY_NAME"
echo "======================================="

# Activate conda environment
echo "Activating conda environment..."
# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || eval "$(mamba shell.bash hook)" 2>/dev/null || true
conda activate ccc-gpu 2>/dev/null || mamba activate ccc-gpu

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "Error: dist/ directory not found. Run ./scripts/00-build_package.sh  first."
    exit 1
fi

# Check for packages
if [ -z "$(ls -A dist/)" ]; then
    echo "Error: No packages found in dist/ directory."
    exit 1
fi

# Install twine if not already installed
echo "Ensuring twine is installed..."
python -m pip install --upgrade twine

# Check the package before upload (optional but recommended)
echo "Checking package integrity..."
python -m twine check dist/*

# Upload to repository
echo "Uploading to $REPOSITORY_NAME..."

# Use project-local .pypirc if it exists
if [ -f ".pypirc" ]; then
    echo "Using project .pypirc configuration..."
    python -m twine upload --config-file .pypirc --repository $REPOSITORY --verbose dist/*
else
    if [ "$REPOSITORY" = "testpypi" ]; then
        echo "Note: You will need to provide your TestPyPI credentials"
        echo "You can create an account at: https://test.pypi.org/account/register/"
    else
        echo "Note: You will need to provide your PyPI credentials"
        echo "You can create an account at: https://pypi.org/account/register/"
    fi
    echo ""
    python -m twine upload --repository $REPOSITORY --verbose dist/*
fi

echo "======================================="
echo "Upload complete!"
echo "Your package should be available at:"
echo "$REPOSITORY_URL"
echo ""
echo "To install from $REPOSITORY_NAME:"
echo "$INSTALL_CMD"
echo "======================================="