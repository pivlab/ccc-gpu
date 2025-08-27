#!/bin/bash
# Script to upload cccgpu package to test PyPI
# Requires TWINE_USERNAME and TWINE_PASSWORD environment variables or ~/.pypirc

set -e  # Exit on error

echo "======================================="
echo "Uploading cccgpu package to test PyPI"
echo "======================================="

# Activate conda environment
echo "Activating conda environment..."
# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || eval "$(mamba shell.bash hook)" 2>/dev/null || true
conda activate ccc-gpu 2>/dev/null || mamba activate ccc-gpu

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "Error: dist/ directory not found. Run ./scripts/build_package.sh first."
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

# Upload to test PyPI
echo "Uploading to test PyPI..."

# Use project-local .pypirc if it exists
if [ -f ".pypirc" ]; then
    echo "Using project .pypirc configuration..."
    python -m twine upload --config-file .pypirc --repository testpypi --verbose dist/*
else
    echo "Note: You will need to provide your test PyPI credentials"
    echo "You can create an account at: https://test.pypi.org/account/register/"
    echo ""
    python -m twine upload --repository testpypi --verbose dist/*
fi

echo "======================================="
echo "Upload complete!"
echo "Your package should be available at:"
echo "https://test.pypi.org/project/cccgpu/"
echo ""
echo "To install from test PyPI:"
echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cccgpu"
echo "======================================="