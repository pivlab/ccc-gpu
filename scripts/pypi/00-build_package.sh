#!/bin/bash
# ============================================================================
# Legacy Script: Build cccgpu package for single Python version (3.10)
# ============================================================================
#
# This script is the original single-version builder, kept for backwards
# compatibility. It builds a wheel for Python 3.10 only.
#
# For building wheels for multiple Python versions (3.10-3.15), use:
#   • Fast local builds:    ./01-build-multi-python-conda.sh
#   • Production builds:    ./02-build-with-cibuildwheel.sh
#
# See README.md for detailed documentation.
# ============================================================================

set -e  # Exit on error

echo "====================================="
echo "Building cccgpu package (Python 3.10)"
echo "====================================="
echo ""
echo "⚠️  Note: This script builds for Python 3.10 only"
echo "    For multiple Python versions, use:"
echo "      • ./01-build-multi-python-conda.sh (recommended)"
echo "      • ./02-build-with-cibuildwheel.sh"
echo ""

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

# Install patchelf if not available (required by auditwheel)
echo "Ensuring patchelf is available..."
if ! command -v patchelf &> /dev/null; then
    echo "Installing patchelf..."
    conda install -y patchelf
else
    echo "patchelf is already installed"
fi

# Build the package
echo "Building source distribution and wheel..."
python -m build

# Fix wheel platform tags for PyPI compatibility
echo "Fixing wheel platform tags..."
if ls dist/*.whl 1> /dev/null 2>&1; then
    echo "Repairing wheels for manylinux compatibility..."
    
    # Create a temporary directory for repaired wheels
    mkdir -p dist/repaired
    
    for wheel in dist/*.whl; do
        if [[ "$wheel" == *"linux_x86_64"* ]]; then
            echo "Repairing wheel: $(basename $wheel)"
            
            # First check what platform tag auditwheel suggests
            echo "Checking wheel compatibility..."
            auditwheel_output=$(auditwheel show "$wheel" 2>&1 || true)
            echo "$auditwheel_output"
            
            # Extract the suggested platform tag from auditwheel output
            suggested_tag=$(echo "$auditwheel_output" | grep -o 'manylinux_[0-9_]*_x86_64' | head -n1 || echo "")
            
            if [[ -n "$suggested_tag" ]]; then
                echo "Using suggested platform tag: $suggested_tag"
                target_plat="$suggested_tag"
            else
                echo "No suggested tag found, trying manylinux_2_17_x86_64"
                target_plat="manylinux_2_17_x86_64"
            fi
            
            # Use auditwheel to repair and bundle libraries
            if auditwheel repair "$wheel" --plat "$target_plat" -w dist/repaired/; then
                echo "Successfully repaired $(basename $wheel) with $target_plat"
                
                # Remove the original unrepaired wheel
                rm "$wheel"
                
                # Move repaired wheel back to dist/
                mv dist/repaired/*.whl dist/
                
                echo "Replaced with repaired wheel"
            else
                echo "ERROR: auditwheel repair failed for $(basename $wheel)"
                echo "Attempting to use auditwheel with more compatible platform..."
                
                # Try manylinux2014 which is more widely compatible
                if auditwheel repair "$wheel" --plat manylinux2014_x86_64 -w dist/repaired/; then
                    echo "Successfully repaired with manylinux2014"
                    rm "$wheel"
                    mv dist/repaired/*.whl dist/
                else
                    echo "WARNING: Could not repair wheel. Manual fix required."
                    echo "You can try:"
                    echo "  1. Build in a manylinux Docker container"
                    echo "  2. Use wheel rename tool to change platform tag"
                    echo "  3. Install older GCC toolchain"
                    echo "  4. Manual repair: auditwheel repair $wheel --plat <compatible-tag> -w dist/"
                fi
            fi
        fi
    done
    
    # Clean up temporary directory
    rm -rf dist/repaired
    
    echo "Wheel repair process completed."
else
    echo "No wheel files found to repair."
fi

# List built packages
echo "====================================="
echo "Built packages:"
ls -la dist/
echo "====================================="

echo "Build complete! Packages are in the dist/ directory."
echo "To upload to test PyPI, run: ./scripts/pypi/10-upload_to_test_pypi.sh"