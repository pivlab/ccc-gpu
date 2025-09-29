#!/bin/bash
# Script to build cccgpu wheels for multiple Python versions using conda environments
# This script creates separate conda environments for Python 3.10-3.15 and builds wheels in each

set -e  # Exit on error

echo "========================================================================="
echo "Building cccgpu wheels for Python 3.10-3.15 using conda environments"
echo "========================================================================="
echo ""

# Python versions to build for
PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14" "3.15")

# Base environment to clone from
BASE_ENV="ccc-gpu"

# Project root directory (assumes script is in scripts/pypi/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Base environment: $BASE_ENV"
echo "Python versions to build: ${PYTHON_VERSIONS[*]}"
echo ""

# Initialize conda for bash
echo "Initializing conda..."
if eval "$(conda shell.bash hook)" 2>/dev/null; then
    echo "✓ Conda initialized"
elif eval "$(mamba shell.bash hook)" 2>/dev/null; then
    echo "✓ Mamba initialized"
else
    echo "✗ Error: Could not initialize conda or mamba"
    echo "  Please ensure conda/mamba is installed and in PATH"
    exit 1
fi

# Check if base environment exists
if ! conda env list | grep -q "^${BASE_ENV} "; then
    echo "✗ Error: Base environment '${BASE_ENV}' not found"
    echo "  Please ensure the ccc-gpu environment exists"
    conda env list
    exit 1
fi

echo "✓ Base environment '${BASE_ENV}' found"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info libs/ccc/*.egg-info
mkdir -p dist
echo "✓ Build directories cleaned"
echo ""

# Counter for successfully built wheels
BUILT_COUNT=0
FAILED_VERSIONS=()

# Build wheels for each Python version
for PY_VERSION in "${PYTHON_VERSIONS[@]}"; do
    echo "========================================================================"
    echo "Building wheel for Python ${PY_VERSION}"
    echo "========================================================================"

    # Environment name
    ENV_NAME="ccc-gpu-py${PY_VERSION/./}"  # e.g., ccc-gpu-py310

    # Check if Python version is available in conda
    echo "Checking if Python ${PY_VERSION} is available..."
    if ! conda search "python=${PY_VERSION}*" -c conda-forge | grep -q "python"; then
        echo "⚠ Warning: Python ${PY_VERSION} not found in conda-forge"
        echo "  Skipping this version..."
        FAILED_VERSIONS+=("${PY_VERSION} (not available)")
        echo ""
        continue
    fi

    # Create or update conda environment
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '${ENV_NAME}' already exists"
        echo "Do you want to recreate it? (y/N)"
        read -t 5 -r RECREATE || RECREATE="n"
        if [[ $RECREATE =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n "${ENV_NAME}" -y
            echo "Creating new environment..."
            conda create -n "${ENV_NAME}" -c nvidia -c conda-forge python="${PY_VERSION}" cuda=12.5 -y
        else
            echo "Using existing environment"
        fi
    else
        echo "Creating new environment '${ENV_NAME}'..."
        conda create -n "${ENV_NAME}" -c nvidia -c conda-forge python="${PY_VERSION}" cuda=12.5 -y
    fi

    echo "✓ Environment '${ENV_NAME}' ready"

    # Activate environment and install dependencies
    echo "Installing build dependencies..."
    conda activate "${ENV_NAME}" || {
        echo "✗ Error: Failed to activate environment '${ENV_NAME}'"
        FAILED_VERSIONS+=("${PY_VERSION} (activation failed)")
        continue
    }

    # Verify Python version
    ACTUAL_PY_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "Python version in environment: ${ACTUAL_PY_VERSION}"

    # Install all build dependencies from pyproject.toml [build-system] requires
    # This is required when using --no-isolation
    echo "Installing all build dependencies..."
    pip install --upgrade pip setuptools wheel build || {
        echo "✗ Error: Failed to install pip/setuptools/wheel"
        FAILED_VERSIONS+=("${PY_VERSION} (pip install failed)")
        conda deactivate
        continue
    }

    # Install exact dependencies from pyproject.toml
    echo "Installing build-system.requires dependencies..."
    pip install "scikit-build-core>=0.10" "pybind11>=2.11.0" "cmake>=3.15" ninja "setuptools>=42" wheel || {
        echo "✗ Error: Failed to install build requirements"
        FAILED_VERSIONS+=("${PY_VERSION} (build requirements failed)")
        conda deactivate
        continue
    }

    # Verify critical build tools are available
    echo "Verifying build dependencies..."
    python -c "import scikit_build_core" || {
        echo "✗ Error: scikit-build-core not importable"
        FAILED_VERSIONS+=("${PY_VERSION} (scikit-build-core import failed)")
        conda deactivate
        continue
    }

    which cmake || echo "⚠ Warning: cmake not in PATH"
    which ninja || echo "⚠ Warning: ninja not in PATH"
    cmake --version || echo "⚠ Warning: cmake not working"
    ninja --version || echo "⚠ Warning: ninja not working"

    # Install runtime dependencies (so auditwheel can check imports)
    echo "Installing runtime dependencies..."
    pip install numpy scipy numba pandas scikit-learn || {
        echo "⚠ Warning: Failed to install runtime dependencies (continuing anyway)"
    }

    # Install auditwheel for wheel repair (optional but recommended)
    echo "Installing auditwheel..."
    pip install auditwheel || {
        echo "⚠ Warning: auditwheel not installed (wheel may not be portable)"
    }

    echo "✓ All dependencies installed and verified"

    # Set CUDA environment variables (conda CUDA location)
    echo "Setting CUDA environment variables..."
    export CUDA_HOME="${CONDA_PREFIX}"
    export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
    export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=${CONDA_PREFIX}/bin/nvcc"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

    echo "CUDA_HOME: ${CUDA_HOME}"
    echo "CUDACXX: ${CUDACXX}"
    which nvcc || echo "Warning: nvcc not found in PATH"
    nvcc --version || echo "Warning: could not run nvcc"

    # Build the wheel using --no-isolation to access conda packages
    echo "Building wheel for Python ${PY_VERSION}..."
    if python -m build --wheel --no-isolation --outdir dist/; then
        echo "✓ Wheel built successfully"

        # Find the built wheel
        WHEEL_FILE=$(ls -t dist/cccgpu-*-cp${PY_VERSION/./}-*.whl 2>/dev/null | head -1)

        if [ -n "$WHEEL_FILE" ]; then
            echo "Wheel file: ${WHEEL_FILE}"

            # Try to repair wheel with auditwheel
            if command -v auditwheel &> /dev/null; then
                echo "Checking wheel with auditwheel..."
                if auditwheel show "${WHEEL_FILE}"; then
                    echo "Attempting to repair wheel..."
                    # Try to repair, but don't fail if it doesn't work
                    if auditwheel repair "${WHEEL_FILE}" --plat manylinux_2_17_x86_64 -w dist/ 2>/dev/null; then
                        echo "✓ Wheel repaired successfully"
                        # Remove the old unrepaired wheel
                        rm "${WHEEL_FILE}"
                    else
                        echo "⚠ Could not repair wheel (keeping original)"
                    fi
                fi
            fi

            BUILT_COUNT=$((BUILT_COUNT + 1))
        else
            echo "✗ Error: Wheel file not found in dist/"
            FAILED_VERSIONS+=("${PY_VERSION} (wheel not found)")
        fi
    else
        echo "✗ Error: Build failed for Python ${PY_VERSION}"
        FAILED_VERSIONS+=("${PY_VERSION} (build failed)")
    fi

    # Deactivate environment
    conda deactivate

    echo ""
done

# Summary
echo "========================================================================="
echo "Build Summary"
echo "========================================================================="
echo "Successfully built wheels: ${BUILT_COUNT}/${#PYTHON_VERSIONS[@]}"
echo ""

if [ ${#FAILED_VERSIONS[@]} -gt 0 ]; then
    echo "Failed versions:"
    for VERSION in "${FAILED_VERSIONS[@]}"; do
        echo "  - ${VERSION}"
    done
    echo ""
fi

# List built wheels
echo "Built wheels in dist/:"
ls -lh dist/*.whl 2>/dev/null || echo "  (no wheels found)"
echo ""

echo "========================================================================="
echo "Next steps:"
echo "  1. List wheels: ./scripts/pypi/03-list-built-wheels.sh"
echo "  2. Test wheels manually (see scripts/pypi/README.md)"
echo "  3. Upload: ./scripts/pypi/10-upload_to_test_pypi.sh"
echo "========================================================================="

if [ ${BUILT_COUNT} -eq 0 ]; then
    echo "✗ No wheels were built successfully"
    exit 1
fi

echo "✓ Build complete!"