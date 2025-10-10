#!/bin/bash
# Script to create conda environments for Python 3.10-3.14 for testing
# This script creates separate conda environments without building wheels

set -e  # Exit on error

echo "========================================================================="
echo "Creating conda environments for Python 3.10-3.14 for testing"
echo "========================================================================="
echo ""

# Python versions to build for
PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14")

# Base environment to clone from
BASE_ENV="ccc-gpu"

# Project root directory (assumes script is in scripts/pypi/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Extract package version from pyproject.toml
PACKAGE_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

echo "Project root: $PROJECT_ROOT"
echo "Package version: $PACKAGE_VERSION"
echo "Base environment: $BASE_ENV"
echo "Python versions to create environments for: ${PYTHON_VERSIONS[*]}"
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

# Counter for successfully created environments
CREATED_COUNT=0
FAILED_VERSIONS=()

# Create environments for each Python version
for PY_VERSION in "${PYTHON_VERSIONS[@]}"; do
    echo "========================================================================"
    echo "Creating environment for Python ${PY_VERSION}"
    echo "========================================================================"

    # Environment name
    ENV_NAME="ccc-gpu-py${PY_VERSION/./}"  # e.g., ccc-gpu-py310

    # Check if stable Python version is available (skip RC/alpha/beta)
    echo "Checking if Python ${PY_VERSION} (stable) is available..."
    if ! conda search "python=${PY_VERSION}.*" -c conda-forge 2>/dev/null | \
         grep -E "python\s+${PY_VERSION}\.[0-9]+" | \
         grep -v "rc\|alpha\|beta" | \
         grep -q .; then
        echo "⚠ Warning: Python ${PY_VERSION} stable version not available"
        echo "  Only pre-release versions found, skipping..."
        FAILED_VERSIONS+=("${PY_VERSION} (not stable yet)")
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

    # Activate environment to verify it works
    echo "Verifying environment..."
    conda activate "${ENV_NAME}" || {
        echo "✗ Error: Failed to activate environment '${ENV_NAME}'"
        FAILED_VERSIONS+=("${PY_VERSION} (activation failed)")
        continue
    }

    # Verify Python version
    ACTUAL_PY_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "✓ Python version in environment: ${ACTUAL_PY_VERSION}"

    # Deactivate environment
    conda deactivate

    CREATED_COUNT=$((CREATED_COUNT + 1))
    echo "✓ Environment created successfully"

    echo ""
done

# Summary
echo "========================================================================="
echo "Environment Creation Summary"
echo "========================================================================="
echo "Successfully created environments: ${CREATED_COUNT}/${#PYTHON_VERSIONS[@]}"
echo ""

if [ ${#FAILED_VERSIONS[@]} -gt 0 ]; then
    echo "Failed versions:"
    for VERSION in "${FAILED_VERSIONS[@]}"; do
        echo "  - ${VERSION}"
    done
    echo ""
fi

# List created environments
echo "Created environments:"
for PY_VERSION in "${PYTHON_VERSIONS[@]}"; do
    ENV_NAME="ccc-gpu-py${PY_VERSION/./}"
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  ✓ ${ENV_NAME} (Python ${PY_VERSION})"
    fi
done
echo ""

echo "========================================================================="
echo "Next steps:"
echo "  To activate an environment:"
echo "    conda activate ccc-gpu-py310  # for Python 3.10"
echo "    conda activate ccc-gpu-py311  # for Python 3.11"
echo "    etc."
echo "========================================================================="

if [ ${CREATED_COUNT} -eq 0 ]; then
    echo "✗ No environments were created successfully"
    exit 1
fi

echo "✓ Environment creation complete!"