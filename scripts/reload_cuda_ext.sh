#!/bin/bash

# Usage: source ./scripts/reload_cuda_ext.sh

# Define variables
CONDA_ENV_NAME="ccc-gpu"
PACKAGE_NAME="cccgpu"

# Define colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a package is installed
is_package_installed() {
    python -c "import pkg_resources; print(pkg_resources.get_distribution('${PACKAGE_NAME}').version)" 2>/dev/null
    return $?
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}" >&2
    return 1
fi

# Check if the environment exists
if ! conda env list | grep -q "^${CONDA_ENV_NAME}\s"; then
    echo -e "${RED}Error: conda environment '${CONDA_ENV_NAME}' does not exist${NC}" >&2
    return 1
fi

# Activate the environment
echo -e "${BLUE}Activating conda environment '${CONDA_ENV_NAME}'...${NC}" >&2
conda activate ${CONDA_ENV_NAME}

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed or not in PATH${NC}" >&2
    return 1
fi

# Check if package is installed before uninstalling
if is_package_installed >/dev/null; then
    echo -e "${BLUE}Uninstalling existing ${PACKAGE_NAME} package...${NC}" >&2
    pip uninstall ${PACKAGE_NAME} -y
else
    echo -e "${BLUE}${PACKAGE_NAME} is not currently installed${NC}" >&2
fi

# Install the package
echo -e "${BLUE}Installing ${PACKAGE_NAME} package...${NC}" >&2
pip install .

# Verify installation
if is_package_installed >/dev/null; then
    echo -e "${GREEN}Done!${NC}" >&2
else
    echo -e "${RED}Error: Failed to install ${PACKAGE_NAME}${NC}" >&2
    return 1
fi
