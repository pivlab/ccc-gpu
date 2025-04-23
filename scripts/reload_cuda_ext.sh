#!/bin/bash

# Usage: source ./scripts/reload_cuda_ext.sh

# Define colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    return 1
fi

# Check if the environment exists
if ! conda env list | grep -q "^ccc-gpu\s"; then
    echo -e "${RED}Error: conda environment 'ccc-gpu' does not exist${NC}"
    return 1
fi

# Activate the environment
echo -e "${BLUE}Activating conda environment 'ccc-gpu'...${NC}"
conda activate ccc-gpu

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed or not in PATH${NC}"
    return 1
fi

# Check if ccc-gpu is installed before uninstalling
if pip list | grep -q "^ccc-gpu\s"; then
    echo -e "${BLUE}Uninstalling existing ccc-gpu package...${NC}"
    pip uninstall ccc-gpu -y
else
    echo -e "${BLUE}ccc-gpu is not currently installed${NC}"
fi

# Install the package
echo -e "${BLUE}Installing ccc-gpu package...${NC}"
pip install .

# Verify installation
if pip list | grep -q "^ccc-gpu\s"; then
    echo -e "${GREEN}Successfully installed ccc-gpu${NC}"
else
    echo -e "${RED}Error: Failed to install ccc-gpu${NC}"
    return 1
fi
