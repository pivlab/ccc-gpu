#!/bin/bash

# Usage: source ./scripts/reload_cuda_ext.sh

conda activate ccc-gpu

pip uninstall ccc-gpu -y
pip install .
