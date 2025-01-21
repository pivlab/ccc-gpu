#!/bin/bash

# Used to setup the development environment for CCC
# Can be loaded by PyCharm on startup

# Find the conda path
CONDA_PATH=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | awk '{print $1}')
source ${CONDA_PATH}/etc/profile.d/conda.sh

# Activate the conda environment
conda activate ccc-gpu

# Set the PYTHONPATH
export PYTHONPATH=`readlink -f ./libs/`:$PYTHONPATH

# Set the pybind11 path for cmake to find it
# (Locally installed pybind11 by conda can not be found by cmake)
PYBIND11_DIR=$(find ~/ -type d -path "*/anaconda3/pkgs/*/pybind11/share/cmake/pybind11" 2>/dev/null)
export CMAKE_PREFIX_PATH=$PYBIND11_DIR:$CMAKE_PREFIX_PATH

# Set the CUDA_HOME and LD_LIBRARY_PATH
# Set the CUDA_HOME and LD_LIBRARY_PATH
export LD_LIBRARY_PATH="~/anaconda3/envs/ccc-cuda/lib/:$LD_LIBRARY_PATH"
export LIBRARY_PATH="~/anaconda3/envs/ccc-cuda/lib/:$LD_LIBRARY_PATH"
export CUDA_HOME="~/anaconda3/envs/ccc-cuda"
