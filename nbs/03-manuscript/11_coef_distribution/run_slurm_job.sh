#!/bin/bash
#SBATCH --job-name=CCC_GPU_Coef_Dist
#SBATCH --output=_tmp/CCC_GPU_Coef_Dist.%j.out
#SBATCH --error=_tmp/CCC_GPU_Coef_Dist.%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cores=2
#SBATCH --mem=400GB

# Create temporary directory for outputs if it doesn't exist
mkdir -p _tmp

python 11_00-gtex_general_plots.py