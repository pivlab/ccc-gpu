# #!/bin/bash
#SBATCH --job-name=CCC_GPU_Intersections
#SBATCH --output=_tmp/CCC_GPU_Intersections.%j.out
#SBATCH --error=_tmp/CCC_GPU_Intersections.%j.err
#SBATCH --time=03:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=512B
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haoyu.zhang@cuanschutz.edu

python compute_intersections.py
