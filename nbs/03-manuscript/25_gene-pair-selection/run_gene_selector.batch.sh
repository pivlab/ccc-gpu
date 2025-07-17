#!/bin/bash
#SBATCH --job-name=gene_selector
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=gene_selector_%A_%a.out
#SBATCH --error=gene_selector_%A_%a.err
#SBATCH --array=0-19

# Load conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ccc-gpu

# Configuration
DATA_DIR="/path/to/your/data/directory"  # Update this path
TISSUE="liver"  # Update this with your desired tissue
OUTPUT_DIR="/path/to/your/output/directory"  # Update this path
COMBINATION_INDEX=${SLURM_ARRAY_TASK_ID}  # Use array task ID as combination index

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Log job information
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Combination Index: ${COMBINATION_INDEX}"
echo "Data Directory: ${DATA_DIR}"
echo "Tissue: ${TISSUE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Started at: $(date)"

# Run the gene pair selector
# Note: The script will automatically create a subdirectory with combination_name
python gene_pair_selector.py \
    --data-dir "${DATA_DIR}" \
    --tissue "${TISSUE}" \
    --output "${OUTPUT_DIR}" \
    --combination-index ${COMBINATION_INDEX} \
    --sort-by combined

# Check exit status
if [ $? -eq 0 ]; then
    echo "Gene pair selector completed successfully for combination ${COMBINATION_INDEX}"
else
    echo "Gene pair selector failed for combination ${COMBINATION_INDEX}"
    exit 1
fi

echo "Finished at: $(date)" 