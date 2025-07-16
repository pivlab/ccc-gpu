#!/bin/bash
#SBATCH --job-name=gene_pair_selector
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/gene_selector_%A_%a.out
#SBATCH --error=logs/gene_selector_%A_%a.err
#SBATCH --partition=short

# Load conda environment
source ~/.bashrc
conda activate ccc-gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Set input and output paths
INPUT_FILE="/path/to/your/gene_intersection_data.pkl"
OUTPUT_BASE_DIR="/path/to/your/output/directory"

# Create output directory for this combination
OUTPUT_DIR="${OUTPUT_BASE_DIR}/combination_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTPUT_DIR}"

# Run the gene pair selector
python gene_pair_selector.py \
    "${INPUT_FILE}" \
    "${OUTPUT_DIR}" \
    --combination-index "${SLURM_ARRAY_TASK_ID}" \
    --sort-by combined \
    --log-file "${OUTPUT_DIR}/gene_selector_${SLURM_ARRAY_TASK_ID}.log"

# Print completion message
echo "Completed processing combination ${SLURM_ARRAY_TASK_ID}"
echo "Results saved to: ${OUTPUT_DIR}" 