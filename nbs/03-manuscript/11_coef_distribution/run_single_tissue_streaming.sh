#!/bin/bash
#SBATCH --job-name=gtex_single
#SBATCH --output=logs/gtex_single_%j.out
#SBATCH --error=logs/gtex_single_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu

# Single-tissue streaming version for testing or individual runs
# Usage: sbatch --export=TISSUE=whole_blood run_single_tissue_streaming.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# Get tissue from environment variable, default to whole_blood
TISSUE=${TISSUE:-whole_blood}
CHUNK_SIZE=50000
GENE_PAIRS_PERCENT=0.7

# Log the configuration
echo "=========================================="
echo "GTEx Coefficient Analysis (Single Tissue)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Tissue: $TISSUE"
echo "Chunk Size: $CHUNK_SIZE"
echo "Gene Pairs Percent: $GENE_PAIRS_PERCENT"
echo "Memory Limit: ${SLURM_MEM_MB}MB"
echo "Start Time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ccc-gpu

# Run the streaming version
python 11_00-gtex_general_plots_streaming.py \
    --tissue $TISSUE \
    --chunk-size $CHUNK_SIZE \
    --gene-pairs-percent $GENE_PAIRS_PERCENT \
    --log-dir "logs/${TISSUE}_$(date +%Y%m%d_%H%M%S)"

# Log completion
EXIT_CODE=$?
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed with exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE 