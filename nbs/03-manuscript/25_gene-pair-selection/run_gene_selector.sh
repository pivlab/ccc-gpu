#!/bin/bash
#SBATCH --job-name=CCC_GPU_GENE_SELECTOR
#SBATCH --output=logs/CCC_GPU_GENE_SELECTOR.%j.out.log
#SBATCH --error=logs/CCC_GPU_GENE_SELECTOR.%j.err.log
#SBATCH --time=48:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cores=4
#SBATCH --mem=350GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haoyu.zhang@cuanschutz.edu

# Load conda environment
# eval "$(mamba shell hook --shell bash)"
# mamba activate ccc-gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths
DATA_DIR="/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections"
OUTPUT_BASE_DIR="/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection"
CATEGORY_ID=10

# Define tissues to process (modify as needed)
TISSUES=(
    # "liver"
    # "lung"
    # "brain_cortex"
    # "heart_left_ventricle"
    # "muscle_skeletal"
    # "adipose_subcutaneous"
    # "kidney_cortex"
    # "testis"
    # "thyroid"
    "whole_blood"
)

# Process each tissue with the current combination index
for tissue in "${TISSUES[@]}"; do
    # Create output directory for this tissue and combination
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${tissue}_combination_${CATEGORY_ID}"
    mkdir -p "${OUTPUT_DIR}"
    
    echo "Processing tissue: ${tissue}, combination: ${CATEGORY_ID}"
    
    # Run the gene pair selector
    python gene_pair_selector.py \
        --data-dir "${DATA_DIR}" \
        --tissue "${tissue}" \
        --output "${OUTPUT_DIR}" \
        --combination-index "${CATEGORY_ID}" \
        --sort-by combined \
        --log-file "${OUTPUT_DIR}/${tissue}_Combination_${CATEGORY_ID}.log"
    
    # Check if processing was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed ${tissue} with combination ${CATEGORY_ID}"
    else
        echo "Error processing ${tissue} with combination ${CATEGORY_ID}"
    fi
done

echo "Completed processing all tissues for combination ${CATEGORY_ID}" 