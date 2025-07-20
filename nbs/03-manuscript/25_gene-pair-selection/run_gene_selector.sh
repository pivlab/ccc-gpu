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

# Define combinations to process
COMBINATIONS=(8 9 10 11 12)

# Define tissues to process (modify as needed)
TISSUES=(
        "adipose_subcutaneous"
        "adipose_visceral_omentum"
        "adrenal_gland"
        "artery_aorta"
        "artery_coronary"
        "artery_tibial"
        "bladder"
        "brain_amygdala"
        "brain_anterior_cingulate_cortex_ba24"
        "brain_caudate_basal_ganglia"
        "brain_cerebellar_hemisphere"
        "brain_cerebellum"
        "brain_cortex"
        "brain_frontal_cortex_ba9"
        "brain_hippocampus"
        "brain_hypothalamus"
        "brain_nucleus_accumbens_basal_ganglia"
        "brain_putamen_basal_ganglia"
        "brain_spinal_cord_cervical_c1"
        "brain_substantia_nigra"
        "breast_mammary_tissue"
        "cells_cultured_fibroblasts"
        "cells_ebvtransformed_lymphocytes"
        "cervix_ectocervix"
        "cervix_endocervix"
        "colon_sigmoid"
        "colon_transverse"
        "esophagus_gastroesophageal_junction"
        "esophagus_mucosa"
        "esophagus_muscularis"
        "fallopian_tube"
        "heart_atrial_appendage"
        "heart_left_ventricle"
        "kidney_cortex"
        "kidney_medulla"
        "liver"
        "lung"
        "minor_salivary_gland"
        "muscle_skeletal"
        "nerve_tibial"
        "ovary"
        "pancreas"
        "pituitary"
        "prostate"
        "skin_not_sun_exposed_suprapubic"
        "skin_sun_exposed_lower_leg"
        "small_intestine_terminal_ileum"
        "spleen"
        "stomach"
        "testis"
        "thyroid"
        "uterus"
        "vagina"
        "whole_blood"
)

# Process each combination across all tissues
for COMBINATION_ID in "${COMBINATIONS[@]}"; do
    echo "=========================================="
    echo "Starting combination ${COMBINATION_ID}"
    echo "=========================================="
    
    for tissue in "${TISSUES[@]}"; do
        echo "Processing tissue: ${tissue}, combination: ${COMBINATION_ID}"
        
        # Run the gene pair selector (it will create tissue_name/combination_name subdirectories)
        python gene_pair_selector.py \
            --data-dir "${DATA_DIR}" \
            --tissue "${tissue}" \
            --output "${OUTPUT_BASE_DIR}" \
            --combination-index "${COMBINATION_ID}" \
            --sort-by combined
        
        # Check if processing was successful
        if [ $? -eq 0 ]; then
            echo "Successfully processed ${tissue} with combination ${COMBINATION_ID}"
        else
            echo "Error processing ${tissue} with combination ${COMBINATION_ID}"
        fi
    done
    
    echo "Completed all tissues for combination ${COMBINATION_ID}"
    echo ""
done

echo "=========================================="
echo "Completed processing all combinations: ${COMBINATIONS[*]}"
echo "Total jobs: $((${#COMBINATIONS[@]} * ${#TISSUES[@]}))"
echo "==========================================" 