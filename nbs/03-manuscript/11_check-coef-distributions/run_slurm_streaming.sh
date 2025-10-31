#!/bin/bash
#SBATCH --job-name=gtex_all_tissues
#SBATCH --output=logs/gtex_all_tissues_%j.out
#SBATCH --error=logs/gtex_all_tissues_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cores=2
#SBATCH --mem=300GB

# GTEx Coefficient Analysis - Multi-Tissue Streaming with Loop
# This script processes all 54 GTEx tissues sequentially in a single job

# Create logs directory if it doesn't exist
mkdir -p logs

# All GTEx tissues to process
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

# Configuration
CHUNK_SIZE=50000
GENE_PAIRS_PERCENT=0.7
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Job info
echo "============================================================"
echo "GTEx Coefficient Analysis - All Tissues Streaming"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Total Tissues: ${#TISSUES[@]}"
echo "Chunk Size: $CHUNK_SIZE"
echo "Gene Pairs Percent: $GENE_PAIRS_PERCENT"
echo "Memory Limit: ${SLURM_MEM_MB}MB"
echo "Time Limit: ${SLURM_TIMELIMIT}"
echo "============================================================"

# Activate conda environment
eval "$(mamba shell hook --shell bash)"
mamba activate ccc-gpu

# Process statistics
PROCESSED_COUNT=0
FAILED_COUNT=0
FAILED_TISSUES=()

# Process each tissue in the loop
for TISSUE in "${TISSUES[@]}"; do
    TISSUE_START_TIME=$(date)
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    
    echo ""
    echo "🧬 Processing tissue $PROCESSED_COUNT/${#TISSUES[@]}: $TISSUE"
    echo "⏰ Start time: $TISSUE_START_TIME"
    echo "----------------------------------------------------------"
    
    # Run the streaming version for this tissue
    python 11_00-gtex_general_plots_streaming.py \
        --tissue "$TISSUE" \
        --chunk-size $CHUNK_SIZE \
        --gene-pairs-percent $GENE_PAIRS_PERCENT \
        --log-dir "logs/${TISSUE}_${TIMESTAMP}"
    
    # Check exit code
    EXIT_CODE=$?
    TISSUE_END_TIME=$(date)
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ SUCCESS: $TISSUE completed successfully"
        echo "⏰ End time: $TISSUE_END_TIME"
    else
        echo "❌ FAILED: $TISSUE failed with exit code $EXIT_CODE"
        echo "⏰ End time: $TISSUE_END_TIME"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_TISSUES+=("$TISSUE")
    fi
    
    echo "----------------------------------------------------------"
    echo "📊 Progress: $PROCESSED_COUNT/${#TISSUES[@]} tissues processed"
    echo "✅ Successful: $((PROCESSED_COUNT - FAILED_COUNT))"
    echo "❌ Failed: $FAILED_COUNT"
    echo ""
done

# Final summary
TOTAL_END_TIME=$(date)
SUCCESS_COUNT=$((PROCESSED_COUNT - FAILED_COUNT))

echo "============================================================"
echo "🎯 FINAL SUMMARY"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Total Runtime: Start $(date) to End $TOTAL_END_TIME"
echo ""
echo "📊 Results:"
echo "  Total tissues: ${#TISSUES[@]}"
echo "  Successfully processed: $SUCCESS_COUNT"
echo "  Failed: $FAILED_COUNT"
echo "  Success rate: $(( SUCCESS_COUNT * 100 / ${#TISSUES[@]} ))%"
echo ""

if [ $FAILED_COUNT -gt 0 ]; then
    echo "❌ Failed tissues:"
    for FAILED_TISSUE in "${FAILED_TISSUES[@]}"; do
        echo "  - $FAILED_TISSUE"
    done
    echo ""
fi

echo "📁 Output locations:"
echo "  - Job log: logs/gtex_all_tissues_${SLURM_JOB_ID}.out"
echo "  - Analysis logs: logs/<tissue>_${TIMESTAMP}/"
echo "  - Figures: [output-dir]/coefs_comp/gtex_<tissue>/"
echo ""
echo "💡 Next steps:"
echo "  - Check individual tissue logs: ls -la logs/*_${TIMESTAMP}/"
echo "  - View generated plots: ls -la [output-dir]/coefs_comp/gtex_*/"
echo "============================================================"

# Exit with error if any tissues failed
if [ $FAILED_COUNT -gt 0 ]; then
    echo "Job completed with $FAILED_COUNT failures"
    exit 1
else
    echo "🎉 All tissues processed successfully!"
    exit 0
fi 