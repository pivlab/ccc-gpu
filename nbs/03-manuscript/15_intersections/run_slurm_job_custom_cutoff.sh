#!/bin/bash
#SBATCH --job-name=CCC_GPU_Intersections
#SBATCH --output=_tmp/CCC_GPU_Intersections.%j.out
#SBATCH --error=_tmp/CCC_GPU_Intersections.%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cores=8
#SBATCH --mem=512GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haoyu.zhang@cuanschutz.edu

# Create temporary directory for outputs if it doesn't exist
mkdir -p _tmp

# Create timestamped log directory for all tissues
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Created log directory: $LOG_DIR"
echo "All tissue logs will be saved in this directory"

# Define list of GTEx tissues to process
TISSUES=(
    # "adipose_subcutaneous"
    # "adipose_visceral_omentum"
    # "adrenal_gland"
    # "artery_aorta"
    # "artery_coronary"
    # "artery_tibial"
    # "bladder"
    # "brain_amygdala"
    # "brain_anterior_cingulate_cortex_ba24"
    # "brain_caudate_basal_ganglia"
    # "brain_cerebellar_hemisphere"
    # "brain_cerebellum"
    # "brain_cortex"
    # "brain_frontal_cortex_ba9"
    # "brain_hippocampus"
    # "brain_hypothalamus"
    # "brain_nucleus_accumbens_basal_ganglia"
    # "brain_putamen_basal_ganglia"
    # "brain_spinal_cord_cervical_c1"
    # "brain_substantia_nigra"
    # "breast_mammary_tissue"
    # "cells_cultured_fibroblasts"
    # "cells_ebvtransformed_lymphocytes"
    # "cervix_ectocervix"
    # "cervix_endocervix"
    # "colon_sigmoid"
    # "colon_transverse"
    # "esophagus_gastroesophageal_junction"
    # "esophagus_mucosa"
    # "esophagus_muscularis"
    # "fallopian_tube"
    # "heart_atrial_appendage"
    # "heart_left_ventricle"
    # "kidney_cortex"
    # "kidney_medulla"
    # "liver"
    # "lung"
    # "minor_salivary_gland"
    # "muscle_skeletal"
    # "nerve_tibial"
    # "ovary"
    # "pancreas"
    # "pituitary"
    # "prostate"
    # "skin_not_sun_exposed_suprapubic"
    # "skin_sun_exposed_lower_leg"
    # "small_intestine_terminal_ileum"
    # "spleen"
    # "stomach"
    # "testis"
    # "thyroid"
    # "uterus"
    # "vagina"
    "whole_blood",
)

# Get total number of tissues
TOTAL_TISSUES=${#TISSUES[@]}

echo "Starting gene pair intersections analysis for $TOTAL_TISSUES tissues"
echo "Job started at: $(date)"

# Initialize counters
PROCESSED=0
SKIPPED=0
FAILED=0

# Process each tissue
for i in "${!TISSUES[@]}"; do
    TISSUE="${TISSUES[$i]}"
    TISSUE_NUM=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing tissue $TISSUE_NUM/$TOTAL_TISSUES: $TISSUE"
    echo "Started at: $(date)"
    echo "Log file: $LOG_DIR/compute_intersections_${TISSUE}.log"
    echo "=========================================="
    
    # Run compute_intersections.py for this tissue with shared log directory
    if python compute_intersections.py --gtex-tissue "$TISSUE" --custom-low-quantile 0.80 --custom-high-quantile 0.95  --log-dir "$LOG_DIR" ; then
        echo "✓ Successfully processed $TISSUE"
        ((PROCESSED++))
    else
        exit_code=$?
        echo "✗ Failed to process $TISSUE (exit code: $exit_code)"
        ((FAILED++))
        # Continue with other tissues instead of stopping
    fi
    
    echo "Finished $TISSUE at: $(date)"
    
    # Show progress summary
    REMAINING=$((TOTAL_TISSUES - TISSUE_NUM))
    echo "Progress: $PROCESSED processed, $SKIPPED skipped, $FAILED failed, $REMAINING remaining"
done

echo ""
echo "=========================================="
echo "All tissues processing completed!"
echo "Final Summary:"
echo "  Total tissues: $TOTAL_TISSUES"
echo "  Successfully processed: $PROCESSED"
echo "  Skipped (output exists): $SKIPPED"
echo "  Failed: $FAILED"
echo "  All log files saved in: $LOG_DIR"
echo "Job completed at: $(date)"
echo "=========================================="

# Exit with error code if any tissues failed
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED tissues failed to process"
    exit 1
else
    echo "SUCCESS: All tissues processed successfully"
    exit 0
fi
