#!/bin/bash
#SBATCH --job-name=CCC_GPU_UPSET_PLOT
#SBATCH --output=_tmp/CCC_GPU_UPSET_PLOT.%j.out
#SBATCH --error=_tmp/CCC_GPU_UPSET_PLOT.%j.err
#SBATCH --time=48:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cores=4
#SBATCH --mem=350GB
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

python gene_pair_counter.py --data-dir /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections/ ./counts_for_all_tissues.pkl --plot --threads 4
