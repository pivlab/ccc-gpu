# Compute Coefficient Intersections

Analyzes gene pair intersections across correlation methods (CCC, Pearson, Spearman) to identify agreements and disagreements in high/low correlation gene pairs.

## Overview

This tool identifies gene pairs with high or low correlations for each method, computes their intersections, and generates UpSet plots to visualize agreement patterns between methods.

## Usage

### Basic Usage

Process a single tissue with default quantile thresholds:
```bash
python 05_compute_intersections.py \
    --gtex-tissue whole_blood
```

### Using Tissue-Specific Thresholds

Use pre-computed tissue-specific thresholds from threshold files:
```bash
python 05_compute_intersections.py \
    --gtex-tissue brain_cortex \
    --tissue-threshold-dir /path/to/tissue_thresholds \
    --low-percentile 0.80 \
    --high-percentile 0.95
```

### Regenerate Plots from Existing Data

Load existing intersection data to regenerate plots:
```bash
python 05_compute_intersections.py \
    --gtex-tissue whole_blood \
    --use-existing
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gtex-tissue` | `"whole_blood"` | GTEx tissue to analyze |
| `--gene-sel-strategy` | `""` | Gene selection strategy |
| `--top-n-genes` | `"all"` | Number of top genes |
| `--data-dir` | `/pividori_lab/haoyu_projects/ccc-gpu/data/gtex/` | Path to data directory |
| `--q-diff` | `0.30` | Default threshold (quantile difference from extremes) |
| `--tissue-threshold-dir` | None | Directory with tissue-specific threshold files |
| `--low-percentile` | `0.30` | Percentile for low threshold (when using tissue thresholds) |
| `--high-percentile` | `0.70` | Percentile for high threshold (when using tissue thresholds) |
| `--output-dir` | `/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections` | Output directory |
| `--use-existing` | False | Load existing intersection.pkl to regenerate plots |
| `--log-dir` | Auto-generated | Custom log directory (default: timestamped) |

## Threshold Methods

### Default Quantile Thresholds

Uses quantiles from the data itself:
- Lower threshold: `q_diff` quantile (default: 30th percentile)
- Upper threshold: `1 - q_diff` quantile (default: 70th percentile)

### Tissue-Specific Thresholds

Uses pre-computed thresholds from files (format: `{tissue}-null_coefs_percentiles.pkl`):
- Loads thresholds from specified percentiles (e.g., 80th for low, 95th for high)
- Provides tissue-specific cutoff values for more accurate analysis

## Input Requirements

- Combined correlation file: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-all.pkl`
  - Must contain columns: `ccc`, `pearson`, `spearman`
- (Optional) Tissue threshold files: `{tissue_threshold_dir}/{tissue}-null_coefs_percentiles.pkl`

## Output

For each tissue:
- **`upsetplot_gtex_{tissue}_full.svg`**: Full UpSet plot with all intersections
- **`upsetplot_gtex_{tissue}_trimmed.svg`**: Trimmed UpSet plot with selected combinations
- **`gene_pair_intersections-gtex_v8-{tissue}-{strategy}.pkl`**: Pickle file with intersection data

All outputs are saved to both the output directory and the log directory.

## SLURM Cluster Usage

### Default Thresholds

Process all 54 GTEx tissues with default quantile thresholds:
```bash
sbatch run_slurm_job.sh
```

### Custom Cutoffs

Process all tissues with tissue-specific thresholds:
```bash
sbatch run_slurm_job_custom_cutoff.sh
```

The custom cutoff script uses:
- Low percentile: 80th
- High percentile: 95th
- Tissue threshold directory: `/pividori_lab/haoyu_projects/ccc-gpu/data/gtex/tissue_thresholds`

Both SLURM scripts:
- Process all 54 GTEx tissues sequentially
- Use 512GB memory allocation
- Save logs to timestamped directory

## UpSet Plot Categories

The plots show intersections for:
- **CCC (high/low)**: Clustermatch correlation thresholds
- **Pearson (high/low)**: Pearson correlation thresholds
- **Spearman (high/low)**: Spearman correlation thresholds

The trimmed plot focuses on meaningful combinations:
- Full agreements (all methods high/low)
- Partial agreements (2 methods agree)
- Disagreements (methods disagree)

