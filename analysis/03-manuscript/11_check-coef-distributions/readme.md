# GTEx Coefficient Distribution Analysis

Memory-efficient script for generating distribution plots and percentile summaries for correlation coefficients (CCC, Pearson, Spearman) across GTEx tissues.

## Overview

This tool processes large correlation coefficient datasets in chunks to avoid out-of-memory errors. It generates:
- **Cumulative histograms**: Shows percentage of gene pairs with coefficients ≤ x
- **Regular histograms**: Distribution plots for each coefficient method
- **Density plots**: Overlaid smooth density curves for all methods
- **Percentiles summary**: CSV file with percentiles (0.00-1.00 in 0.01 steps) for each coefficient

## Usage

### Basic Usage

Process a single tissue:
```bash
python 10-gtex_general_plots_streaming.py \
    --tissue whole_blood \
    --gene-selection-strategy var_pc_log2
```

### Custom Parameters

Specify chunk size and directories:
```bash
python 10-gtex_general_plots_streaming.py \
    --tissue brain_cortex \
    --gene-selection-strategy var_pc_log2 \
    --top-n-genes all \
    --chunk-size 50000 \
    --data-dir /path/to/data \
    --output-dir /path/to/figures \
    --log-dir ./logs/custom_run
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tissue` | `"whole_blood"` | GTEx tissue type to analyze |
| `--top-n-genes` | `"all"` | Number of top genes ('all' for all genes) |
| `--gene-selection-strategy` | `"var_pc_log2"` | Gene selection strategy |
| `--gene-pairs-percent` | `0.70` | Percentage for cumulative histogram |
| `--chunk-size` | `100000` | Number of rows to process per chunk |
| `--data-dir` | `/pividori_lab/haoyu_projects/ccc-gpu/data/gtex` | Base data directory |
| `--output-dir` | `/pividori_lab/haoyu_projects/ccc-gpu/figures` | Base output directory |
| `--log-dir` | Auto-generated | Custom log directory (default: `./logs/streaming_YYYYMMDD_HHMMSS`) |
| `--ccc-label` | `"CCC"` | Label for CCC coefficient |
| `--pearson-label` | `"Pearson"` | Label for Pearson coefficient |
| `--spearman-label` | `"Spearman"` | Label for Spearman coefficient |

## Input Requirements

The script expects a combined correlation file:
- Path: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-all.pkl`
- Format: DataFrame with columns `ccc`, `pearson`, `spearman`

## Output

For each tissue, generates files in `{output_dir}/coefs_comp/gtex_{tissue}/`:

- **`dist-cumulative_histograms.svg`**: Cumulative distribution plot with quantiles
- **`dist-histograms.svg`**: Individual histograms for each coefficient
- **`dist-density.svg`**: Overlaid density curves for all coefficients
- **`percentiles_summary_{tissue}.csv`**: Percentiles (0.00-1.00) for each coefficient

All files are also copied to the log directory.

## SLURM Cluster Usage

For processing multiple tissues on a SLURM cluster, use:
```bash
sbatch 10-run_slurm_streaming.sh
```

The SLURM script processes all 54 GTEx tissues sequentially in a single job with:
- 24-hour time limit
- 300GB memory allocation
- Automatic error handling and progress tracking
- Per-tissue log directories

## Memory Efficiency

- Processes data in configurable chunks (default: 100,000 rows)
- Computes histograms incrementally without loading full dataset
- Automatically frees memory after processing each chunk
- Suitable for very large correlation matrices

