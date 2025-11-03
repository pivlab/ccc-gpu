# Combine Intersections Counter

Counts and visualizes gene pair intersections across multiple GTEx tissues by accumulating counts from individual tissue intersection files. This is a workaround for not able to use the python upset plot package since it consumes too much memory.

## Overview

This tool processes gene pair intersection datasets from multiple tissues and counts how many gene pairs exist for each unique combination of correlation method indicators (high/low for CCC, Pearson, Spearman). It generates bar plots and saves individual tissue counts.

## Usage

### Process All Tissues

Count intersections across all available tissues:
```bash
python coef_intersection_counter.py \
    --data-dir /path/to/gene_pair_intersections \
    counts_for_all_tissues.pkl \
    --plot
```

### Process Single Tissue

Process only a specific tissue:
```bash
python coef_intersection_counter.py \
    --data-dir /path/to/gene_pair_intersections \
    --tissue whole_blood \
    counts_whole_blood.pkl \
    --plot
```

### Custom Parameters

With custom thread count and output options:
```bash
python coef_intersection_counter.py \
    --data-dir /path/to/gene_pair_intersections \
    output_counts.pkl \
    --plot \
    --threads 8 \
    --top-n 50 \
    --log-file custom_analysis.log
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | Required | Path to directory containing `gene_pair_intersections*.pkl` files |
| `--tissue` | None | Process only a specific tissue (optional) |
| `output_file` | Required | Path to output pickle file for results |
| `--top-n` | `25` | Number of top combinations to display |
| `--log-file` | Auto-generated | Path to log file |
| `--no-display` | False | Skip displaying results to console |
| `--plot` | False | Generate bar plot of results (SVG format) |
| `--threads` | CPU count | Number of threads for parallel processing |

## Input Requirements

Input directory should contain intersection files in the format:
- `gene_pair_intersections-gtex_v8-{tissue}-{strategy}.pkl`

Each file should contain a DataFrame with columns:
- **Boolean indicators**: `Pearson (high)`, `Pearson (low)`, `Spearman (high)`, `Spearman (low)`, `Clustermatch (high)`, `Clustermatch (low)`
- **Numeric values**: `ccc`, `pearson`, `spearman`
- **Multi-index**: Gene pair indices

## Output

For each run, generates files in a timestamped folder:

- **`{tissue}_counts.pkl`**: Individual count files for each processed tissue
- **`{output_file}.pkl`**: Accumulated counts across all processed tissues
- **`{output_file}.svg`**: Bar plot visualization (if `--plot` is used)
- **`{output_file}_analysis.log`**: Detailed processing log
- **`intermediate_counts_after_{n}_files.pkl`**: Intermediate results (multi-tissue mode only)
- **`intermediate_plot_after_{n}_files.svg`**: Intermediate plots (multi-tissue mode only)

## Parallel Processing

The script automatically uses parallel processing for large datasets:
- **Small datasets** (< 50,000 rows): Single-threaded
- **Medium datasets** (50,000 - 500,000 rows): ThreadPoolExecutor
- **Large datasets** (> 500,000 rows): ProcessPoolExecutor

Use `--threads` to specify the number of threads (default: CPU count).

## SLURM Cluster Usage

Process all tissues on a SLURM cluster:
```bash
sbatch run_slurm_job.sh
```

The SLURM script:
- Processes all 54 GTEx tissues
- Uses 350GB memory allocation
- Saves outputs to timestamped log directory
- Generates plots with `--plot` flag

Example command from the script:
```bash
python coef_intersection_counter.py \
    --data-dir /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections/ \
    ./counts_for_all_tissues.pkl \
    --plot \
    --threads 4
```

## Bar Plot Visualization

The generated bar plot shows:
- Gene pair counts for each indicator combination
- UpSet plot-style indicators showing which methods agree (high/low)
- Ordered combinations prioritizing:
  - Full agreements (all methods high/low)
  - Partial agreements (2 methods agree)
  - Disagreements (methods disagree)

Counts are formatted with K/M/B units for readability.

