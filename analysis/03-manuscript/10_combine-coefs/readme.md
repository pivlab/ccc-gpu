# Combine Coefficients CLI Tool

This tool combines correlation coefficients from different methods (CCC-GPU, Pearson, Spearman) into a single dataframe for specified tissues and gene selection strategy.

Note that a high-memory cluster node or local machine with at least 300GB of memory is required to run this script since the internal steps consumes significant memory.

## Usage

### Basic Usage

Process all 54 GTEx tissues:
```bash
python combine_coefs.py --gene-selection-strategy 
```

### Tissue Selection

Use `--include` to process only matching tissues (regex patterns):
```bash
python combine_coefs.py --include brain --gene-selection-strategy 
```

Use `--exclude` to skip matching tissues:
```bash
python combine_coefs.py --exclude brain --exclude cells --gene-selection-strategy 
```

Combine both for complex filtering:
```bash
python combine_coefs.py --include heart --exclude atrial --gene-selection-strategy 
```

### Cache Management

By default, cache files are NOT saved. Enable caching for faster future runs:
```bash
python combine_coefs.py --save-cache --gene-selection-strategy 
```

Clear existing cache before processing:
```bash
python combine_coefs.py --clear-cache --gene-selection-strategy 
```

### Advanced Options

Force reprocessing even if output exists:
```bash
python combine_coefs.py --force --gene-selection-strategy 
```

Keep temporary cache files after completion:
```bash
python combine_coefs.py --no-cleanup --gene-selection-strategy 
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--top-n-genes` | `"all"` | Number of top genes to process |
| `--data-dir` | `/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8` | Base data directory path |
| `--include` | None | Include tissues matching regex pattern (can be used multiple times) |
| `--exclude` | None | Exclude tissues matching regex pattern (can be used multiple times) |
| `--gene-selection-strategy` | `""` | Gene selection strategy (required) |
| `--log-level` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--log-dir` | `logs` | Directory for log files (relative to script location) |
| `--temp-dir` | Auto-generated | Directory for temporary cache files |
| `--save-cache` | False | Save intermediate cache files for faster future runs |
| `--clear-cache` | False | Clear existing cache files before processing |
| `--force` | False | Force reprocessing even if output file already exists |
| `--no-cleanup` | False | Keep temporary cache files after completion |

## Input Requirements

For each selected tissue, the tool expects:
1. Gene expression data: `{data_dir}/gene_selection/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}.pkl`
2. CCC-GPU correlation matrix: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-ccc_gpu.pkl`
3. Pearson correlation matrix: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-pearson.pkl`
4. Spearman correlation matrix: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-spearman.pkl`
5. Gene mapping file: `{data_dir}/gtex_gene_id_symbol_mappings.pkl`

## Output

For each processed tissue:
- Combined correlation dataframe: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-all.pkl`
  - Data stored in float16 format for ~50% smaller file sizes
  - Contains columns: `ccc`, `pearson`, `spearman`
- Timestamped log directory: `{log_dir}/run_{timestamp}/combine_coefs_{tissue_count}tissues_{strategy}.log`
  - Includes processing progress, memory usage, and timing statistics

## Processing Details

- **Alignment**: Spearman correlation is used as the reference for aligning all methods
- **Memory Optimization**: Automatically converts correlation matrices from float32 to float16
- **Skip Existing**: By default, skips tissues with existing output files (use `--force` to override)
- **Progress Tracking**: Shows estimated time remaining and average time per tissue

## Troubleshooting

1. **Import Error**: Ensure the CCC package is properly installed
2. **File Not Found**: Check that all input files exist in the expected locations for each selected tissue
3. **Memory Issues**: Consider using a subset of genes or processing fewer tissues at once
4. **Permission Errors**: Ensure write permissions for output and log directories
