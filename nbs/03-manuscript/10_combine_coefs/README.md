# Combine Coefficients CLI Tool

This tool combines correlation coefficients from different methods (CCC-GPU, Pearson, Spearman) into a single dataframe for a specified tissue type and gene selection strategy.

## Features

- **Memory Optimization**: Converts correlation matrices from float32 to float16 for ~50% memory reduction
- **Single Responsibility Principle**: Each class has a clear, focused responsibility
- **Configuration Management**: Centralized configuration using dataclasses
- **Comprehensive Logging**: Detailed logging to both file and console with memory usage tracking
- **Error Handling**: Robust error handling with informative messages
- **Type Hints**: Full type annotations for better code maintainability
- **CLI Interface**: User-friendly command-line interface with argparse

## Usage

### Basic Usage

Run with default parameters (whole blood tissue):
```bash
./combine_coefs.py
```

### Custom Parameters

Specify different tissue and parameters:
```bash
./combine_coefs.py --gtex-tissue brain --gene-selection-strategy var_pc_log2
```

### Advanced Usage

Use custom data directory and logging:
```bash
./combine_coefs.py \
    --data-dir /path/to/your/data \
    --gtex-tissue heart \
    --gene-selection-strategy var_pc_log2 \
    --top-n-genes 5000 \
    --log-level DEBUG \
    --log-dir ./custom_logs
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--top-n-genes` | `"all"` | Number of top genes to process |
| `--data-dir` | `/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8` | Base data directory path |
| `--gtex-tissue` | `"whole_blood"` | GTEx tissue type to process |
| `--gene-selection-strategy` | `"var_pc_log2"` | Gene selection strategy |
| `--log-level` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--log-dir` | `nbs/03-manuscript/10_combine_coefs/logs` | Directory for log files |

## Input Requirements

The tool expects the following input files to exist:
1. Gene expression data: `{data_dir}/gene_selection/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}.pkl`
2. CCC-GPU correlation matrix: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-ccc_gpu.pkl`
3. Pearson correlation matrix: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-pearson.pkl`
4. Spearman correlation matrix: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-spearman.pkl`
5. Gene mapping file: `{data_dir}/gtex_gene_id_symbol_mappings.pkl`

## Output

The tool generates:
- Combined correlation dataframe saved as: `{data_dir}/similarity_matrices/{top_n_genes}/gtex_v8_data_{tissue}-{strategy}-all.pkl`
  - **Memory Optimized**: Data stored in float16 format for ~50% smaller file sizes
- Log file in the specified log directory: `combine_coefs_{tissue}_{strategy}.log`
  - Includes detailed memory usage statistics and optimization information

## Logging

The tool provides comprehensive logging including:
- Input file validation
- Data loading progress with memory usage statistics
- Memory optimization details (float32 → float16 conversion)
- Processing steps for each correlation method
- Index alignment information
- Final output statistics with memory usage
- Error messages with full context

Logs are written to both the console and a file in the logs directory.

## Error Handling

The tool includes robust error handling for:
- Missing input files
- Invalid configuration parameters
- Data processing errors
- Index mismatches between correlation matrices
- File I/O errors

## Design Patterns Used

1. **Dataclass Pattern** - For configuration management
2. **Single Responsibility Principle** - Each class has one clear purpose
3. **Dependency Injection** - Configuration injected into processor
4. **Template Method Pattern** - Consistent processing workflow
5. **Strategy Pattern** - Different correlation methods handled uniformly
6. **Command Pattern** - CLI interface separates invocation from execution

## Dependencies

- pandas
- pathlib (built-in)
- argparse (built-in)
- logging (built-in)
- ccc (custom package with get_upper_triag utility)

## Example Output

```bash
$ ./combine_coefs.py --gtex-tissue whole_blood --log-level INFO
2024-01-01 10:00:00 - __main__ - INFO - Starting combine coefficients CLI tool
2024-01-01 10:00:00 - __main__ - INFO - Configuration: Config(...)
2024-01-01 10:00:01 - __main__ - INFO - All input files validated successfully
2024-01-01 10:00:02 - __main__ - INFO - Loaded 56000 gene mappings
2024-01-01 10:00:03 - __main__ - INFO - Converting ccc_gpu matrix from float32 to float16 for memory optimization
2024-01-01 10:00:03 - __main__ - INFO - Memory usage: 1800.0MB → 900.0MB (saved 900.0MB, 50.0%)
2024-01-01 10:00:04 - __main__ - INFO - Converting pearson matrix from float32 to float16 for memory optimization
2024-01-01 10:00:04 - __main__ - INFO - Memory usage: 1800.0MB → 900.0MB (saved 900.0MB, 50.0%)
2024-01-01 10:00:05 - __main__ - INFO - Successfully combined correlations into DataFrame with shape: (15000000, 3)
2024-01-01 10:00:05 - __main__ - INFO - Final DataFrame dtype: float16, memory usage: 180.0MB
2024-01-01 10:00:06 - __main__ - INFO - Results saved successfully. File size: 190.0MB
2024-01-01 10:00:06 - __main__ - INFO - Memory optimization: Using float16 reduces memory usage by ~50% compared to float32
2024-01-01 10:00:06 - __main__ - INFO - CLI tool completed successfully
```

## Troubleshooting

1. **Import Error**: Ensure the CCC package is properly installed
2. **File Not Found**: Check that all input files exist in the expected locations
3. **Memory Issues**: Consider using a subset of genes if processing large datasets
4. **Permission Errors**: Ensure write permissions for output and log directories
