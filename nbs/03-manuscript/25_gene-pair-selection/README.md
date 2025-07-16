# Gene Pair Selector

A command-line tool for selecting and filtering gene pairs based on boolean indicator combinations and sorting by CCC (Clustered Correlation Coefficient) distance to Pearson and Spearman correlations.

## Overview

This script processes gene intersection files containing multi-index gene pairs with boolean indicators (high/low classifications for Pearson, Spearman, and Clustermatch) and numeric correlation values. It filters data based on predefined combinations and sorts by CCC distance metrics.

## Features

- **Robust Data Handling**: Handles overflow, infinite values, and NaN values gracefully
- **Tissue-Specific Processing**: Supports 53 GTEx tissues with automatic file discovery
- **Predefined Combinations**: 20 predefined boolean indicator combinations for batch processing
- **Multiple Distance Metrics**: Combined, mean, and max distance calculations
- **Comprehensive Logging**: Detailed logging of all operations and data quality issues
- **Batch Processing**: Designed for SLURM job arrays

## Data Format

The input data should be a pickle file containing a pandas DataFrame with:

### Multi-Index
- Two levels representing gene pairs (e.g., Gene1, Gene2)

### Boolean Columns
- `Pearson (high)`, `Pearson (low)`
- `Spearman (high)`, `Spearman (low)`
- `Clustermatch (high)`, `Clustermatch (low)`

### Numeric Columns
- `ccc`: Clustered Correlation Coefficient values
- `pearson`: Pearson correlation values
- `spearman`: Spearman correlation values

## Installation

No additional installation required beyond standard Python scientific packages:

```bash
conda activate ccc-gpu  # or your preferred environment
```

Required packages:
- pandas
- numpy
- pickle (standard library)
- logging (standard library)
- argparse (standard library)

## Usage

### Basic Usage

```bash
python gene_pair_selector.py \
    --data-dir /path/to/data \
    --tissue liver \
    --output /path/to/output \
    --combination-index 0
```

### List Available Options

```bash
# List all available tissues
python gene_pair_selector.py --list-tissues

# List all predefined combinations
python gene_pair_selector.py --list-combinations
```

### Arguments

- `--data-dir`: Directory containing gene pair intersection files
- `--tissue`: Tissue to analyze (from 53 available GTEx tissues)
- `--output`: Output directory for results
- `--combination-index`: Index of predefined combination to use (0-19)
- `--sort-by`: Distance metric to sort by (`combined`, `mean`, `max`). Default: `combined`
- `--log-file`: Path to log file (optional, defaults to timestamped file in logs/ folder)
- `--list-tissues`: List available tissues and exit
- `--list-combinations`: List predefined combinations and exit

### Available Tissues

The script supports 53 GTEx tissues including:
- adipose_subcutaneous, adipose_visceral_omentum
- brain_cortex, brain_cerebellum, brain_hippocampus
- liver, lung, heart_left_ventricle
- muscle_skeletal, whole_blood
- And many more...

### Predefined Combinations

20 predefined combinations are available (index 0-19) representing different boolean indicator patterns:

- **Index 0**: `[False, False, False, True, True, True]` - High agreement across all methods
- **Index 1**: `[False, False, False, False, True, True]` - Pearson/Clustermatch high agreement
- **Index 7**: `[True, True, True, False, False, False]` - Full low agreement
- And more...

Use `--list-combinations` to see all available combinations.

## Overflow Handling

The script includes robust handling for numerical overflow issues:

### Data Cleaning Features
- **Automatic Type Conversion**: Converts all numeric data to float64 for better precision
- **Infinite Value Handling**: Replaces `inf` and `-inf` values with NaN
- **Extreme Value Capping**: Caps values beyond 3× IQR from 5th/95th percentiles
- **NaN Filtering**: Removes rows with NaN values in required columns before calculations

### Distance Calculation Improvements
- **Robust Operations**: Uses numpy operations for better numerical stability
- **Overflow Detection**: Monitors for infinite values during calculations
- **Data Validation**: Verifies data integrity at each step

### Warning Suppression
- Suppresses overflow warnings during controlled operations
- Maintains detailed logging of data quality issues
- Provides comprehensive statistics on data cleaning operations
- Automatically creates timestamped log files in the logs/ directory

## Output Files

The script generates several output files:

1. **filtered_data_cache.pkl**: Data filtered by chosen combination
2. **sorted_data_cache.pkl**: Data sorted by CCC distance
3. **selection_metadata.pkl**: Metadata about the selection process
4. **logs/gene_pair_selector_YYYYMMDD_HHMMSS.log**: Timestamped log file

## Batch Processing with SLURM

Use the provided sbatch script for batch processing:

```bash
# Edit the sbatch script to set your paths
vim run_gene_selector.sbatch

# Submit job array for all 20 combinations
sbatch run_gene_selector.sbatch
```

The sbatch script:
- Runs all 20 combinations in parallel
- Uses SLURM array jobs (0-19)
- Automatically activates conda environment
- Provides proper error handling and logging

## Distance Metrics

The script calculates three distance metrics:

1. **Combined Distance**: `|ccc - pearson| + |ccc - spearman|`
2. **Mean Distance**: `(|ccc - pearson| + |ccc - spearman|) / 2`
3. **Max Distance**: `max(|ccc - pearson|, |ccc - spearman|)`

## Examples

### Process liver tissue with combination 0
```bash
python gene_pair_selector.py \
    --data-dir /data/intersections \
    --tissue liver \
    --output /results/liver_combo_0 \
    --combination-index 0 \
    --sort-by combined
```

### Process brain cortex with combination 5
```bash
python gene_pair_selector.py \
    --data-dir /data/intersections \
    --tissue brain_cortex \
    --output /results/brain_combo_5 \
    --combination-index 5 \
    --sort-by mean
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure the data directory contains files matching the pattern `*{tissue}*.pkl`
2. **Overflow Errors**: The script now handles these automatically with robust data cleaning
3. **Memory Issues**: Use adequate memory allocation (32GB recommended for large datasets)
4. **Conda Environment**: Ensure proper conda environment activation in sbatch scripts

### Data Quality Issues

The script automatically handles:
- Infinite values in correlation data
- NaN values in any column
- Extreme values that cause overflow
- Type conversion issues

All data quality issues are logged with detailed statistics.

## Performance

- **Memory Usage**: ~32GB recommended for large datasets
- **Processing Time**: Varies by dataset size and combination complexity
- **Scalability**: Designed for parallel processing with SLURM job arrays

## Testing

A comprehensive test suite is included:

```bash
# Run overflow handling tests
python test_overflow_fixes.py
```

The test creates synthetic data with problematic values (inf, NaN, extreme values) and verifies robust handling.

## Version History

- **v1.0**: Initial version with basic functionality
- **v1.1**: Added non-interactive mode and predefined combinations
- **v1.2**: Added tissue-specific processing
- **v1.3**: Added robust overflow handling and data cleaning (current)

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Use `--list-tissues` and `--list-combinations` for available options
3. Run the test suite to verify functionality
4. Review data quality warnings in the log files 