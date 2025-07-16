# Gene Pair Selector

This script processes a gene intersection file and allows users to select from predefined combinations of boolean indicators, then filters and sorts the data based on CCC distance to Pearson and Spearman values. **Designed for batch processing and sbatch job submission.**

## Features

- **Non-interactive mode**: Perfect for sbatch job submission
- **Predefined combinations**: 20 pre-selected combinations for systematic analysis
- **CCC distance calculation**: Measures distance between CCC and Pearson/Spearman values
- **Multiple sorting options**: Combined, mean, or max distance metrics
- **Comprehensive caching**: Saves filtered and sorted results
- **Detailed logging**: Complete processing logs and metadata

## Usage

### Basic Usage
```bash
python gene_pair_selector.py input_file.pkl output_dir --combination-index INDEX
```

### List Available Combinations
```bash
python gene_pair_selector.py input_file.pkl output_dir --list-combinations
```

### Full Example
```bash
python gene_pair_selector.py intersection_data.pkl ./output/ --combination-index 0 --sort-by combined
```

## Arguments

### Required Arguments
- `input_file`: Path to input pickle file containing gene pair intersections
- `output_dir`: Directory to save output files
- `--combination-index`: Index of predefined combination to use (0-19)

### Optional Arguments
- `--sort-by {combined,mean,max}`: Distance metric to sort by (default: combined)
- `--log-file`: Path to log file (default: gene_pair_selector.log)
- `--list-combinations`: List all available predefined combinations and exit

## Predefined Combinations

The script uses 20 predefined combinations in the format:
`[Spearman(low), Pearson(low), Clustermatch(low), Spearman(high), Pearson(high), Clustermatch(high)]`

```
Index  Spearman(L)  Pearson(L)  Clustermatch(L)  Spearman(H)  Pearson(H)  Clustermatch(H)
0      False        False       False            True         True        True
1      False        False       False            False        True        True
2      False        False       False            True         False       True
3      False        False       False            True         True        False
4      False        True        True             False        False       False
5      True         False       True             False        False       False
6      True         True        False            False        False       False
7      True         True        True             False        False       False
8      False        True        False            False        False       True
9      True         False       False            False        False       True
10     True         True        False            False        False       True
11     True         False       False            False        True        True
12     False        True        False            True         False       True
13     False        False       True             False        True        False
14     True         False       True             False        True        False
15     True         False       False            False        True        False
16     False        False       True             True         True        False
17     False        False       True             True         False       False
18     False        True        True             True         False       False
19     False        True        False            True         False       False
```

## Expected Input Data Structure

The input pickle file should contain a pandas DataFrame with:

### Multi-index
- Gene pairs as multi-index (e.g., gene1, gene2)

### Boolean Columns
- `Pearson (high)`, `Pearson (low)`
- `Spearman (high)`, `Spearman (low)`
- `Clustermatch (high)`, `Clustermatch (low)`

### Numeric Columns
- `ccc`: Clustermatch correlation coefficient values
- `pearson`: Pearson correlation coefficient values
- `spearman`: Spearman correlation coefficient values

## Output Files

The script generates the following files in the output directory:

1. **`filtered_data_cache.pkl`**: Gene pairs filtered by chosen combination
2. **`sorted_data_cache.pkl`**: Filtered data sorted by CCC distance
3. **`selection_metadata.pkl`**: Metadata about the selection process
4. **`gene_pair_selector.log`**: Detailed processing log

## Distance Metrics

The script calculates several distance metrics:

- **`ccc_pearson_diff`**: Absolute difference between CCC and Pearson values
- **`ccc_spearman_diff`**: Absolute difference between CCC and Spearman values
- **`ccc_combined_distance`**: Sum of both differences
- **`ccc_mean_distance`**: Mean of both differences
- **`ccc_max_distance`**: Maximum of both differences

## Batch Processing Examples

### Single Job
```bash
python gene_pair_selector.py data.pkl ./output_combo_0/ --combination-index 0
```

### Multiple Jobs (sbatch)
```bash
#!/bin/bash
#SBATCH --job-name=gene_selector
#SBATCH --array=0-19
#SBATCH --output=logs/gene_selector_%A_%a.out
#SBATCH --error=logs/gene_selector_%A_%a.err

python gene_pair_selector.py data.pkl ./output_combo_${SLURM_ARRAY_TASK_ID}/ --combination-index ${SLURM_ARRAY_TASK_ID}
```

## Error Handling

- **Invalid combination index**: Script validates the index range (0-19)
- **Empty filtered data**: Warns if no gene pairs match the chosen combination
- **Missing data columns**: Validates required columns exist
- **Data type validation**: Ensures boolean/numeric columns are correct types

## Workflow

1. **Combination Selection**: Use predefined combination by index
2. **Data Loading**: Load and validate the input pickle file
3. **Data Filtering**: Filter data based on chosen combination
4. **Distance Calculation**: Calculate CCC distances to Pearson and Spearman
5. **Sorting**: Sort by specified distance metric (smallest distance first)
6. **Output**: Save filtered data, sorted data, and metadata

## Performance Notes

- Memory efficient: Processes one file at a time
- Optimized for large datasets
- Suitable for HPC environments
- Comprehensive logging for debugging 