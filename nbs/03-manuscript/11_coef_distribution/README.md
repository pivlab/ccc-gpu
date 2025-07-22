# GTEx Coefficient Distribution Analysis

This script generates comprehensive plots to compare coefficient values from Pearson, Spearman and CCC (Clustermatch), including their distributions, cumulative histograms, and joint plots.

## Features

- **🔍 Comprehensive Analysis**: Generates histograms, cumulative histograms, and joint plots
- **📊 Dual Output**: Saves figures to both original destinations and timestamped log folders
- **🖥️ CLI Interface**: Configurable parameters via command-line arguments
- **📋 Detailed Logging**: Comprehensive logging with progress tracking and error handling
- **🧹 Data Cleaning**: Automatically removes rows with NaN values in coefficient columns
- **🏗️ Robust Error Handling**: Graceful handling of missing files and processing errors
- **📁 Organized Output**: Timestamped log directories for easy tracking

## Requirements

```bash
# Core dependencies
pandas
scipy
seaborn
matplotlib
svgutils

# CCC package (project-specific)
# Ensure the CCC package is installed and available in your environment
```

## Data Processing

### Automatic Data Cleaning

The script automatically performs the following data cleaning steps:

1. **NaN Value Removal**: Automatically removes any rows containing NaN values in the three coefficient columns (CCC, Pearson, Spearman)
2. **Detailed Logging**: Reports the number and percentage of rows removed during cleaning
3. **Verification**: Double-checks that no NaN values remain after cleaning
4. **Statistics**: All reported statistics are calculated on the clean data

**Example cleaning log output:**
```
🧹 Cleaning data: removing rows with NaN values...
  ❌ Removed 1,234 rows with NaN values (2.47%)
  ✅ Clean data shape: (48,766, 3)
✅ Data cleaning verified: no NaN values in coefficient columns
```

## Usage

### Basic Usage (Default Parameters)

```bash
python 11_00-gtex_whole_blood-general_plots.py
```

This will:
- Use `whole_blood` tissue data
- Use default data directories and labels
- Automatically clean data by removing NaN values
- Create timestamped log directory in `./logs/YYYYMMDD_HHMMSS/`
- Save figures to both original and log directories

### Custom Tissue Analysis

```bash
python 11_00-gtex_whole_blood-general_plots.py --tissue lung
```

### Custom Data Directory

```bash
python 11_00-gtex_whole_blood-general_plots.py \
    --data-dir /path/to/your/gtex/data \
    --output-dir /path/to/output/figures
```

### Custom Log Directory

```bash
python 11_00-gtex_whole_blood-general_plots.py \
    --log-dir ./custom_logs/my_analysis
```

### Complete Custom Configuration

```bash
python 11_00-gtex_whole_blood-general_plots.py \
    --tissue adipose_subcutaneous \
    --gene-pairs-percent 0.80 \
    --ccc-label "ClusterMatch" \
    --pearson-label "Pearson r" \
    --spearman-label "Spearman ρ" \
    --log-dir ./analysis_logs/adipose_analysis
```

## Command-Line Arguments

### Configuration Arguments
- `--tissue`: GTEx tissue type to analyze (default: `whole_blood`)
- `--top-n-genes`: Number of top genes to use (default: `all`)
- `--gene-selection-strategy`: Gene selection strategy (default: `var_pc_log2`)
- `--gene-pairs-percent`: Percentage for cumulative histogram (default: `0.70`)

### Directory Arguments
- `--data-dir`: Base data directory (default: `/pividori_lab/haoyu_projects/ccc-gpu/data/gtex`)
- `--output-dir`: Base output directory (default: `/pividori_lab/haoyu_projects/ccc-gpu/figures`)
- `--log-dir`: Custom log directory (default: `./logs/YYYYMMDD_HHMMSS`)

### Label Arguments
- `--ccc-label`: Label for CCC coefficient (default: `CCC`)
- `--pearson-label`: Label for Pearson coefficient (default: `Pearson`)
- `--spearman-label`: Label for Spearman coefficient (default: `Spearman`)

## Generated Outputs

### Figure Files (saved to both original and log directories)
- `dist-histograms.svg`: Distribution histograms for all coefficients
- `dist-cum_histograms.svg`: Cumulative distribution histograms
- `dist-pearson_vs_ccc.svg`: Joint plot comparing Pearson and CCC
- `dist-spearman_vs_ccc.svg`: Joint plot comparing Spearman and CCC
- `dist-spearman_vs_pearson.svg`: Joint plot comparing Spearman and Pearson
- `dist-main.svg`: Composite figure combining all plots

### Log Files
- `coefficient_analysis.log`: Comprehensive analysis log with timestamps and progress tracking

## Output Directory Structure

```
Original Output Directory:
/pividori_lab/haoyu_projects/ccc-gpu/figures/coefs_comp/gtex_{tissue}/
├── dist-histograms.svg
├── dist-cum_histograms.svg
├── dist-pearson_vs_ccc.svg
├── dist-spearman_vs_ccc.svg
├── dist-spearman_vs_pearson.svg
└── dist-main.svg

Log Directory:
./logs/20240101_120000/
├── coefficient_analysis.log
├── dist-histograms.svg
├── dist-cum_histograms.svg
├── dist-pearson_vs_ccc.svg
├── dist-spearman_vs_ccc.svg
├── dist-spearman_vs_pearson.svg
└── dist-main.svg
```

## Expected Input Data Structure

The script expects the following directory structure:

```
{data_dir}/
├── gene_selection/{top_n_genes}/
│   └── gtex_v8_data_{tissue}-{gene_sel_strategy}.pkl
└── similarity_matrices/{top_n_genes}/
    └── gtex_v8_data_{tissue}-{gene_sel_strategy}-all.pkl
```

## Logging Features

The script provides comprehensive logging including:
- ✅ Input file validation
- 📊 Data loading and basic statistics
- 🧹 Data cleaning progress and results
- 🔄 Progress tracking for each plot generation step
- 📋 Summary of generated files
- ❌ Error handling with detailed error messages
- 📁 Clear indication of output locations

## Example Log Output

```
2024-01-01 12:00:00 - INFO - Logging initialized - Log file: ./logs/20240101_120000/coefficient_analysis.log
================================================================================
GTEx COEFFICIENT DISTRIBUTION ANALYSIS
================================================================================
2024-01-01 12:00:00 - INFO - Configuration:
2024-01-01 12:00:00 - INFO -   tissue: whole_blood
2024-01-01 12:00:00 - INFO -   gene_pairs_percent: 0.7
2024-01-01 12:00:01 - INFO - Validating input files...
2024-01-01 12:00:01 - INFO - ✅ Gene expression file: /path/to/gene/file.pkl
2024-01-01 12:00:01 - INFO - ✅ Correlation file: /path/to/correlation/file.pkl
2024-01-01 12:00:01 - INFO - Loading correlation data from: /path/to/correlation/file.pkl
2024-01-01 12:00:02 - INFO - ✅ Data loaded successfully - Shape: (50000, 3)
2024-01-01 12:00:02 - INFO - 🧹 Cleaning data: removing rows with NaN values...
2024-01-01 12:00:02 - INFO -   ❌ Removed 1,234 rows with NaN values (2.47%)
2024-01-01 12:00:02 - INFO -   ✅ Clean data shape: (48,766, 3)
2024-01-01 12:00:02 - INFO - ✅ Data cleaning verified: no NaN values in coefficient columns
2024-01-01 12:00:02 - INFO - Generating histogram plots...
2024-01-01 12:00:03 - INFO - 📊 Figure copied to log: dist-histograms.svg
2024-01-01 12:00:03 - INFO - ✅ Histogram plots generated successfully
... (more detailed progress) ...
2024-01-01 12:00:10 - INFO - ✅ All plots generated successfully!
```

## Error Handling

The script includes robust error handling for:
- Missing input files
- Data loading failures
- Data cleaning issues
- Plot generation errors
- File copying failures
- Invalid arguments

All errors are logged with detailed information to help with debugging.

## Migration from Original Jupyter Notebook

This script replaces the original Jupyter notebook with:
- ✅ Better structure and modularity
- ✅ Command-line interface for flexibility
- ✅ Comprehensive logging and error handling
- ✅ Automatic data cleaning with NaN removal
- ✅ Dual output to preserve original behavior while adding log copies
- ✅ Progress tracking and detailed statistics
- ✅ Clean, maintainable code structure 