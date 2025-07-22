# GTEx Coefficient Distribution Analysis

This script generates comprehensive plots to compare coefficient values from Pearson, Spearman and CCC (Clustermatch), including their distributions, cumulative histograms, and joint plots.

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
python 11_00-gtex_general_plots.py
```

This will:
- Use `whole_blood` tissue data
- Use default data directories and labels
- Automatically clean data by removing NaN values
- Create timestamped log directory in `./logs/YYYYMMDD_HHMMSS/`
- Save figures to both original and log directories

### Custom Tissue Analysis

```bash
python 11_00-gtex_general_plots.py --tissue lung
```

### Custom Data Directory

```bash
python 11_00-gtex_general_plots.py \
    --data-dir /path/to/your/gtex/data \
    --output-dir /path/to/output/figures
```

### Custom Log Directory

```bash
python 11_00-gtex_general_plots.py \
    --log-dir ./custom_logs/my_analysis
```

### Complete Custom Configuration

```bash
python 11_00-gtex_general_plots.py \
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
