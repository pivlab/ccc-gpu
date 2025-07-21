# Gene Pair Display Toolkit

This directory contains a simple CLI script to display and export bottom gene pairs from processed gene pair selection data.

## Files

- `report_top_gene_pairs.py` - Main CLI script to display bottom gene pairs
- `README.md` - This documentation file

## Overview

The `report_top_gene_pairs.py` script loads processed gene pair data from the `gene_pair_selector.py` output and creates human-readable reports with:

- Metadata information (tissue, combination, processing details)
- Data summary statistics
- Bottom N gene pairs formatted in a table
- Export to text files for further analysis
- Export to CSV files for data analysis and visualization

## Installation

The script requires the following Python packages:
- pandas
- numpy

Install with:
```bash
pip install pandas numpy
```

## Usage

The script supports two main usage patterns:

### 1. Direct File Input

Use this when you know the exact path to the sorted data cache file:

```bash
python report_top_gene_pairs.py --input /path/to/sorted_data_cache.pkl --output results.txt
```

### 2. Tissue and Combination Input

Use this for easier access when you know the tissue and combination:

```bash
python report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --bottom 100
```

## Command Line Arguments

### Required Arguments (mutually exclusive)

- `--input` : Direct path to `sorted_data_cache.pkl` file
- `--tissue` : Tissue name (requires `--combination`)

### Optional Arguments

- `--combination` : Combination name (e.g., `c-high-p-low-s-low`) - required when using `--tissue`
- `--data-dir` : Base directory for gene pair selection results (default: `/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection`)
- `--output` : Output text file path (default: auto-generated based on tissue/combination)
- `--csv` : Output CSV file path (default: auto-generated based on tissue/combination)
- `--bottom` : Number of bottom gene pairs to display (default: 30)

## Examples

### Example 1: Direct File Access

```bash
python report_top_gene_pairs.py \
  --input /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection/whole_blood_combination_10/c-high-p-low-s-low/sorted_data_cache.pkl \
  --bottom 30 \
  --output whole_blood_bottom_30.txt \
  --csv whole_blood_bottom_30.csv
```

**Output:** Creates:
- `whole_blood_bottom_30.txt` with complete metadata and formatted table
- `whole_blood_bottom_30.csv` with bottom 30 gene pairs in CSV format for analysis

### Example 2: Tissue and Combination Access

```bash
python report_top_gene_pairs.py \
  --tissue whole_blood \
  --combination c-high-p-low-s-low \
  --bottom 50 \
  --output wb_c_high_p_low_s_low_bottom_50.txt
```

**Output:** Same as Example 1, but automatically finds the file based on tissue and combination.

### Example 3: CSV Export Only

```bash
python report_top_gene_pairs.py \
  --tissue whole_blood \
  --combination c-high-p-low-s-low \
  --bottom 50 \
  --csv gene_pairs_for_analysis.csv
```

**Output:** Creates both text and CSV files, with CSV formatted for easy import into R, Python, or Excel.

### Example 4: Quick Preview

```bash
python report_top_gene_pairs.py \
  --tissue whole_blood \
  --combination c-high-p-low-s-low \
  --bottom 10
```

**Output:** Creates auto-named files:
- `whole_blood_c-high-p-low-s-low_bottom_10_gene_pairs.txt` (text report)
- `whole_blood_c-high-p-low-s-low_bottom_10_gene_pairs.csv` (CSV data)

### Example 5: Different Tissue

```bash
python report_top_gene_pairs.py \
  --tissue liver \
  --combination c-low-p-high-s-both \
  --bottom 100 \
  --data-dir /custom/path/to/results
```

## Output Format

The script generates two output files:

### Text Report (.txt file)

### 1. Header Section
- Report title and generation timestamp
- Metadata from the gene pair selection process
- Data summary (shape, columns, statistics)

### 2. Statistics Section
- Descriptive statistics for all numeric columns (ccc, pearson, spearman, distance metrics)
- Count, mean, std, min, 25%, 50%, 75%, max

### 3. Bottom Gene Pairs Table
- Formatted table with gene pairs and all their values
- Boolean columns displayed as T/F
- Numeric values with appropriate precision
- Shows the bottom N gene pairs from the original sorted data

### CSV Export (.csv file)

- **Gene pair columns**: `level_0` (gene1) and `level_1` (gene2) for multi-index data
- **Boolean columns**: `True`/`False` values for all correlation categories
- **Numeric columns**: All original numeric values preserved
- **Machine readable**: Ready for import into R, Python, Excel, or other analysis tools
- **Contains only data**: No metadata or statistics, just the bottom N gene pairs

## Sample Output

```
Gene Pair Analysis Report
================================================================================
Generated: 2025-07-17 10:37:21

METADATA
--------------------
Tissue: whole_blood
Combination: c-high-p-low-s-low
Data Dir: /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections
...

DATA SUMMARY
--------------------
Total gene pairs: 6,120,027
Data shape: (6120027, 14)
Columns: ['Pearson (high)', 'Pearson (low)', ...]

NUMERIC COLUMN STATISTICS
------------------------------
                ccc       pearson      spearman  ...
count  6.120027e+06  6.120027e+06  6.120027e+06  ...
mean   1.605449e-02  8.072147e-03  1.716317e-02  ...
...

Bottom 30 Gene Pairs (from original sorted data)
============================================================

Gene1        Gene2        Pearson (high) Pearson (low) ... ccc      pearson  spearman
----------------------------------------------------------------------------------------
ENSG00000071127.16 ENSG00000100226.15    F        T     ... 0.122070 0.013237 0.009102
ENSG00000118900.14 ENSG00000087074.7     F        T     ... 0.120911 0.015747 0.002115
...
```

## Understanding the Data

### Column Meanings

**Boolean Columns:**
- `Pearson (high)`, `Pearson (low)`: Whether Pearson correlation is in high/low category
- `Spearman (high)`, `Spearman (low)`: Whether Spearman correlation is in high/low category  
- `Clustermatch (high)`, `Clustermatch (low)`: Whether Clustermatch correlation is in high/low category

**Numeric Columns:**
- `ccc`: CCC (Canonical Correlation Coefficient) value
- `pearson`: Pearson correlation coefficient
- `spearman`: Spearman correlation coefficient
- `ccc_pearson_diff`: Absolute difference |ccc - pearson|
- `ccc_spearman_diff`: Absolute difference |ccc - spearman|
- `distance_combined`: Sum of CCC differences (combined distance)
- `distance_mean`: Mean of CCC differences
- `distance_max`: Maximum of CCC differences

### Combination Names

Combination names follow the pattern: `c-X-p-Y-s-Z` where:
- `c` = Clustermatch: `high`, `low`, `both`, `none`
- `p` = Pearson: `high`, `low`, `both`, `none`  
- `s` = Spearman: `high`, `low`, `both`, `none`

Examples:
- `c-high-p-low-s-low`: Clustermatch high, Pearson low, Spearman low
- `c-both-p-none-s-high`: Clustermatch both, Pearson none, Spearman high

## Troubleshooting

### File Not Found Errors

If you get "Could not find sorted data cache" errors:

1. **Check the data directory**: Verify `--data-dir` points to the correct location
2. **Check tissue name**: Ensure tissue name matches directory structure
3. **Check combination name**: Verify combination name is correct
4. **Use direct input**: Try using `--input` with the full file path

### Memory Issues

For very large datasets:
- Reduce `--bottom` number to display fewer rows
- The script loads the full dataset but only processes the requested bottom N rows for display

### Performance

The script is optimized for display purposes:
- Loads data once
- Processes only the bottom N rows for formatting
- Minimal memory usage for text generation

## Related Tools

- `../25_gene-pair-selection/gene_pair_selector.py` - Main gene pair selection and filtering script
- Output from this script can be used as input for further analysis or visualization

## Version History

- v1.0: Initial release with direct file input and tissue/combination input support
- v1.1: Updated to display bottom gene pairs instead of top, default changed to 30
- v1.2: Added CSV export functionality with `--csv` argument
- Auto-generated output filenames and comprehensive metadata reporting 