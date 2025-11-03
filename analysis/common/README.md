# Common Analysis Tools

This directory contains command-line tools for gene expression analysis using the CCC-GPU package.

## Available Tools

1. **[Single Gene Pair Correlation Analysis](#single-gene-pair-correlation-analysis)** (`compute_single_gene_pair_correlations_cli.py`)
2. **[Gene Expression-Metadata Correlation Analysis](#gene-expression-metadata-correlation-analysis)** (`metadata_corr_cli.py`)

---

# Single Gene Pair Correlation Analysis

A command-line tool for exploring gene expression data and computing correlations between specific gene pairs using CCC (Clustered Correlation Coefficient), Spearman, and Pearson correlation methods.

## Features

- **Data Exploration**: Browse available tissues and genes with their symbols
- **Gene Pair Correlation**: Compute three correlation coefficients (CCC, Pearson, Spearman) for any gene pair
- **Flexible Gene Input**: Accept both gene symbols (e.g., TP53) and Ensembl IDs (e.g., ENSG00000141510.16)
- **Tissue-Specific Analysis**: Analyze correlations within specific tissue contexts
- **Robust Gene Resolution**: Handle version numbers and case-insensitive matching
- **Comprehensive Error Handling**: Clear error messages and debugging support

## Installation Requirements

```bash
# Required packages
pip install pandas numpy
# CCC-GPU package (install from source as per project instructions)
```

## Quick Start

### 1. Explore Available Data

```bash
# List all available tissues
python compute_single_gene_pair_correlations_cli.py --list-tissues

# Show genes available in whole blood tissue
python compute_single_gene_pair_correlations_cli.py --show-genes whole_blood

# Show more genes (default is 20)
python compute_single_gene_pair_correlations_cli.py --show-genes liver --n-genes 50
```

### 2. Compute Gene Pair Correlations

```bash
# Basic correlation analysis between TP53 and BRCA1 in whole blood
python compute_single_gene_pair_correlations_cli.py TP53 BRCA1 --tissue whole_blood

# Use Ensembl IDs instead of symbols
python compute_single_gene_pair_correlations_cli.py ENSG00000141510.16 ENSG00000012048.20 --tissue liver

# Mixed input (symbol and Ensembl ID)
python compute_single_gene_pair_correlations_cli.py TP53 ENSG00000012048.20 --tissue brain_cortex
```

### 3. Save Results and Logs

```bash
# Save results and logs to a specific directory
python compute_single_gene_pair_correlations_cli.py TP53 BRCA1 --tissue whole_blood \
    --output-dir ./results

# Combine with debug logging for detailed output
python compute_single_gene_pair_correlations_cli.py TP53 BRCA1 --tissue liver \
    --output-dir ./detailed_analysis --debug
```

### 4. Custom Data Paths

```bash
# Use custom data directory and gene mapping file
python compute_single_gene_pair_correlations_cli.py TP53 BRCA1 --tissue whole_blood \
    --data-dir /custom/path/to/tissue/data \
    --gene-mapping /custom/path/to/gene_mappings.pkl \
    --output-dir ./custom_results
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `genes` | str+ | Required | Two gene symbols or Ensembl IDs for correlation analysis |
| `--tissue` | str | Required | Tissue name for correlation analysis |
| `--data-dir` | str | `/mnt/data/proj_data/ccc-gpu/data/tutorial/data_by_tissue` | Directory containing tissue expression data |
| `--gene-mapping` | str | `/mnt/data/proj_data/ccc-gpu/data/tutorial/gtex_gene_id_symbol_mappings.pkl` | Gene mapping file path |
| `--output-dir` | str | None | Directory to save output files and logs (optional) |
| `--list-tissues` | flag | False | List all available tissues and exit |
| `--show-genes` | str | None | Show genes for specified tissue and exit |
| `--n-genes` | int | 20 | Number of genes to display |
| `--debug` | flag | False | Enable debug logging |

## Output Format

### Tissue and Gene Discovery
```
=== Available Tissues (49) ===
 1. adipose_subcutaneous
 2. adipose_visceral_omentum
 3. adrenal_gland
 ...

=== Tissue: whole_blood ===
Total genes: 56,200
Total samples: 755

First 20 genes:
------------------------------------------------------------
#    Gene Symbol     Ensembl ID          
------------------------------------------------------------
1    DDX11L1         ENSG00000223972.5   
2    WASH7P          ENSG00000227232.5   
3    MIR6859-1       ENSG00000278267.1   
...
```

### Correlation Results
```
============================================================
GENE PAIR CORRELATION RESULTS
============================================================
Gene 1: TP53 (ENSG00000141510.16)
Gene 2: BRCA1 (ENSG00000012048.20)
Tissue: whole_blood
Samples: 755
------------------------------------------------------------
         CCC: 0.123456
     PEARSON: 0.234567
    SPEARMAN: 0.345678
============================================================
Results saved to:
  JSON: TP53_BRCA1_whole_blood_20240925_143022_correlation_results.json
  Pickle: TP53_BRCA1_whole_blood_20240925_143022_correlation_results.pkl
Log file: gene_pair_correlation_analysis_20240925_143022.log
```

### Output Files (when --output-dir is used)

1. **JSON Results File**: `{gene1}_{gene2}_{tissue}_{timestamp}_correlation_results.json`
   - Human-readable format with all correlation results
   - Can be easily imported into other tools or scripts

2. **Pickle Results File**: `{gene1}_{gene2}_{tissue}_{timestamp}_correlation_results.pkl`
   - Python-specific format preserving exact data types
   - Optimal for downstream analysis in Python

3. **Log File**: `gene_pair_correlation_analysis_{timestamp}.log`
   - Detailed processing information and debug messages
   - Useful for troubleshooting and audit trails

Example JSON output:
```json
{
  "gene1_symbol": "TP53",
  "gene1_ensembl_id": "ENSG00000141510.16",
  "gene2_symbol": "BRCA1", 
  "gene2_ensembl_id": "ENSG00000012048.20",
  "tissue": "whole_blood",
  "n_samples": 755,
  "ccc": 0.123456,
  "pearson": 0.234567,
  "spearman": 0.345678
}
```

## Input Data Format

### Tissue Expression Files
- **Format**: Pickle (.pkl) files
- **Naming**: `gtex_v8_data_{tissue_name}.pkl`
- **Structure**: DataFrame with Ensembl gene IDs as index, sample IDs as columns
- **Content**: Log2-transformed gene expression values

### Gene Mapping File
- **Format**: Pickle (.pkl) file
- **Structure**: DataFrame with columns `gene_ens_id` and `gene_symbol`
- **Content**: Mapping between Ensembl gene IDs and HUGO gene symbols

## Statistical Methods

### Correlation Coefficients

1. **CCC (Clustered Correlation Coefficient)**
   - GPU-accelerated implementation
   - Robust to outliers and non-linear relationships
   - Particularly suited for detecting complex correlation patterns

2. **Pearson Correlation**
   - Standard linear correlation coefficient
   - Measures linear relationship strength

3. **Spearman Correlation**
   - Rank-based correlation coefficient
   - Robust to outliers and monotonic relationships

## Example Workflows

### 1. Cancer Gene Analysis
```bash
# Explore brain tissues for TP53-related genes
python compute_single_gene_pair_correlations_cli.py --list-tissues | grep brain

# Analyze TP53 interactions in different brain regions with output saving
python compute_single_gene_pair_correlations_cli.py TP53 MDM2 --tissue brain_cortex \
    --output-dir ./cancer_gene_analysis --debug
python compute_single_gene_pair_correlations_cli.py TP53 CDKN1A --tissue brain_hippocampus \
    --output-dir ./cancer_gene_analysis --debug
```

### 2. Housekeeping Gene Analysis
```bash
# Compare expression correlation of housekeeping genes
python compute_single_gene_pair_correlations_cli.py GAPDH ACTB --tissue whole_blood
python compute_single_gene_pair_correlations_cli.py GAPDH ACTB --tissue liver
python compute_single_gene_pair_correlations_cli.py GAPDH ACTB --tissue muscle_skeletal
```

### 3. Tissue-Specific Gene Discovery
```bash
# Find genes in specific tissues and analyze their relationships
python compute_single_gene_pair_correlations_cli.py --show-genes heart_left_ventricle --n-genes 100 | grep MYH
python compute_single_gene_pair_correlations_cli.py MYH6 MYH7 --tissue heart_left_ventricle
```

## Error Handling

The tool provides comprehensive error handling:

- **Gene not found**: Suggestions to check spelling or use `--show-genes`
- **Tissue not found**: List of available tissues
- **Data issues**: Clear messages about insufficient samples or missing data
- **Path issues**: Validation of data directory and gene mapping file

## Performance Considerations

- **Memory usage**: ~100-500MB depending on tissue size
- **Computation time**: 1-5 seconds per gene pair
- **CCC computation**: GPU-accelerated when available

---

# Gene Expression-Metadata Correlation Analysis

## Overview

This tool computes correlations between specific gene expression levels and all available metadata columns across multiple GTEx tissues. It uses the **Clustered Correlation Coefficient (CCC)** method, which is particularly suited for detecting non-linear relationships and complex correlation patterns.

### Key Features

- **Multi-gene Analysis**: Process multiple genes simultaneously
- **Cross-tissue Analysis**: Analyze correlations across all available GTEx tissues
- **Comprehensive Metadata Coverage**: Correlate against all metadata columns automatically
- **Statistical Significance**: Permutation-based p-value calculation with customizable iterations
- **Flexible Tissue Filtering**: Include/exclude tissues using pattern matching
- **Parallel Processing**: Multi-threaded computation support
- **Detailed Logging**: Individual logs per gene-tissue combination plus comprehensive summaries
- **Multiple Output Formats**: Results in both pickle (.pkl) and CSV formats
- **Runtime Tracking**: Detailed performance monitoring and optimization insights

## Requirements

### Dependencies

```python
pandas
numpy
ccc  # Clustered Correlation Coefficient library
```

### Required Data Files

The tool expects specific data files in predetermined locations:

1. **Expression Data**: GTEx v8 expression files in the format `gtex_v8_data_{tissue_name}-var_pc_log2.pkl`
2. **Metadata**: GTEx v8 sample metadata (`gtex_v8-sample_metadata.pkl`)
3. **Gene Mappings**: Gene ID to symbol mappings (`gtex_gene_id_symbol_mappings.pkl`)

## Installation

```bash
# Clone or download the script
# Ensure all required Python packages are installed
pip install pandas numpy ccc
```

## Usage

### Basic Usage

```bash
# Analyze single gene across all tissues
python metadata_corr_cli.py RASSF2

# Analyze multiple genes
python metadata_corr_cli.py RASSF2 TP53 BRCA1

# Specify custom output directory
python metadata_corr_cli.py RASSF2 --output-dir ./results
```

### Advanced Usage

```bash
# Include only specific tissues (pattern matching)
python metadata_corr_cli.py RASSF2 --include brain liver

# Exclude specific tissues
python metadata_corr_cli.py RASSF2 --exclude cells brain

# Custom permutation settings and parallel processing
python metadata_corr_cli.py RASSF2 --permutations 500000 --n-jobs 16

# Combined filtering and custom settings
python metadata_corr_cli.py TP53 BRCA1 \
    --include muscle heart \
    --exclude cells \
    --permutations 1000000 \
    --n-jobs 32 \
    --output-dir ./tp53_brca1_analysis
```

### Discovery Commands

```bash
# List all available tissues
python metadata_corr_cli.py GENE --list-tissues

# List all available metadata columns
python metadata_corr_cli.py GENE --list-metadata-columns
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gene_symbols` | str+ | Required | Gene symbol(s) to analyze (e.g., RASSF2 TP53) |
| `--expr-data-dir` | str | `/mnt/data/proj_data/ccc-gpu/data/tutorial/data_by_tissue` | Directory containing expression data files |
| `--include` | str* | None | Include only tissues matching these patterns |
| `--exclude` | str* | None | Exclude tissues matching these patterns |
| `--permutations` | int | 100,000 | Number of permutations for p-value calculation |
| `--n-jobs` | int | 24 | Number of parallel jobs for computation |
| `--output-dir` | str | `.` | Directory to save output files |
| `--list-metadata-columns` | flag | False | List available metadata columns and exit |
| `--list-tissues` | flag | False | List available tissue files and exit |

## Input File Formats

### Expression Data Files
- **Format**: Pickle (.pkl) files
- **Structure**: DataFrame with genes as rows, samples as columns
- **Naming**: `gtex_v8_data_{tissue_name}-var_pc_log2.pkl`
- **Content**: Log2-transformed, variance-filtered gene expression data

### Metadata File
- **Format**: Pickle (.pkl) file
- **Structure**: DataFrame with samples as rows, metadata columns as columns
- **Content**: All GTEx v8 sample metadata including demographics, sampling info, etc.

### Gene Mapping File
- **Format**: Pickle (.pkl) file
- **Structure**: DataFrame with columns `gene_ens_id` and `gene_symbol`
- **Content**: Mapping between Ensembl gene IDs and gene symbols

## Output Files

### Per Gene-Tissue Results
- **Individual Results**: `{gene}_{tissue}_correlation_results.pkl`
- **Individual Logs**: `{gene}_{tissue}.log`
- **Content**: Correlation results for each metadata column

### Per Gene Summaries
- **Combined Results**: `{gene}_all_tissues_correlation_results.pkl`
- **Combined CSV**: `{gene}_all_tissues_correlation_results.csv`
- **Content**: All tissues combined for single gene

### Overall Results
- **Mega Results**: `_all_genes_all_tissues_correlation_results.pkl`
- **Mega CSV**: `_all_genes_all_tissues_correlation_results.csv`
- **Summary Log**: `_{genes}_summary_execution.log`
- **Summary Tables**: `_{genes}_summary_tables.log`

### Result DataFrame Structure

```python
# Each results DataFrame contains:
{
    'ccc_value': float,      # CCC correlation coefficient
    'p_value': float,        # Permutation-based p-value
    'status': str,           # 'success', 'all_nan', 'insufficient_variation', or 'error'
    'tissue': str,           # Tissue name
    'gene_symbol': str,      # Gene symbol
    'gene_id': str,         # Ensembl gene ID
    'n_samples': int        # Number of samples used
}
```

## Analysis Workflow

### 1. **Gene Discovery**
- Converts gene symbols to Ensembl IDs using gene mapping
- Validates gene existence across tissues

### 2. **Tissue Processing**
- Loads expression data for each tissue
- Filters to common samples between expression and metadata
- Handles missing data and insufficient variation gracefully

### 3. **Correlation Analysis**
- Computes CCC between gene expression and each metadata column
- Calculates statistical significance via permutation testing
- Handles various data types and edge cases

### 4. **Results Compilation**
- Aggregates results across tissues and genes
- Generates comprehensive summary statistics
- Creates ranked lists of strongest correlations

### 5. **Performance Monitoring**
- Tracks runtime for each gene-tissue combination
- Identifies computational bottlenecks
- Provides optimization recommendations

## Statistical Methods

### Clustered Correlation Coefficient (CCC)
- **Purpose**: Detects both linear and non-linear relationships
- **Advantages**: Robust to outliers, captures complex patterns
- **Implementation**: Uses permutation-based significance testing

### Significance Levels
- `***`: p < 0.001 (highly significant)
- `**`: p < 0.01 (significant)
- `*`: p < 0.05 (marginally significant)
- `ns`: p ≥ 0.05 (not significant)

## Performance Considerations

### Computational Requirements
- **Memory**: ~2-8 GB depending on tissue size and number of genes
- **CPU**: Benefits from multi-core systems (default: 24 cores)
- **Time**: ~1-5 minutes per gene-tissue combination

### Optimization Tips
- **Parallel Processing**: Increase `--n-jobs` for faster computation
- **Permutations**: Reduce `--permutations` for faster (less precise) p-values
- **Tissue Filtering**: Use `--include`/`--exclude` to focus on relevant tissues
- **Batch Processing**: Process multiple genes together for efficiency

## Example Workflows

### 1. Cancer Gene Analysis
```bash
# Analyze tumor suppressor genes across cancer-relevant tissues
python metadata_corr_cli.py TP53 BRCA1 BRCA2 PTEN \
    --include breast ovary lung liver \
    --permutations 1000000 \
    --n-jobs 32 \
    --output-dir ./cancer_genes_analysis
```

### 2. Brain-Specific Gene Study
```bash
# Focus on brain tissues for neurological genes
python metadata_corr_cli.py APOE MAPT SNCA \
    --include brain \
    --exclude cells \
    --output-dir ./brain_genes
```

### 3. Exploratory Analysis
```bash
# Quick exploration with reduced permutations
python metadata_corr_cli.py GENE_OF_INTEREST \
    --permutations 10000 \
    --n-jobs 8 \
    --output-dir ./exploratory
```

## Troubleshooting

### Common Issues

1. **Gene Not Found**: Check gene symbol spelling and availability in gene mapping
2. **No Expression Data**: Verify gene is expressed in selected tissues
3. **Memory Errors**: Reduce number of parallel jobs or process fewer genes at once
4. **File Not Found**: Ensure all required data files exist in expected locations

### Error Codes
- **Gene symbol not found**: Gene not in mapping file
- **No common samples**: Expression and metadata samples don't overlap
- **All NaN values**: Metadata column contains only missing values
- **Insufficient variation**: Metadata column has ≤1 unique values

## Output Interpretation

### Top Results Tables
- Results ranked by absolute CCC value
- Include significance levels and tissue information
- Show strongest correlations across all analyses

### Summary Statistics
- **Mean |CCC|**: Average absolute correlation strength
- **Max |CCC|**: Strongest correlation found
- **Success Rate**: Proportion of successful analyses
- **Runtime Metrics**: Performance characteristics

## Citation

If you use this tool in your research, please cite the CCC method and relevant GTEx publications.

## Version Information

- **Script**: metadata_corr_cli.py
- **Converted from**: 00-data-exploration.ipynb
- **GTEx Version**: v8
- **CCC Implementation**: Uses ccc.coef module

## Support

For issues related to:
- **CCC Method**: Refer to CCC library documentation
- **GTEx Data**: Consult GTEx consortium resources
- **Script Usage**: Check this README or examine log files for detailed error messages 