# Gene Pair Analysis Tool

This tool provides comprehensive analysis of gene pair correlation data with support for both single analysis and large-scale batch processing across multiple tissues and correlation combinations.

## Overview

The script can operate in two main modes:

1. **Single Analysis Mode**: Analyze gene pairs from a specific tissue/combination
2. **Batch Processing Mode**: Process all tissues and combinations

## Features

- **Flexible Input**: Direct file input, tissue/combination specification, or full batch processing
- **Parallel Processing**: Multi-threaded processing for efficient batch analysis
- **Comprehensive Logging**: Timestamped logs with detailed progress tracking
- **Multiple Output Formats**: Text reports and CSV exports
- **Statistical Summaries**: Descriptive statistics for each tissue-combination

## Installation

Requires Python 3.7+ with the following packages:
```bash
pip install pandas numpy pathlib concurrent.futures
```

## Usage

### Single Analysis Mode

Analyze a specific tissue and combination:
```bash
python 00-report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --top 100
```

Direct file input:
```bash
python 00-report_top_gene_pairs.py --input /path/to/sorted_data_cache.pkl --top 50 --output my_results.txt
```

### Batch Processing Mode

Process all tissues and combinations:
```bash
python 00-report_top_gene_pairs.py --data-dir /path/to/gene_pair_selection --top 1000 --batch
```

With custom worker threads:
```bash
python 00-report_top_gene_pairs.py --data-dir /path/to/gene_pair_selection --top 500 --batch --workers 8
```

## Data Structure Requirements

The tool expects the following directory structure:
```
data_dir/
├── tissue1/
│   ├── c-high-p-low-s-high/
│   │   ├── sorted_data_cache.pkl
│   │   ├── selection_metadata.pkl
│   │   └── [other files]
│   ├── c-high-p-high-s-low/
│   │   └── [same structure]
│   └── [other combinations]
├── tissue2/
│   └── [same structure]
└── [other tissues]
```

### Expected Combinations

The tool recognizes these standard combinations:
- `c-high-p-low-s-high`
- `c-high-p-high-s-low`
- `c-high-p-low-s-low`
- `c-high-p-none-s-low`
- `c-high-p-low-s-none`

## Arguments

### Required Arguments (Mutually Exclusive)
- `--batch`: Run batch processing mode
- `--input FILE`: Direct path to sorted_data_cache.pkl file
- `--tissue TISSUE`: Tissue name (requires --combination)

### Optional Arguments
- `--data-dir DIR`: Base directory for gene pair selection results (default: `/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection`)
- `--combination COMBO`: Combination name (required with --tissue)
- `--top N`: Number of top gene pairs to analyze (default: 100)
- `--workers N`: Number of parallel workers for batch processing (default: 4)
- `--output FILE`: Output text file path (single mode only)
- `--csv FILE`: Output CSV file path (single mode only)

## Output Files

### Single Mode Outputs
- `{tissue}_{combination}_top_{N}_gene_pairs.txt`: Human-readable report
- `{tissue}_{combination}_top_{N}_gene_pairs.csv`: CSV export of top gene pairs

### Batch Mode Outputs

#### Per Combination Directory
- `top_{N}_gene_pairs.txt`: Analysis report for this tissue-combination
- `top_{N}_gene_pairs.csv`: CSV export of top gene pairs

#### Log Directory
- `logs/batch_analysis_{timestamp}/gene_pair_analysis.log`: Comprehensive processing log

## Batch Processing Workflow

1. **Discovery**: Scan data directory for all tissues and valid combinations
2. **Parallel Processing**: Process each tissue-combination pair concurrently
3. **Analysis**: Generate top N gene pairs analysis for each tissue-combination
4. **Report Generation**: Create individual analysis reports and CSV exports for each combination



## Logging

### Single Mode
- Console output with basic progress information
- Error messages and warnings

### Batch Mode
- Timestamped log directory: `logs/batch_analysis_{YYYYMMDD_HHMMSS}/`
- Detailed log file with function-level tracking
- Progress updates for each tissue-combination processed
- Performance timing information
- Error handling and recovery information

## Performance Considerations

- **Memory Usage**: Large datasets may require substantial RAM for data loading
- **Parallel Processing**: Adjust `--workers` based on available CPU cores and memory
- **Disk Space**: Batch mode generates many output files; ensure adequate storage
- **Processing Time**: Full batch processing of ~50 tissues × 5 combinations may take significant time

## Examples

### Basic Usage
```bash
# Single tissue analysis
python 00-report_top_gene_pairs.py --tissue liver --combination c-high-p-low-s-low

# Full batch processing
python 00-report_top_gene_pairs.py --batch --top 500
```

### Advanced Usage
```bash
# Custom data directory with high parallelism
python 00-report_top_gene_pairs.py \
    --data-dir /custom/path/gene_pair_selection \
    --batch \
    --top 1000 \
    --workers 16

# Single analysis with custom output
python 00-report_top_gene_pairs.py \
    --tissue whole_blood \
    --combination c-high-p-high-s-low \
    --top 200 \
    --output blood_high_corr_results.txt \
    --csv blood_high_corr_data.csv
```

## Metadata Correlation Enhancement

The `10-correlate-top-genes-with-metadata.py` script extends the gene pair combination functionality by enriching each gene pair with metadata correlation information from GTEx.

### Overview

This enhancement correlates each gene in the gene pairs with GTEx metadata variables (such as age, sex, BMI, etc.) and adds the top metadata correlations as additional columns to the combined dataset.

### Usage

#### Basic Usage

```bash
python 10-correlate-top-genes-with-metadata.py --top 10000 --enhance-metadata --metadata-corr-file /path/to/correlation_results.pkl
```

#### Full Configuration

```bash
python 10-correlate-top-genes-with-metadata.py \
  --top 10000 \
  --enhance-metadata \
  --metadata-corr-file /pividori_lab/haoyu_projects/ccc-gpu/data/gtex/all_genes_all_tissues_correlation_results.pkl \
  --top-metadata-correlations 5 \
  --output-dir ./results_with_metadata \
  --data-dir /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection
```

### Arguments

- `--enhance-metadata`: Enable metadata correlation enhancement
- `--metadata-corr-file`: Path to the metadata correlation results pickle file
- `--top-metadata-correlations`: Number of top metadata correlations to extract per gene (default: 5)

### Output Enhancement

When metadata enhancement is enabled, the following columns are added to each gene pair:

#### Gene 1 Metadata Correlations
- `gene1_top1_metadata` through `gene1_topN_metadata`: Metadata variable names
- `gene1_top1_ccc` through `gene1_topN_ccc`: CCC correlation values
- `gene1_top1_pvalue` through `gene1_topN_pvalue`: Statistical significance p-values

#### Gene 2 Metadata Correlations
- `gene2_top1_metadata` through `gene2_topN_metadata`: Metadata variable names
- `gene2_top1_ccc` through `gene2_topN_ccc`: CCC correlation values
- `gene2_top1_pvalue` through `gene2_topN_pvalue`: Statistical significance p-values

#### Common Metadata Correlations
- `common_top1_metadata` through `common_topN_metadata`: Metadata variables significantly correlated with both genes

### Metadata Correlation Algorithm

1. **Individual Gene Analysis**: For each gene, identify the top N metadata variables with the strongest absolute correlations (p-value ≤ 0.05)

2. **Common Correlation Analysis**: Find metadata variables significantly correlated with both genes using a minimum rank approach:
   - Filter significant correlations (p-value ≤ 0.05) for both genes
   - Compute a combined score using `min(|ccc_gene1|, |ccc_gene2|)`
   - Rank by combined score to identify the strongest common correlations

### Coverage Analysis

The script provides detailed coverage analysis:
- Total unique genes in the dataset
- Number of genes with metadata correlations available
- Coverage rate percentage
- Warnings for low coverage scenarios

### Output Files

Enhanced output includes `_with_metadata` suffix:
- `combined_{combination}_top_gene_pairs_with_metadata.pkl`
- `combined_{combination}_top_gene_pairs_with_metadata.csv`

### Performance Considerations

- Metadata enhancement increases processing time significantly
- Memory usage increases due to additional columns
- Processing is optimized with batch operations and progress tracking
- Large datasets may benefit from processing smaller subsets

## Fresh Metadata Correlation Computation

The `--compute-fresh-correlations` option provides even more powerful functionality by computing **fresh, real-time** metadata correlations for each gene pair using the `metadata_corr_cli.py` script.

### Overview

Unlike the pre-computed metadata enhancement, this feature dynamically calls the metadata correlation CLI for each gene pair, computing correlations with all available GTEx metadata variables on-the-fly.

### Usage

#### Basic Fresh Correlation Computation

```bash
python 10-correlate-top-genes-with-metadata.py --top 1000 --compute-fresh-correlations
```

#### Combined with Pre-computed Enhancement

```bash
python 10-correlate-top-genes-with-metadata.py \
  --top 1000 \
  --enhance-metadata \
  --metadata-corr-file /path/to/precomputed.pkl \
  --compute-fresh-correlations \
  --top-metadata-correlations 3
```

### Arguments

- `--compute-fresh-correlations`: Enable real-time metadata correlation computation
- `--top-metadata-correlations`: Number of top correlations to compute per gene (applies to both pre-computed and fresh)

### Fresh Correlation Output

When fresh correlation computation is enabled, **21 additional columns** are added (assuming `--top-metadata-correlations 5`):

#### Gene 1 Fresh Correlations (15 columns)
- `gene1_fresh_top1_metadata` through `gene1_fresh_top5_metadata`: Metadata variable names
- `gene1_fresh_top1_ccc` through `gene1_fresh_top5_ccc`: Fresh CCC correlation values  
- `gene1_fresh_top1_pvalue` through `gene1_fresh_top5_pvalue`: Fresh p-values

#### Gene 2 Fresh Correlations (15 columns)
- `gene2_fresh_top1_metadata` through `gene2_fresh_top5_metadata`: Metadata variable names
- `gene2_fresh_top1_ccc` through `gene2_fresh_top5_ccc`: Fresh CCC correlation values
- `gene2_fresh_top1_pvalue` through `gene2_fresh_top5_pvalue`: Fresh p-values

#### Common Fresh Correlations (5 columns)
- `common_fresh_top1_metadata` through `common_fresh_top5_metadata`: Common metadata variables

### Fresh Correlation Algorithm

1. **Individual Gene Processing**: For each gene in a pair, call `metadata_corr_cli.py` to compute correlations with all GTEx metadata
2. **Result Parsing**: Parse CLI output to extract successful correlations (p-value ≤ 0.05)
3. **Top Selection**: Rank by absolute CCC value and select top N correlations per gene
4. **Common Analysis**: Apply the min-rank algorithm to find metadata variables significantly correlated with both genes

### Technical Implementation

- **CLI Integration**: Programmatically calls `metadata_corr_cli.py` with optimized parameters:
  - `--permutations 10000` (reduced for speed)
  - `--n-jobs 4` (limited to prevent system overload)
  - `--timeout 300` seconds per gene pair
- **Temporary Management**: Creates isolated temp directories for each gene pair
- **Error Handling**: Graceful handling of CLI failures, timeouts, and missing data
- **Progress Tracking**: Progress updates every 10 gene pairs (less frequent due to computational cost)

### Output Files

Fresh correlation outputs include `_with_fresh_metadata` suffix:
- `combined_{combination}_top_gene_pairs_with_fresh_metadata.pkl`
- `combined_{combination}_top_gene_pairs_with_fresh_metadata.csv`

Combined mode creates: `_with_metadata_with_fresh_metadata` suffix.

### Performance Characteristics

⚠️ **Computationally Intensive**: Fresh correlation computation is **extremely resource-intensive**:

- **Time Complexity**: ~5-30 seconds per gene pair (depending on gene and metadata complexity)
- **For 10,000 gene pairs**: Expect 14-83 hours of computation time
- **Memory Usage**: Significant temporary file storage required
- **I/O Intensive**: Heavy disk usage for temporary CLI results

**Recommendations**:
- Start with small datasets (10-100 gene pairs) for testing
- Use adequate storage space for temporary files
- Consider running on compute clusters for large datasets
- Monitor system resources during execution

### Use Cases

- **Exploratory Analysis**: When pre-computed correlations don't exist for your specific genes
- **Comprehensive Studies**: When you need the most current and complete correlation analysis
- **Validation**: To verify and extend pre-computed correlation results
- **Custom Metadata**: When working with updated or specialized metadata variables

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that data directory structure matches expected format
2. **Memory errors**: Reduce `--top` number or `--workers` count
3. **Permission errors**: Ensure write access to output directories and log directories
4. **Low metadata coverage**: If very few genes have metadata correlations available, verify:
   - The metadata correlation file contains data for your genes of interest
   - Gene symbols match between datasets
   - The correlation file is complete and not a sample/test file
5. **Fresh correlation CLI failures**: If `--compute-fresh-correlations` fails:
   - Verify the metadata correlation CLI script is accessible
   - Check that GTEx expression data is available in the expected directory
   - Ensure sufficient disk space for temporary files
   - Monitor system resources (memory, CPU) during intensive processing
   - Consider reducing `--top-metadata-correlations` for faster processing

### Debug Information

Use the generated log files to troubleshoot issues:
- Check `gene_pair_analysis.log` for detailed processing information
- Verify tissue and combination discovery in the early log entries
- Monitor progress and identify failed tissue-combination pairs

## Data Format Notes

- Gene pairs are expected to be stored as multi-index DataFrames with `['gene1', 'gene2']` index
- Numeric correlation columns should include `ccc`, `pearson`, `spearman`
- The data should be pre-sorted with best gene pairs at the top (lowest index values)

## License

This tool is part of the CCC-GPU project analysis pipeline. 