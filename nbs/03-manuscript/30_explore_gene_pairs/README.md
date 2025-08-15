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
python report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --top 100
```

Direct file input:
```bash
python report_top_gene_pairs.py --input /path/to/sorted_data_cache.pkl --top 50 --output my_results.txt
```

### Batch Processing Mode

Process all tissues and combinations:
```bash
python report_top_gene_pairs.py --data-dir /path/to/gene_pair_selection --top 1000 --batch
```

With custom worker threads:
```bash
python report_top_gene_pairs.py --data-dir /path/to/gene_pair_selection --top 500 --batch --workers 8
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
python report_top_gene_pairs.py --tissue liver --combination c-high-p-low-s-low

# Full batch processing
python report_top_gene_pairs.py --batch --top 500
```

### Advanced Usage
```bash
# Custom data directory with high parallelism
python report_top_gene_pairs.py \
    --data-dir /custom/path/gene_pair_selection \
    --batch \
    --top 1000 \
    --workers 16

# Single analysis with custom output
python report_top_gene_pairs.py \
    --tissue whole_blood \
    --combination c-high-p-high-s-low \
    --top 200 \
    --output blood_high_corr_results.txt \
    --csv blood_high_corr_data.csv
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that data directory structure matches expected format
2. **Memory errors**: Reduce `--top` number or `--workers` count
3. **Permission errors**: Ensure write access to output directories and log directories

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