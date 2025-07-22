# GTEx Coefficient Distribution Analysis

## 🚀 **NEW: Memory-Efficient Streaming Version**

### **Problem Solved**: OOM Kills with Large Datasets

If you're experiencing out-of-memory (OOM) kills when processing large GTEx datasets, use the **streaming version** (`11_00-gtex_general_plots_streaming.py`) that processes data in chunks to avoid memory issues.

### **Key Benefits of Streaming Version**

- **✅ Memory Efficient**: Processes data in configurable chunks (default: 50K-100K rows)
- **✅ OOM Prevention**: Avoids loading entire datasets into memory at once
- **✅ Progress Tracking**: Detailed logging of chunk processing progress
- **✅ Enhanced Plots**: Improved visualization with positive-only range and better formatting
- **✅ Same Results**: Produces equivalent plots with identical statistical accuracy
- **✅ Robust Processing**: Handles datasets of any size within system limits

#### **🎨 Recent Plot Improvements**
- **Positive Range Only**: Both histograms now show only 0-1.0 range (no negative values)
- **Better Titles**: Cumulative plot has subtitle showing processed gene pairs count
- **Threshold Visualization**: Vertical lines at coefficient thresholds + horizontal line at target %
- **Cleaner Layout**: Improved spacing and consistent formatting across both plot types

### **Quick Start - Streaming Version**

#### For Large Datasets (Recommended)
```bash
# Use streaming version for datasets causing OOM kills
conda activate ccc-gpu
python 11_00-gtex_general_plots_streaming.py --tissue whole_blood --chunk-size 50000
```

#### Submit SLURM Jobs (Memory-Optimized)
```bash
# Multi-tissue: Process all 53 GTEx tissues in one job
sbatch run_slurm_streaming.sh

# Single tissue: Process one tissue for testing
sbatch --export=TISSUE=whole_blood run_single_tissue_streaming.sh
```

### **When to Use Which Version**

| Dataset Size | Recommended Script | Memory Requirement |
|--------------|-------------------|-------------------|
| < 1M rows | `11_00-gtex_general_plots.py` | 16-32GB |
| > 1M rows | `11_00-gtex_general_plots_streaming.py` | 16GB |
| OOM kills | `11_00-gtex_general_plots_streaming.py` | 16GB |
| All 53 tissues | `sbatch run_slurm_streaming.sh` | 32GB total |

### **Streaming Version Features**

#### **1. Multi-Tissue Sequential Processing**
- **📊 All 53 GTEx Tissues**: Process all tissues with one SLURM job
- **🔄 Sequential Loop**: Processes tissues one after another within single job
- **🎯 Optimized Resources**: 32GB memory, 24 hours total runtime
- **📋 Progress Monitoring**: Real-time progress tracking with detailed logging

#### **2. Chunked Processing**
- Processes data in configurable chunks (default: 100K rows)
- Accumulates histograms incrementally across chunks
- Immediately frees memory after each chunk

#### **3. Memory Monitoring**
- Logs memory usage and processing progress
- Reports data cleaning statistics per chunk
- Provides total processing summary

#### **4. Enhanced Output**
- **Overlaid Density Plot**: `dist-histograms.svg` 
  - **✅ NEW**: Smooth density curves for all three coefficients on one plot
  - **✅ NEW**: Matches original `ccc.plots.plot_histogram` style and output filename
  - **✅ NEW**: Uses cubic interpolation for smooth curves from histogram data
  - **✅ NEW**: Clean overlaid visualization like the original non-streaming version
- **Regular Histograms**: `dist-histograms_streaming.svg` 
  - **✅ NEW**: Only shows positive range (0 to 1.0), no negative values
  - **✅ NEW**: Statistics calculated from positive range only
- **Cumulative Histograms**: `dist-cum_histograms_streaming.svg`
  - **✅ NEW**: Subtitle shows processed gene pairs count
  - **✅ NEW**: Vertical threshold lines for each coefficient
  - **✅ NEW**: Only shows positive range (0 to 1.0)
- Same statistical accuracy as full-memory processing

#### **5. Custom Parameters**
```bash
python 11_00-gtex_general_plots_streaming.py \
    --tissue adipose_subcutaneous \
    --chunk-size 75000 \              # Adjust based on available memory
    --gene-pairs-percent 0.8 \
    --log-dir ./streaming_analysis
```

#### **6. Multi-Tissue Job Monitoring**
```bash
# Check job status
squeue -u $USER

# Monitor real-time progress (all tissues in one log)
tail -f logs/gtex_all_tissues_*.out

# Check completed analyses
ls -la logs/*_[timestamp]/

# View final job summary
tail -n 50 logs/gtex_all_tissues_*.out
```

---

## Standard Version (Original)

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
- `--pearson-label`: Label for Pearson coefficient (default: `