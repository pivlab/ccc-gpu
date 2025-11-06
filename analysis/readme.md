# Analysis Directory

This directory contains all analysis scripts, notebooks, and tools for the CCC-GPU project, including the complete reproducible pipeline for manuscript results using GTEx v8 gene expression data.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Numbering System](#numbering-system)
- [Pipeline Workflow](#pipeline-workflow)
- [Directory Details](#directory-details)
- [Quick Start](#quick-start)
- [Computational Requirements](#computational-requirements)
- [Documentation](#documentation)

---

## Directory Structure

```
analysis/
├── 03-manuscript/              # Main manuscript analysis pipeline
│   ├── 00_preprocessing/       # Data preprocessing and setup
│   ├── 05_compute-correlations/    # Compute correlation matrices
│   ├── 10_combine-coefs/       # Combine correlation coefficients
│   ├── 11_check-coef-distributions/    # Distribution analysis
│   ├── 15_compute-coef-intersections/  # Method agreement analysis
│   ├── 20_combine-intersections/   # Cross-tissue aggregation
│   ├── 25_metadata-correlation/    # Metadata correlation analysis
│   ├── 30_gene-pair-selection/     # Select interesting gene pairs
│   ├── 35_explore-gene-pairs/      # Deep-dive analysis
│   ├── 80-general-analysis/    # General exploratory analysis
│   │   └── kendall-and-spearman/   # Method comparison
│   └── 90_benchmarking/        # Performance benchmarking
│       ├── ccc-vs-cccgpu/      # GPU vs CPU comparison
│       └── spearman-and-pearson/   # Execution time analysis
├── 99-tutorials/               # Tutorial notebooks
└── common/                     # Shared analysis tools
```

---

## Numbering System

The `03-manuscript/` directory uses a **two-digit prefix numbering system** to indicate the sequential order of analysis steps:

### Core Analysis Pipeline (00-35)
- **00**: Preprocessing - Initial data preparation
- **05**: Computation - Primary correlation analysis
- **10**: Combination - Merge results from different methods
- **11**: Validation - Quality checks and distributions
- **15**: Advanced Analysis - Method agreement and intersections
- **20**: Aggregation - Combine results across tissues
- **25**: Metadata - Gene-metadata correlations
- **30**: Selection - Identify interesting gene pairs
- **35**: Exploration - Deep-dive analysis and visualization

### Supplementary Analyses (80+)
- **80**: General Analysis - Exploratory analysis and method comparisons
- **90**: Benchmarking - Performance validation and comparisons

This numbering ensures scripts are executed in the correct order for reproducible research.

---

## Pipeline Workflow

The analysis follows a sequential workflow from raw data to manuscript results:

```
GTEx v8 Raw Data
    ↓
[00] Preprocessing
    • Download and prepare GTEx v8 data
    ↓
[05] Compute Correlations
    • Calculate CCC-GPU, Pearson, and Spearman for all gene pairs
    • Process all 54 GTEx tissues
    ↓
[10] Combine Coefficients
    • Merge correlation matrices from different methods
    • Create unified dataframes per tissue
    ↓
[11] Check Distributions
    • Generate distribution plots and statistics
    • Compute percentile summaries
    ↓
[15] Compute Intersections
    • Identify high/low correlation agreements
    • Create UpSet plots showing method overlaps
    ↓
[20] Combine Intersections
    • Aggregate intersection counts across tissues
    • Visualize cross-tissue patterns
    ↓
[25] Metadata Correlation (parallel track)
    • Analyze gene expression vs GTEx metadata
    • Compute permutation-based p-values
    ↓
[30] Gene Pair Selection
    • Select gene pairs based on correlation patterns
    • 20 predefined combinations (e.g., c-high-p-low-s-low)
    ↓
[35] Explore Gene Pairs
    • In-depth analysis of selected pairs
    • Generate manuscript figures and reports
```

---

## Directory Details

### `80-general-analysis/`

**Purpose**: General exploratory analysis and method comparisons for manuscript validation

**Subdirectories**:

#### `kendall-and-spearman/`
- Comparative analysis between Kendall's tau and Spearman's rho correlation methods
- Includes analysis on both real GTEx v8 data and synthetic random data
- **Key Files**:
  - `correlation_comparison_gtex.ipynb` - Method comparison using real GTEx whole blood gene expression data
  - `correlation_comparison_random_data.ipynb` - Method comparison using synthetic data with controlled properties
- **Function**: Validate that Kendall and Spearman correlations are highly redundant (r > 0.95), justifying the manuscript's focus on Spearman coefficient only
- **Output**: Scatter plots, correlation statistics, and linear regression analysis demonstrating method agreement

---

### `90_benchmarking/`

**Purpose**: Performance benchmarking and validation

**Subdirectories**:

#### `ccc-vs-cccgpu/`
- GPU vs CPU performance comparison
- Speedup analysis across different core counts (6, 12, 24)
- Scaling tests (features and samples)
- **Documentation**: [90_benchmarking/ccc-vs-cccgpu/readme.md](03-manuscript/90_benchmarking/ccc-vs-cccgpu/readme.md)

#### `spearman-and-pearson/`
- Execution time benchmarking
- Comparative analysis: CCC-GPU vs CCC vs Spearman vs Pearson
- Includes command-line benchmark tool and visualization notebooks

---

### `99-tutorials/`

**Purpose**: Educational resources and tutorials

**Key Files**:
- `05-walkthrough-with-gtex-data.ipynb` - Complete walkthrough using GTEx v8 data

**Function**: Learn the CCC-GPU workflow with real examples

---

## Quick Start

### 1. Prerequisites
- GTEx v8 gene expression data (download via `00_preprocessing`)
- GPU support for CCC-GPU acceleration
- Python environment with required packages
- Sufficient memory (varies by step, see Computational Requirements)

### 2. Running the Pipeline

Execute steps sequentially following the numbering system:

```bash
# Step 00: Preprocessing
cd 03-manuscript/00_preprocessing
jupyter notebook 00-prepross-gtex-data.ipynb

# Step 05: Compute correlations
cd ../05_compute-correlations
python compute_correlations_cli.py --data-dir /path/to/gtex --output-dir ./results

# Step 10: Combine coefficients
cd ../10_combine-coefs
python combine_coefs.py --data-dir /path/to/correlations --output-dir ./combined

# Continue with steps 11, 15, 20, 25, 30, 35...
```

### 3. Exploring Specific Tools

```bash
# Use common tools for ad-hoc analysis
cd common
python compute_single_gene_pair_correlations_cli.py --help
python metadata_corr_cli.py --help

# Run tutorials
cd ../99-tutorials
jupyter notebook 05-walkthrough-with-gtex-data.ipynb
```

---

## Computational Requirements

| Step | Memory | Time | Parallelization | Notes |
|------|--------|------|-----------------|-------|
| 00_preprocessing | Moderate | Hours | Single-threaded | One-time setup |
| 05_compute-correlations | Moderate | Hours | GPU-accelerated | Fastest with GPU |
| 10_combine-coefs | ~300GB | Hours | Single-threaded | High memory |
| 11_check-coef-distributions | Low | Minutes | Streaming | Memory-efficient |
| 15_compute-coef-intersections | ~512GB | Hours | Single-threaded | Highest memory |
| 20_combine-intersections | ~350GB | Hours | Multi-threaded | Auto-scales |
| 25_metadata-correlation | ~8GB | Hours | 24 cores | Parallelizable |
| 30_gene-pair-selection | Moderate | Variable | Batch-friendly | Tissue-level parallelism |
| 35_explore-gene-pairs | ~8GB | 1-30 hours | Depends on N | Scales with gene pairs |

**Note**: Times and memory requirements are approximate and depend on hardware, dataset size, and configuration.

---

## Documentation

Each subdirectory contains its own README with detailed documentation:

- **Common Tools**: [common/readme.md](common/readme.md)
- **Compute Correlations**: [05_compute-correlations/readme.md](03-manuscript/05_compute-correlations/readme.md)
- **Combine Coefficients**: [10_combine-coefs/readme.md](03-manuscript/10_combine-coefs/readme.md)
- **Check Distributions**: [11_check-coef-distributions/readme.md](03-manuscript/11_check-coef-distributions/readme.md)
- **Compute Intersections**: [15_compute-coef-intersections/readme.md](03-manuscript/15_compute-coef-intersections/readme.md)
- **Combine Intersections**: [20_combine-intersections/readme.md](03-manuscript/20_combine-intersections/readme.md)
- **Metadata Correlation**: [25_metadata-correlation/readme.md](03-manuscript/25_metadata-correlation/readme.md)
- **Gene Pair Selection**: [30_gene-pair-selection/readme.md](03-manuscript/30_gene-pair-selection/readme.md)
- **Explore Gene Pairs**: [35_explore-gene-pairs/readme.md](03-manuscript/35_explore-gene-pairs/readme.md)
- **Benchmarking**: [90_benchmarking/ccc-vs-cccgpu/readme.md](03-manuscript/90_benchmarking/ccc-vs-cccgpu/readme.md)
