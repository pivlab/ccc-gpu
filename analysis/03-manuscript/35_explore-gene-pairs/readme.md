# Explore Gene Pairs

Tools and notebooks for exploring and analyzing selected gene pairs from correlation coefficient intersections.

## Scripts Overview

### 00-report_top_gene_pairs.py

Reports top gene pairs from gene pair selection results. Can operate in single analysis mode or batch processing mode.

**Usage:**

```bash
# Single tissue/combination analysis
python 00-report_top_gene_pairs.py \
    --tissue whole_blood \
    --combination c-high-p-low-s-low \
    --top 1000

# Batch processing of all tissues and combinations
python 00-report_top_gene_pairs.py \
    --data-dir /path/to/gene_pair_selection \
    --top 1000 \
    --batch
```

**Key Arguments:**
- `--data-dir`: Directory containing gene pair selection results
- `--tissue`: Specific tissue to process (single mode)
- `--combination`: Specific combination category (single mode)
- `--top`: Number of top gene pairs to report
- `--batch`: Process all tissues and combinations
- `--csv`: Output CSV files in addition to text reports

**Output:**
- `top_{N}_gene_pairs.csv`: Top gene pairs with correlation metrics
- Gene symbols mapped from Ensembl IDs

### 10-correlate-top-genes-with-metadata.py

Combines top gene pairs across tissues and automatically computes correlations with GTEx metadata for comprehensive analysis.

**Usage:**

```bash
# Process top 1000 gene pairs with metadata correlations
python 10-correlate-top-genes-with-metadata.py --top 1000

# Custom output directory
python 10-correlate-top-genes-with-metadata.py \
    --top 5000 \
    --output-dir /path/to/output \
    --top-metadata-correlations 10
```

**Key Features:**
- Combines results from all tissues per combination category
- Automatically computes fresh metadata correlations for each gene pair
- Extracts top-N metadata variables for each gene individually and commonly
- Adds 30+ metadata correlation columns per gene pair

**Output:**
- Enhanced datasets: `combined_{combination}_top_gene_pairs_with_metadata.pkl` and `.csv`
- Each gene pair enriched with metadata correlation data

**⚠️ Computational Requirements:**
- Processing time scales linearly with number of gene pairs
- Top 1000: ~1-3 hours
- Top 10000: ~10-30 hours

**For detailed documentation, see:** `README_10-correlate-top-genes-with-metadata.md`

### 12-select_interersting_gene_pairs.ipynb

Jupyter notebook for selecting and analyzing interesting gene pairs from the enriched datasets. Interactive exploration and filtering of gene pairs based on metadata correlations and other criteria.

### 15-gene_pair_inspection.ipynb

Jupyter notebook for detailed inspection and visualization of specific gene pairs. Provides tools for deep-dive analysis of individual gene pair characteristics.

## Workflow

```
gene_pair_selection results
    ↓
00-report_top_gene_pairs.py
    → top_{N}_gene_pairs.csv files
    ↓
10-correlate-top-genes-with-metadata.py
    → enhanced datasets with metadata correlations
    ↓
12-select_interersting_gene_pairs.ipynb
    → curated interesting gene pairs
    ↓
15-gene_pair_inspection.ipynb
    → detailed analysis of selected pairs
```

## Combination Categories

All tools process these 5 combination categories:

1. **c-high-p-high-s-low**: High CCC, High Pearson, Low Spearman
2. **c-high-p-low-s-high**: High CCC, Low Pearson, High Spearman
3. **c-high-p-low-s-low**: High CCC, Low Pearson, Low Spearman
4. **c-high-p-low-s-none**: High CCC, Low Pearson, No Spearman constraint
5. **c-high-p-none-s-low**: High CCC, No Pearson constraint, Low Spearman

## Input/Output Directories

**Default Input:**
- Gene pair selection results: `/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection`

**Default Output:**
- Top gene pairs: `{data_dir}/{tissue}/{combination}/top_{N}_gene_pairs.csv`
- Enhanced with metadata: `/pividori_lab/haoyu_projects/ccc-gpu/results/top_gene_pair_correlation`

