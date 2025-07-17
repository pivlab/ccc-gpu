# Gene Pair Selector

This script processes gene intersection files from a data directory and allows users to select from predefined combinations of boolean indicators for a specific tissue, then filters and sorts the data based on CCC distance to Pearson and Spearman values. **Designed for batch processing and sbatch job submission.**

## Features

- **Non-interactive mode**: Perfect for sbatch job submission
- **Tissue-specific processing**: Analyze specific tissues from a data directory
- **Predefined combinations**: 20 pre-selected combinations for systematic analysis
- **Automatic file discovery**: Finds intersection files based on tissue name
- **CCC distance calculation**: Measures distance between CCC and Pearson/Spearman values
- **Multiple sorting options**: Combined, mean, or max distance metrics
- **Comprehensive caching**: Saves filtered and sorted results
- **Detailed logging**: Complete processing logs and metadata

## Usage

### Basic Usage
```bash
python gene_pair_selector.py --data-dir /path/to/intersections --tissue TISSUE_NAME --output /path/to/output --combination-index INDEX
```

### List Available Tissues
```bash
python gene_pair_selector.py --data-dir /path/to/intersections --tissue liver --output /path/to/output --list-tissues
```

### List Available Combinations
```bash
python gene_pair_selector.py --data-dir /path/to/intersections --tissue liver --output /path/to/output --list-combinations
```

### Full Example
```bash
python gene_pair_selector.py \
    --data-dir /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections \
    --tissue liver \
    --output ./output_liver_combo0/ \
    --combination-index 0 \
    --sort-by combined
```

## Arguments

### Required Arguments
- `--data-dir`: Directory containing gene pair intersection files
- `--tissue`: Tissue to analyze (see available tissues below)
- `--output`: Base output directory (combination-specific subdirectory will be created)
- `--combination-index`: Index of predefined combination to use (0-19)

### Optional Arguments
- `--sort-by {combined,mean,max}`: Distance metric to sort by (default: combined)
- `--log-file`: Path to log file (default: gene_pair_selector.log)
- `--list-combinations`: List all available predefined combinations and exit
- `--list-tissues`: List all available tissues and exit

## Available Tissues

The script supports the following tissues:

```
adipose_subcutaneous
adipose_visceral_omentum
adrenal_gland
artery_aorta
artery_coronary
artery_tibial
bladder
brain_amygdala
brain_anterior_cingulate_cortex_ba24
brain_caudate_basal_ganglia
brain_cerebellar_hemisphere
brain_cerebellum
brain_cortex
brain_frontal_cortex_ba9
brain_hippocampus
brain_hypothalamus
brain_nucleus_accumbens_basal_ganglia
brain_putamen_basal_ganglia
brain_spinal_cord_cervical_c1
brain_substantia_nigra
breast_mammary_tissue
cells_cultured_fibroblasts
cells_ebvtransformed_lymphocytes
cervix_ectocervix
cervix_endocervix
colon_sigmoid
colon_transverse
esophagus_gastroesophageal_junction
esophagus_mucosa
esophagus_muscularis
fallopian_tube
heart_atrial_appendage
heart_left_ventricle
kidney_cortex
kidney_medulla
liver
lung
minor_salivary_gland
muscle_skeletal
nerve_tibial
ovary
pancreas
pituitary
prostate
skin_not_sun_exposed_suprapubic
skin_sun_exposed_lower_leg
small_intestine_terminal_ileum
spleen
stomach
testis
thyroid
uterus
vagina
whole_blood
```

## File Discovery

The script automatically finds intersection files using these patterns:
1. `gene_pair_intersections*{tissue}*.pkl`
2. `*{tissue}*.pkl`

If multiple files match, the first one is used (with a warning).

## Predefined Combinations

The script uses 20 predefined combinations in the format:
`[Spearman(low), Pearson(low), Clustermatch(low), Spearman(high), Pearson(high), Clustermatch(high)]`

```
Index  Name                      Spearman(L)  Pearson(L)  Clustermatch(L)  Spearman(H)  Pearson(H)  Clustermatch(H)
0      c-high-p-high-s-high      False        False       False            True         True        True
1      c-high-p-high-s-none      False        False       False            False        True        True
2      c-high-p-none-s-high      False        False       False            True         False       True
3      c-none-p-high-s-high      False        False       False            True         True        False
4      c-low-p-low-s-none        False        True        True             False        False       False
5      c-low-p-none-s-low        True         False       True             False        False       False
6      c-none-p-low-s-low        True         True        False            False        False       False
7      c-low-p-low-s-low         True         True        True             False        False       False
8      c-high-p-low-s-none       False        True        False            False        False       True
9      c-high-p-none-s-low       True         False       False            False        False       True
10     c-high-p-low-s-low        True         True        False            False        False       True
11     c-high-p-high-s-low       True         False       False            False        True        True
12     c-high-p-low-s-high       False        True        False            True         False       True
13     c-low-p-high-s-none       False        False       True             False        True        False
14     c-low-p-high-s-low        True         False       True             False        True        False
15     c-none-p-high-s-low       True         False       False            False        True        False
16     c-low-p-high-s-high       False        False       True             True         True        False
17     c-low-p-none-s-high       False        False       True             True         False       False
18     c-low-p-low-s-high        False        True        True             True         False       False
19     c-none-p-low-s-high       False        True        False            True         False       False
```

## Expected Input Data Structure

The input pickle files should contain pandas DataFrames with:

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

The script automatically creates a subdirectory with descriptive combination name (e.g., `c-high-p-low-s-low`) and generates the following files:

1. **`filtered_data_cache.pkl`**: Gene pairs filtered by chosen combination
2. **`sorted_data_cache.pkl`**: Filtered data sorted by CCC distance
3. **`selection_metadata.pkl`**: Metadata about the selection process
4. **`gene_pair_selector_YYYYMMDD_HHMMSS.log`**: Timestamped log file

### Output Directory Names

The output subdirectory names follow this pattern: `{combination_name}`

Where combination names use:
- `c-high`: Clustermatch high
- `c-low`: Clustermatch low  
- `c-none`: Neither Clustermatch high nor low
- Similar patterns for `p-` (Pearson) and `s-` (Spearman)

Examples: 
- `c-high-p-low-s-low`
- `c-none-p-high-s-high`

## Distance Metrics

The script calculates several distance metrics:

- **`ccc_pearson_diff`**: Absolute difference between CCC and Pearson values
- **`ccc_spearman_diff`**: Absolute difference between CCC and Spearman values
- **`ccc_combined_distance`**: Sum of both differences
- **`ccc_mean_distance`**: Mean of both differences
- **`ccc_max_distance`**: Maximum of both differences

## Batch Processing Examples

### Single Tissue Processing
```bash
python gene_pair_selector.py \
    --data-dir /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections \
    --tissue liver \
    --output ./output/ \
    --combination-index 0
# This will create: ./output/c-high-p-high-s-high/
```

### Multiple Tissues and Combinations (sbatch)
```bash
#!/bin/bash
#SBATCH --job-name=gene_selector
#SBATCH --array=0-19
#SBATCH --output=logs/gene_selector_%A_%a.out
#SBATCH --error=logs/gene_selector_%A_%a.err

# Define tissues array
TISSUES=(liver lung brain_cortex heart_left_ventricle muscle_skeletal)

# Process each tissue with the current combination
for tissue in "${TISSUES[@]}"; do
    python gene_pair_selector.py \
        --data-dir /pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections \
        --tissue ${tissue} \
        --output ./output/ \
        --combination-index ${SLURM_ARRAY_TASK_ID}
done
# This will create directories like: ./output/c-high-p-high-s-high/
```

## Error Handling

- **Invalid tissue**: Script validates tissue names against available list
- **Missing intersection file**: Provides helpful error with available files
- **Invalid combination index**: Script validates the index range (0-19)
- **Empty filtered data**: Warns if no gene pairs match the chosen combination
- **Missing data columns**: Validates required columns exist
- **Data type validation**: Ensures boolean/numeric columns are correct types

## Workflow

1. **File Discovery**: Find intersection file for specified tissue
2. **Data Loading**: Load and validate the intersection file
3. **Combination Selection**: Use predefined combination by index
4. **Data Filtering**: Filter data based on chosen combination
5. **Distance Calculation**: Calculate CCC distances to Pearson and Spearman
6. **Sorting**: Sort by specified distance metric (smallest distance first)
7. **Output**: Save filtered data, sorted data, and metadata

## Performance Notes

- Memory efficient: Processes one tissue at a time
- Optimized for large datasets
- Suitable for HPC environments
- Comprehensive logging for debugging
- Automatic file discovery reduces manual path management 