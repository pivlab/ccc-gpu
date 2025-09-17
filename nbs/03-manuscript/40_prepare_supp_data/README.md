# CCC Data Processing Script

This directory contains a script to process GTEx similarity matrices (.pkl files) and extract only the CCC (Clustered Correlation Coefficient) data.

## Script: `process_ccc_data.py`

### Description
Processes all .pkl files in the source directory, extracts only the 'ccc' column with multi-indices, and saves individual .parquet files with snappy compression for each input. This significantly reduces file sizes compared to .pkl format.

### Usage

#### Dry Run (recommended first step)
```bash
# Activate the conda environment
conda activate ccc-gpu

# Run dry run to see what files would be processed
python process_ccc_data.py --dry-run
```

#### Full Processing
```bash
# Run with default paths
python process_ccc_data.py

# Run with custom paths
python process_ccc_data.py --source-dir /path/to/source --output-dir /path/to/output
```

### Arguments
- `--source-dir`: Source directory containing .pkl files (default: `/mnt/data/proj_data/ccc-gpu/data/gtex/similarity_matrices/all`)
- `--output-dir`: Output directory for processed parquet files (default: `/mnt/data/proj_data/ccc-gpu/data/gtex/ccc_similarity_matrices_parquet`)
- `--dry-run`: Show what would be processed without actually doing it
- `--debug`: Enable debug logging (shows detailed processing information)

### Requirements
- `pandas`: For reading .pkl files and writing .parquet files
- `pyarrow`: Required for parquet format support with snappy compression
- `tqdm`: For progress bars (optional, script will work without it)

### Logging
The script automatically creates detailed logs in the `logs/` directory with timestamps:
- **Log location**: `logs/process_ccc_data_YYYYMMDD_HHMMSS.log`
- **Log levels**: INFO (default) and DEBUG (with `--debug` flag)
- **Log content**: Processing progress, file details, errors, timing information, and archive sizes
- **Console output**: Key information is also printed to console for real-time monitoring

Example log entries:
```
2024-01-15 10:30:15,123 - INFO - Starting CCC data processing
2024-01-15 10:30:15,124 - INFO - Found 54 .pkl files to process
2024-01-15 10:30:16,200 - INFO - Processing file: gtex_v8_data_whole_blood-var_pc_log2-all.pkl
2024-01-15 10:35:22,456 - INFO - Successfully processed gtex_v8_data_whole_blood-var_pc_log2-all.pkl
```

### Output
The script will create individual `.parquet` files for each source file containing only CCC data with multi-indices preserved. Parquet format with snappy compression provides significant space savings compared to .pkl files while maintaining fast read/write performance.

### File Naming Convention
Input: `gtex_v8_data_whole_blood-var_pc_log2-all.pkl`
Output: `gtex_v8_data_whole_blood-var_pc_log2-all_ccc_only.parquet`

### Example
```bash
# Activate environment and run
conda activate ccc-gpu
python process_ccc_data.py

# Run with debug logging for more detailed information
python process_ccc_data.py --debug

# Expected output structure:
# nbs/03-manuscript/40_prepare_supp_data/
# └── logs/
#     └── process_ccc_data_YYYYMMDD_HHMMSS.log
#
# /mnt/data/proj_data/ccc-gpu/data/gtex/ccc_similarity_matrices_parquet/
# ├── gtex_v8_data_adipose_subcutaneous-var_pc_log2-all_ccc_only.parquet
# ├── gtex_v8_data_adipose_visceral_omentum-var_pc_log2-all_ccc_only.parquet
# ├── gtex_v8_data_adrenal_gland-var_pc_log2-all_ccc_only.parquet
# └── ... (54 individual .parquet files total)
``` 