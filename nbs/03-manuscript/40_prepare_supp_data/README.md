# CCC Data Processing Scripts

This directory contains scripts to process GTEx similarity matrices (.pkl files) and extract CCC (Clustered Correlation Coefficient) data in optimized formats for efficient storage and fast queries.

## Available Scripts

### 1. `process_ccc_to_duckdb.py` (Recommended)

**Description:** Converts pickle files to DuckDB format for ultra-fast queries and efficient storage.

#### Key Features
- **Sub-millisecond query performance** for individual gene pairs
- **4-5x better compression** than parquet format
- **Minimal memory usage** (queries use <1GB RAM vs 13GB+ for parquet)
- **SQL query capabilities** for complex analyses
- **Support for both individual and consolidated databases**

#### Usage

```bash
# Activate the conda environment
conda activate ccc-gpu

# Install DuckDB (if not already installed)
pip install duckdb

# Process all tissues into individual databases
python process_ccc_to_duckdb.py \
    --source-dir /mnt/data/proj_data/ccc-gpu/data/gtex/similarity_matrices/all \
    --output-dir /mnt/data/proj_data/ccc-gpu/manuscript_data/supplementary_data/ccc_duckdb

# Create a single consolidated database (all tissues)
python process_ccc_to_duckdb.py --single-db

# Process specific tissues only
python process_ccc_to_duckdb.py --tissues bladder brain_cortex

# Dry run to see what would be processed
python process_ccc_to_duckdb.py --dry-run
```

#### Arguments
- `--source-dir`: Source directory with .pkl files
- `--output-dir`: Output directory for DuckDB files
- `--single-db`: Create one consolidated database instead of individual ones
- `--tissues`: Process specific tissues only
- `--dry-run`: Show what would be processed without doing it
- `--debug`: Enable debug logging

### 2. `ccc_duckdb_query.py` - Query Interface

**Description:** Python wrapper for fast queries on DuckDB databases.

#### Usage as Module

```python
from ccc_duckdb_query import CCCDatabase

# Open database
db = CCCDatabase("/path/to/bladder_ccc.duckdb")

# Query single gene pair
ccc = db.get_correlation("ENSG00000141510.16", "ENSG00000133703.11")

# Get all correlations for a gene
correlations = db.get_gene_correlations("ENSG00000141510.16", min_ccc=0.5)

# Get top correlations
top_pairs = db.get_top_correlations(threshold=0.9, limit=100)

# Batch query multiple pairs
pairs = [("gene1", "gene2"), ("gene3", "gene4")]
results = db.get_batch_correlations(pairs)

# Custom SQL query
df = db.query("SELECT * FROM ccc_data WHERE ccc > 0.95 LIMIT 10")

# Get database statistics
stats = db.get_statistics()

db.close()
```

#### Usage as CLI

```bash
# Get database statistics
python ccc_duckdb_query.py /path/to/database.duckdb --stats

# Query specific gene pair
python ccc_duckdb_query.py /path/to/database.duckdb \
    --gene1 ENSG00000141510.16 --gene2 ENSG00000133703.11

# Get correlations for a gene
python ccc_duckdb_query.py /path/to/database.duckdb \
    --gene ENSG00000141510.16 --limit 50

# Get top correlations
python ccc_duckdb_query.py /path/to/database.duckdb \
    --top 0.9 --limit 100
```

### 3. `process_ccc_data.py` (Legacy - Parquet Output)

**Description:** Original script that creates parquet files. Kept for compatibility but DuckDB format is recommended.

```bash
# Run with default paths
python process_ccc_data.py

# Custom paths
python process_ccc_data.py --source-dir /path/to/source --output-dir /path/to/output
```

## Performance Comparison

| Metric | Parquet | DuckDB | Improvement |
|--------|---------|---------|-------------|
| Storage Size | 302 GB | ~60-80 GB | 4-5x smaller |
| Load Time | >60s timeout | 0s (no loading) | Instant access |
| Single Query | >100ms | <1ms | 100x+ faster |
| Memory Usage | 13+ GB | <1 GB | 13x+ less |
| Random Access | Very slow | Sub-millisecond | Orders of magnitude |

## Requirements

```bash
# Core requirements
conda activate ccc-gpu
pip install pandas duckdb numpy tqdm

# Optional for parquet support
pip install pyarrow
```

## Output Structure

### DuckDB Format (Recommended)
```
/output_directory/
├── bladder_ccc.duckdb          # Individual tissue databases
├── brain_cortex_ccc.duckdb
├── whole_blood_ccc.duckdb
└── all_tissues_ccc.duckdb      # Optional consolidated database
```

### Database Schema
```sql
-- Individual tissue table
CREATE TABLE ccc_data (
    gene1 VARCHAR NOT NULL,
    gene2 VARCHAR NOT NULL,
    ccc REAL NOT NULL,
    PRIMARY KEY (gene1, gene2)
);

-- Indexes for fast lookups
CREATE INDEX idx_gene2 ON ccc_data(gene2);
CREATE INDEX idx_ccc ON ccc_data(ccc);
```

## Example Workflow

```bash
# 1. Convert all pickle files to DuckDB
conda activate ccc-gpu
python process_ccc_to_duckdb.py

# 2. Test query performance
python ccc_duckdb_query.py /path/to/bladder_ccc.duckdb --stats

# 3. Use in Python scripts
from ccc_duckdb_query import CCCDatabase

with CCCDatabase("bladder_ccc.duckdb") as db:
    # Fast queries for your analysis
    ccc = db.get_correlation("gene1", "gene2")
```

## Advantages of DuckDB Format

1. **No Loading Required**: Direct queries without loading entire dataset
2. **Memory Efficient**: Uses memory-mapped IO, minimal RAM footprint
3. **Fast Random Access**: Indexed lookups in microseconds
4. **SQL Support**: Complex queries and aggregations
5. **Better Compression**: Columnar storage with efficient encoding
6. **Concurrent Access**: Multiple readers can query simultaneously
7. **ACID Compliance**: Data integrity guarantees 