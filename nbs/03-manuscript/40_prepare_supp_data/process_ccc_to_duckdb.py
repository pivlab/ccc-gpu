#!/usr/bin/env python3
"""
Convert GTEx CCC correlation data from pickle format to DuckDB format for efficient storage and fast queries.

This script processes all .pkl files containing gene correlation data and creates optimized
DuckDB databases that provide:
- Fast random access to gene pairs (sub-millisecond queries)
- Significantly reduced storage size
- SQL query capabilities
- Minimal memory usage for queries
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import pandas as pd
import duckdb
import numpy as np
from tqdm import tqdm


def setup_logging(debug: bool = False) -> str:
    """Set up logging with timestamped log file."""
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"process_ccc_to_duckdb_{timestamp}.log"

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting CCC to DuckDB conversion")
    logger.info(f"Log file: {log_filename}")

    return str(log_filename)


def convert_pickle_to_duckdb(
    pkl_file: Path,
    output_dir: Path,
    single_db: bool = False,
    db_con: Optional[duckdb.DuckDBPyConnection] = None,
    chunk_size: int = 10_000_000
) -> Dict:
    """
    Convert a single pickle file to DuckDB format.

    Args:
        pkl_file: Path to input pickle file
        output_dir: Directory for output database files
        single_db: If True, append to single database (db_con must be provided)
        db_con: Existing DuckDB connection (if single_db is True)
        chunk_size: Number of rows to process at once

    Returns:
        Dictionary with conversion statistics
    """
    logger = logging.getLogger(__name__)
    stats = {}
    start_time = datetime.now()

    # Get tissue name from filename
    tissue_name = pkl_file.stem.replace('gtex_v8_data_', '').replace('-var_pc_log2-all', '')
    stats['tissue'] = tissue_name
    stats['input_file'] = pkl_file.name

    logger.info(f"Processing: {pkl_file.name}")

    try:
        # Load pickle file
        logger.info(f"Loading pickle file...")
        load_start = datetime.now()
        df = pd.read_pickle(pkl_file)
        load_time = (datetime.now() - load_start).total_seconds()

        stats['input_rows'] = len(df)
        stats['input_size_gb'] = pkl_file.stat().st_size / (1024**3)
        logger.info(f"Loaded {len(df):,} rows in {load_time:.1f}s ({stats['input_size_gb']:.2f} GB)")

        # Extract only CCC column and reset index
        logger.info("Preparing data...")
        df_ccc = df[['ccc']].reset_index()
        df_ccc.columns = ['gene1', 'gene2', 'ccc']

        # Convert to appropriate types
        df_ccc['ccc'] = df_ccc['ccc'].astype('float32')
        df_ccc['gene1'] = df_ccc['gene1'].astype(str)
        df_ccc['gene2'] = df_ccc['gene2'].astype(str)

        # Clean up original dataframe to free memory
        del df
        gc.collect()

        # Create or connect to database
        if single_db:
            con = db_con
            table_name = f"ccc_{tissue_name}"
        else:
            db_file = output_dir / f"{tissue_name}_ccc.duckdb"
            con = duckdb.connect(str(db_file))
            table_name = "ccc_data"

        # Create table
        logger.info(f"Creating table {table_name}...")

        if single_db:
            # For single database, include tissue in the table
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    gene1 VARCHAR NOT NULL,
                    gene2 VARCHAR NOT NULL,
                    ccc REAL NOT NULL,
                    PRIMARY KEY (gene1, gene2)
                )
            """)
        else:
            con.execute(f"""
                CREATE TABLE {table_name} (
                    gene1 VARCHAR NOT NULL,
                    gene2 VARCHAR NOT NULL,
                    ccc REAL NOT NULL,
                    PRIMARY KEY (gene1, gene2)
                )
            """)

        # Insert data efficiently
        logger.info(f"Inserting {len(df_ccc):,} rows...")
        insert_start = datetime.now()

        # Use DuckDB's register for bulk insert
        con.register('df_temp', df_ccc)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df_temp")
        con.unregister('df_temp')

        insert_time = (datetime.now() - insert_start).total_seconds()
        stats['insert_time'] = insert_time
        logger.info(f"Data inserted in {insert_time:.1f}s")

        # Clean up dataframe
        del df_ccc
        gc.collect()

        # Create indexes for faster lookups
        logger.info("Creating indexes...")
        index_start = datetime.now()

        # Create index on gene2 for reverse lookups
        con.execute(f"CREATE INDEX idx_{table_name}_gene2 ON {table_name}(gene2)")

        # Create index on ccc for range queries
        con.execute(f"CREATE INDEX idx_{table_name}_ccc ON {table_name}(ccc)")

        # Analyze table for query optimization
        con.execute(f"ANALYZE {table_name}")

        index_time = (datetime.now() - index_start).total_seconds()
        stats['index_time'] = index_time

        # Get final statistics
        result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        stats['output_rows'] = result[0]

        if not single_db:
            # Close connection and get file size
            con.close()
            stats['output_size_gb'] = (output_dir / f"{tissue_name}_ccc.duckdb").stat().st_size / (1024**3)

        stats['total_time'] = (datetime.now() - start_time).total_seconds()

        if 'output_size_gb' in stats:
            stats['compression_ratio'] = stats['input_size_gb'] / stats['output_size_gb']
            logger.info(f"Completed: {stats['input_size_gb']:.2f} GB -> {stats['output_size_gb']:.2f} GB "
                       f"(compression: {stats['compression_ratio']:.1f}x)")

        logger.info(f"Total time: {stats['total_time']:.1f}s")

    except Exception as e:
        logger.error(f"Error processing {pkl_file.name}: {e}")
        stats['error'] = str(e)
        if not single_db and 'con' in locals():
            con.close()

    return stats


def create_consolidated_database(
    pkl_files: List[Path],
    output_dir: Path
) -> Dict:
    """
    Create a single consolidated DuckDB database with all tissues.

    Args:
        pkl_files: List of pickle files to process
        output_dir: Directory for output database

    Returns:
        Dictionary with overall statistics
    """
    logger = logging.getLogger(__name__)

    db_file = output_dir / "all_tissues_ccc.duckdb"
    logger.info(f"Creating consolidated database: {db_file}")

    con = duckdb.connect(str(db_file))
    all_stats = []

    try:
        # Create master table for tissue metadata
        con.execute("""
            CREATE TABLE tissues (
                tissue_id INTEGER PRIMARY KEY,
                tissue_name VARCHAR UNIQUE NOT NULL,
                num_pairs BIGINT,
                min_ccc REAL,
                max_ccc REAL,
                mean_ccc REAL
            )
        """)

        tissue_id = 1

        for pkl_file in tqdm(pkl_files, desc="Processing tissues"):
            stats = convert_pickle_to_duckdb(
                pkl_file=pkl_file,
                output_dir=output_dir,
                single_db=True,
                db_con=con
            )

            if 'error' not in stats:
                # Add tissue metadata
                tissue_name = stats['tissue']
                table_name = f"ccc_{tissue_name}"

                tissue_stats = con.execute(f"""
                    SELECT
                        COUNT(*) as num_pairs,
                        MIN(ccc) as min_ccc,
                        MAX(ccc) as max_ccc,
                        AVG(ccc) as mean_ccc
                    FROM {table_name}
                """).fetchone()

                con.execute("""
                    INSERT INTO tissues (tissue_id, tissue_name, num_pairs, min_ccc, max_ccc, mean_ccc)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [tissue_id, tissue_name, *tissue_stats])

                tissue_id += 1

            all_stats.append(stats)

        # Create a view for easy cross-tissue queries
        logger.info("Creating cross-tissue query views...")

        # Get list of all tissue tables
        tissue_tables = con.execute("""
            SELECT 'ccc_' || tissue_name as table_name, tissue_name
            FROM tissues
        """).fetchall()

        # Create UNION ALL view for searching across all tissues
        union_parts = []
        for table_name, tissue_name in tissue_tables:
            union_parts.append(f"""
                SELECT '{tissue_name}' as tissue, gene1, gene2, ccc
                FROM {table_name}
            """)

        if union_parts:
            union_query = " UNION ALL ".join(union_parts)
            con.execute(f"""
                CREATE VIEW all_correlations AS
                {union_query}
            """)

            logger.info("Created all_correlations view for cross-tissue queries")

        # Optimize database
        logger.info("Optimizing database...")
        con.execute("PRAGMA optimize")

        # Get final database size
        con.close()

        db_size = db_file.stat().st_size / (1024**3)
        logger.info(f"Consolidated database size: {db_size:.2f} GB")

        return {
            'database': str(db_file),
            'tissues_processed': len([s for s in all_stats if 'error' not in s]),
            'tissues_failed': len([s for s in all_stats if 'error' in s]),
            'total_size_gb': db_size,
            'stats': all_stats
        }

    except Exception as e:
        logger.error(f"Error creating consolidated database: {e}")
        con.close()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert GTEx CCC data from pickle to DuckDB format"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/mnt/data/proj_data/ccc-gpu/data/gtex/similarity_matrices/all",
        help="Source directory containing .pkl files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/proj_data/ccc-gpu/manuscript_data/supplementary_data/ccc_duckdb",
        help="Output directory for DuckDB files"
    )
    parser.add_argument(
        "--single-db",
        action="store_true",
        help="Create a single consolidated database instead of one per tissue"
    )
    parser.add_argument(
        "--tissues",
        nargs="+",
        help="Specific tissues to process (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    # Convert paths
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Configuration:")
    logger.info(f"  Source: {source_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Single DB: {args.single_db}")

    # Check source directory
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    # Get list of pickle files
    pkl_files = sorted(source_dir.glob("*.pkl"))

    # Filter by specific tissues if requested
    if args.tissues:
        filtered = []
        for tissue in args.tissues:
            matching = [f for f in pkl_files if tissue in f.name]
            filtered.extend(matching)
        pkl_files = filtered

    if not pkl_files:
        logger.error("No pickle files found to process")
        sys.exit(1)

    logger.info(f"Found {len(pkl_files)} files to process")

    if args.dry_run:
        print("\nFiles that would be processed:")
        for f in pkl_files:
            size_gb = f.stat().st_size / (1024**3)
            print(f"  {f.name} ({size_gb:.2f} GB)")
        print(f"\nOutput would be written to: {output_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    start_time = datetime.now()

    if args.single_db:
        # Create single consolidated database
        results = create_consolidated_database(pkl_files, output_dir)

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Database: {results['database']}")
        print(f"Tissues processed: {results['tissues_processed']}")
        print(f"Tissues failed: {results['tissues_failed']}")
        print(f"Total size: {results['total_size_gb']:.2f} GB")

    else:
        # Create individual databases
        all_stats = []

        for pkl_file in tqdm(pkl_files, desc="Processing files"):
            stats = convert_pickle_to_duckdb(
                pkl_file=pkl_file,
                output_dir=output_dir,
                single_db=False
            )
            all_stats.append(stats)

        # Summary
        successful = [s for s in all_stats if 'error' not in s]
        failed = [s for s in all_stats if 'error' in s]

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Files processed: {len(successful)}/{len(pkl_files)}")

        if successful:
            total_input = sum(s['input_size_gb'] for s in successful)
            total_output = sum(s.get('output_size_gb', 0) for s in successful)
            avg_compression = total_input / total_output if total_output > 0 else 0

            print(f"Total input size: {total_input:.2f} GB")
            print(f"Total output size: {total_output:.2f} GB")
            print(f"Average compression: {avg_compression:.1f}x")

        if failed:
            print(f"\nFailed files ({len(failed)}):")
            for s in failed:
                print(f"  {s['input_file']}: {s['error']}")

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTotal processing time: {total_time/60:.1f} minutes")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()