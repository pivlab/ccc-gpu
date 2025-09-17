#!/usr/bin/env python3
"""
CLI script to process GTEx similarity matrices (.pkl files) and extract CCC data.

This script processes all .pkl files in the source directory, extracts only the 'ccc' 
column with multi-indices, and saves individual .parquet files with snappy compression for each input.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Progress bars will not be displayed.")
    # Fallback tqdm that just returns the iterable
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

try:
    import pyarrow
except ImportError:
    print("Warning: pyarrow not found. Please install it for parquet support: pip install pyarrow")
    sys.exit(1)


def setup_logging() -> str:
    """Set up logging with timestamped log file."""
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Create logs directory
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"process_ccc_data_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CCC data processing")
    logger.info(f"Log file: {log_filename}")
    
    return str(log_filename)


def setup_output_directory(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {output_dir}")
    print(f"Output directory: {output_dir}")


def get_pkl_files(source_dir: Path) -> List[Path]:
    """Get all .pkl files from the source directory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning directory for .pkl files: {source_dir}")
    
    pkl_files = list(source_dir.glob("*.pkl"))
    pkl_files.sort()  # Sort for consistent processing order
    
    logger.info(f"Found {len(pkl_files)} .pkl files to process")
    print(f"Found {len(pkl_files)} .pkl files to process")
    
    # Log the first few filenames for verification
    if pkl_files:
        logger.debug(f"First few files: {[f.name for f in pkl_files[:5]]}")
    
    return pkl_files


def process_single_file(pkl_file: Path, output_dir: Path) -> Path:
    """
    Process a single .pkl file and extract CCC data.
    
    Returns:
        Path to output .parquet file (or None if failed)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load the data
        logger.info(f"Processing file: {pkl_file.name}")
        print(f"Processing: {pkl_file.name}")
        
        data = pd.read_pickle(pkl_file)
        logger.debug(f"Loaded data shape: {data.shape}, columns: {data.columns.tolist()}")
        
        # Extract only the 'ccc' column
        if 'ccc' not in data.columns:
            error_msg = f"'ccc' column not found in {pkl_file.name}. Available columns: {data.columns.tolist()}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        ccc_data = data[['ccc']].copy()
        logger.debug(f"Extracted CCC data shape: {ccc_data.shape}")
        
        # Generate output filename
        base_name = pkl_file.stem  # Remove .pkl extension
        output_parquet = output_dir / f"{base_name}_ccc_only.parquet"
        
        # Save as .parquet with snappy compression
        logger.debug(f"Saving .parquet file: {output_parquet}")
        ccc_data.to_parquet(output_parquet, compression='snappy')
        
        # Log file sizes for comparison
        original_size = pkl_file.stat().st_size / (1024**3)  # GB
        output_size = output_parquet.stat().st_size / (1024**3)  # GB
        compression_ratio = (1 - output_size/original_size) * 100
        
        logger.info(f"Successfully processed {pkl_file.name} -> {output_parquet.name}")
        logger.info(f"Size reduction: {original_size:.2f} GB -> {output_size:.2f} GB ({compression_ratio:.1f}% smaller)")
        print(f"  Saved: {output_parquet.name} ({original_size:.2f}GB -> {output_size:.2f}GB)")
        return output_parquet
        
    except Exception as e:
        error_msg = f"Error processing {pkl_file.name}: {e}"
        logger.error(error_msg)
        print(f"  {error_msg}")
        return None, None





def main():
    """Main function to orchestrate the processing."""
    parser = argparse.ArgumentParser(
        description="Process GTEx similarity matrices and extract CCC data"
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
        default="/mnt/data/proj_data/ccc-gpu/data/gtex/ccc_similarity_matrices_parquet",
        help="Output directory for processed parquet files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging first
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Convert to Path objects
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Script arguments: source_dir={source_dir}, output_dir={output_dir}, dry_run={args.dry_run}")
    
    # Check source directory exists
    if not source_dir.exists():
        error_msg = f"Source directory does not exist: {source_dir}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        sys.exit(1)
    
    # Get list of .pkl files
    pkl_files = get_pkl_files(source_dir)
    
    if not pkl_files:
        error_msg = "No .pkl files found in the source directory"
        logger.error(error_msg)
        print(error_msg)
        sys.exit(1)
    
    if args.dry_run:
        logger.info("Dry run mode - listing files that would be processed")
        print("\nDry run - files that would be processed:")
        for pkl_file in pkl_files:
            print(f"  {pkl_file.name}")
        print(f"\nOutput directory would be: {output_dir}")
        logger.info("Dry run completed")
        return
    
    # Setup output directory
    setup_output_directory(output_dir)
    
    # Process all files
    logger.info(f"Starting processing of {len(pkl_files)} files")
    print(f"\nProcessing {len(pkl_files)} files...")
    output_parquet_files = []
    
    start_time = datetime.now()
    for i, pkl_file in enumerate(tqdm(pkl_files, desc="Processing files"), 1):
        logger.debug(f"Processing file {i}/{len(pkl_files)}: {pkl_file.name}")
        output_parquet = process_single_file(pkl_file, output_dir)
        output_parquet_files.append(output_parquet)
    
    processing_time = datetime.now() - start_time
    logger.info(f"File processing completed in {processing_time}")
    
    # Print summary
    successful_files = sum(1 for f in output_parquet_files if f is not None)
    failed_files = len(pkl_files) - successful_files
    
    total_time = datetime.now() - start_time
    
    summary_msg = f"Processing complete! Successfully processed: {successful_files}/{len(pkl_files)} files"
    if failed_files > 0:
        summary_msg += f" ({failed_files} failed)"
    
    logger.info(summary_msg)
    logger.info(f"Total execution time: {total_time}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    print(f"\n{summary_msg}")
    if failed_files > 0:
        print(f"Failed files: {failed_files}")
    print(f"Total time: {total_time}")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main() 