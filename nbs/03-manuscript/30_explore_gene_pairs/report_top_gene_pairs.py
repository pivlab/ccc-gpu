#!/usr/bin/env python3
"""
Gene Pair Analysis CLI Script

This script can operate in two modes:
1. Single analysis: Load and analyze a single tissue/combination
2. Batch analysis: Process all tissues and combinations

The batch mode:
- Scans all tissues in --data-dir
- Processes each combination within each tissue folder
- Generates top k results and logs for each combination
- Creates comprehensive logging with timestamps

Usage:
    # Single tissue/combination analysis
    python report_top_gene_pairs.py --input /path/to/sorted_data_cache.pkl --output results.txt
    python report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --top 100

    # Batch processing of all tissues and combinations
    python report_top_gene_pairs.py --data-dir /path/to/gene_pair_selection --top 1000 --batch

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import sys
import logging
import os
import shutil
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

# Global constants
EXPECTED_COMBINATIONS = [
    'c-high-p-low-s-high',
    'c-high-p-high-s-low',
    'c-high-p-low-s-low',
    'c-high-p-none-s-low',
    'c-high-p-low-s-none'
]

def setup_logging(log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup comprehensive logging configuration."""
    # Create log directory if batch processing
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler for batch processing
    if log_dir:
        file_handler = logging.FileHandler(log_dir / 'gene_pair_analysis.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_data(file_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load the gene pair dataset from a pickle file.
    
    Args:
        file_path: Path to the input pickle file
        logger: Logger instance
        
    Returns:
        DataFrame with gene pair data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
            
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def load_metadata(metadata_path: str, logger: logging.Logger) -> dict:
    """Load metadata from pickle file if it exists."""
    try:
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.debug(f"Loaded metadata from {metadata_path}")
            return metadata
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}
    except Exception as e:
        logger.warning(f"Could not load metadata from {metadata_path}: {e}")
        return {}

def find_data_file(data_dir: str, tissue: str, combination: str, logger: logging.Logger) -> str:
    """Find the sorted data cache file based on tissue and combination."""
    base_path = Path(data_dir)
    
    # Direct path approach
    candidate = base_path / tissue / combination / "sorted_data_cache.pkl"
    if candidate.exists():
        return str(candidate)
    
    # List available directories for debugging
    if (base_path / tissue).exists():
        available_combos = [d.name for d in (base_path / tissue).iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Could not find sorted data cache for tissue '{tissue}' and combination '{combination}' in {data_dir}. "
            f"Available combinations: {available_combos}"
        )
    else:
        available_tissues = [d.name for d in base_path.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Could not find tissue '{tissue}' in {data_dir}. "
            f"Available tissues: {available_tissues[:10]}..." if len(available_tissues) > 10 else f"Available tissues: {available_tissues}"
        )

def format_gene_pair_output(df: pd.DataFrame, top_n: int = 30) -> str:
    """Format the top gene pairs for text output."""
    top_df = df.head(top_n)
    
    output_lines = []
    output_lines.append(f"Top {top_n} Gene Pairs (from original sorted data)")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    # Add column headers
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
        headers = ["Gene1", "Gene2"] + list(df.columns)
    else:
        headers = ["Index"] + list(df.columns)
    
    # Format header row
    header_line = "{:<12} {:<12}".format(headers[0], headers[1])
    for col in headers[2:]:
        if col in ['ccc', 'pearson', 'spearman', 'distance_combined', 'distance_mean', 'distance_max']:
            header_line += " {:>12}".format(col)
        else:
            header_line += " {:>8}".format(col)
    output_lines.append(header_line)
    output_lines.append("-" * len(header_line))
    
    # Format data rows
    for i, (idx, row) in enumerate(top_df.iterrows()):
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
            gene1, gene2 = idx
            line = f"{gene1:<12} {gene2:<12}"
        else:
            line = f"{str(idx):<12} {'':<12}"
        
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                formatted_val = "NaN"
            elif isinstance(value, bool):
                formatted_val = "T" if value else "F"
            elif isinstance(value, (int, np.integer)):
                formatted_val = str(value)
            elif isinstance(value, (float, np.floating)):
                if col in ['ccc', 'pearson', 'spearman', 'distance_combined', 'distance_mean', 'distance_max']:
                    formatted_val = f"{value:.6f}"
                else:
                    formatted_val = f"{value:.3f}"
            else:
                formatted_val = str(value)
            
            if col in ['ccc', 'pearson', 'spearman', 'distance_combined', 'distance_mean', 'distance_max']:
                line += f" {formatted_val:>12}"
            else:
                line += f" {formatted_val:>8}"
        
        output_lines.append(line)
    
    return "\n".join(output_lines)

def export_to_csv(df: pd.DataFrame, top_n: int, output_path: str, logger: logging.Logger) -> None:
    """Export the top gene pairs to a CSV file."""
    top_df = df.head(top_n)
    
    # Reset index to make gene pairs into columns if they're multi-index
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
        top_df = top_df.reset_index()
        # Set proper column names for gene pairs if they're None
        if top_df.columns[0] is None or top_df.columns[0] == 'level_0':
            top_df.rename(columns={top_df.columns[0]: 'gene1', top_df.columns[1]: 'gene2'}, inplace=True)
    else:
        top_df = top_df.reset_index()
        top_df.rename(columns={'index': 'gene_pair_index'}, inplace=True)
    
    # Save to CSV
    top_df.to_csv(output_path, index=False)
    logger.info(f"CSV exported to: {output_path}")

def create_output_report(df: pd.DataFrame, metadata: dict, tissue: str, combination: str, top_n: int) -> str:
    """Create a complete output report with metadata and top gene pairs."""
    report_lines = []
    
    # Header
    report_lines.append("Gene Pair Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Metadata section
    report_lines.append("METADATA")
    report_lines.append("-" * 20)
    report_lines.append(f"Tissue: {tissue}")
    report_lines.append(f"Combination: {combination}")
    
    # Add metadata from the pickle file if available
    if metadata:
        for key, value in metadata.items():
            if key not in ['chosen_combination_tuple']:  # Skip complex objects
                report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
    
    report_lines.append("")
    
    # Data summary
    report_lines.append("DATA SUMMARY")
    report_lines.append("-" * 20)
    report_lines.append(f"Total gene pairs: {len(df):,}")
    report_lines.append(f"Data shape: {df.shape}")
    if hasattr(df.index, 'names'):
        report_lines.append(f"Index: {df.index.names}")
    report_lines.append(f"Columns: {list(df.columns)}")
    report_lines.append("")
    
    # Statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report_lines.append("NUMERIC COLUMN STATISTICS")
        report_lines.append("-" * 30)
        stats = df[numeric_cols].describe()
        report_lines.append(stats.to_string())
        report_lines.append("")
    
    # Top gene pairs
    report_lines.append("")
    top_output = format_gene_pair_output(df, top_n)
    report_lines.append(top_output)
    
    return "\n".join(report_lines)

def discover_tissues_and_combinations(data_dir: Path, logger: logging.Logger) -> Dict[str, List[str]]:
    """
    Discover all tissues and their combinations in the data directory.
    
    Returns:
        Dictionary mapping tissue names to list of available combinations
    """
    tissue_combinations = {}
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info(f"Scanning data directory: {data_dir}")
    
    for tissue_dir in data_dir.iterdir():
        if not tissue_dir.is_dir():
            continue
            
        tissue_name = tissue_dir.name
        combinations = []
        
        for combo_dir in tissue_dir.iterdir():
            if not combo_dir.is_dir():
                continue
                
            # Check if this combination has the required data file
            data_file = combo_dir / "sorted_data_cache.pkl"
            if data_file.exists():
                combinations.append(combo_dir.name)
        
        if combinations:
            tissue_combinations[tissue_name] = sorted(combinations)
    
    logger.info(f"Discovered {len(tissue_combinations)} tissues with combinations")
    for tissue, combos in tissue_combinations.items():
        logger.debug(f"  {tissue}: {combos}")
    
    return tissue_combinations

def process_single_combination(
    data_dir: Path, 
    tissue: str, 
    combination: str, 
    top_n: int, 
    logger: logging.Logger
) -> Tuple[str, str, int]:
    """
    Process a single tissue-combination pair.
    
    Returns:
        Tuple of (tissue, combination, processed_count)
    """
    try:
        logger.info(f"Processing {tissue} - {combination}")
        
        # Paths
        combo_dir = data_dir / tissue / combination
        data_file = combo_dir / "sorted_data_cache.pkl"
        metadata_file = combo_dir / "selection_metadata.pkl"
        
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            return tissue, combination, 0, None
        
        # Load data
        df = load_data(str(data_file), logger)
        metadata = load_metadata(str(metadata_file), logger)
        
        # Generate output files
        # Text report
        report_file = combo_dir / f"top_{top_n}_gene_pairs.txt"
        report = create_output_report(df, metadata, tissue, combination, top_n)
        with open(report_file, 'w') as f:
            f.write(report)
        
        # CSV export
        csv_file = combo_dir / f"top_{top_n}_gene_pairs.csv"
        export_to_csv(df, top_n, str(csv_file), logger)
        
        logger.info(f"Successfully processed {tissue} - {combination}: {len(df)} total pairs, {top_n} top pairs")
        return tissue, combination, top_n
        
    except Exception as e:
        logger.error(f"Error processing {tissue} - {combination}: {e}")
        return tissue, combination, 0

def batch_process_all_combinations(
    data_dir: Path,
    top_n: int,
    logger: logging.Logger,
    max_workers: int = 4
) -> None:
    """
    Process all tissue-combination pairs in batch mode.
    """
    # Discover all tissues and combinations
    tissue_combinations = discover_tissues_and_combinations(data_dir, logger)
    
    if not tissue_combinations:
        raise ValueError("No tissues with valid combinations found in data directory")
    
    # Calculate total work
    total_tasks = sum(len(combos) for combos in tissue_combinations.values())
    logger.info(f"Starting batch processing of {total_tasks} tissue-combination pairs")
    logger.info(f"Using {max_workers} parallel workers")
    
    # Prepare tasks
    tasks = []
    for tissue, combinations in tissue_combinations.items():
        for combination in combinations:
            tasks.append((data_dir, tissue, combination, top_n, logger))
    
    # Process with thread pool
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_combination, *task): task
            for task in tasks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            tissue, combination = task[1], task[2]
            
            try:
                tissue_result, combo_result, count = future.result()
                completed += 1
                
                progress = (completed / total_tasks) * 100
                logger.info(f"Progress: {completed}/{total_tasks} ({progress:.1f}%) - "
                           f"Completed: {tissue_result} - {combo_result} ({count} pairs)")
                           
            except Exception as e:
                logger.error(f"Task failed for {tissue} - {combination}: {e}")
                completed += 1
    
    logger.info(f"Batch processing completed. Processed {completed}/{total_tasks} combinations")



def main():
    parser = argparse.ArgumentParser(
        description="Gene pair analysis - single or batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single tissue/combination analysis
  python report_top_gene_pairs.py --input /path/to/sorted_data_cache.pkl --output results.txt
  python report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --top 100

  # Batch processing of all tissues and combinations
  python report_top_gene_pairs.py --data-dir /path/to/gene_pair_selection --top 1000 --batch
  python report_top_gene_pairs.py --data-dir /path/to/gene_pair_selection --top 500 --batch --workers 8
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--batch', action='store_true',
                           help='Run in batch mode - process all tissues and combinations')
    mode_group.add_argument('--input', 
                           help='Direct path to sorted_data_cache.pkl file (single mode)')
    mode_group.add_argument('--tissue', 
                           help='Tissue name (single mode, requires --combination)')
    
    # Data source
    parser.add_argument('--data-dir', 
                        default='/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection',
                        help='Base directory for gene pair selection results (default: %(default)s)')
    
    # Single mode arguments
    parser.add_argument('--combination', 
                        help='Combination name (required with --tissue)')
    
    # Output options
    parser.add_argument('--output', 
                        help='Output text file path (single mode only)')
    parser.add_argument('--csv', 
                        help='Output CSV file path (single mode only)')
    
    # Analysis parameters
    parser.add_argument('--top', 
                        type=int, 
                        default=100,
                        help='Number of top gene pairs to analyze (default: %(default)s)')
    
    # Batch processing options
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='Number of parallel workers for batch processing (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.tissue and not args.combination:
        parser.error("--combination is required when using --tissue")
    
    if not args.batch and not args.input and not args.tissue:
        parser.error("Must specify either --batch, --input, or --tissue")
    
    # Setup logging
    if args.batch:
        # Create timestamped log directory
        script_dir = Path(__file__).parent
        log_base_dir = script_dir / "logs"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = log_base_dir / f"batch_analysis_{timestamp}"
        logger = setup_logging(log_dir)
        logger.info(f"Starting batch processing with logs in: {log_dir}")
    else:
        logger = setup_logging()
    
    try:
        data_dir = Path(args.data_dir)
        
        if args.batch:
            # Batch processing mode
            logger.info("=== STARTING BATCH PROCESSING MODE ===")
            logger.info(f"Data directory: {data_dir}")
            logger.info(f"Top gene pairs per combination: {args.top}")
            logger.info(f"Parallel workers: {args.workers}")
            
            start_time = time.time()
            
            # Process all combinations
            batch_process_all_combinations(
                data_dir, args.top, logger, args.workers
            )
            
            total_time = time.time() - start_time
            logger.info(f"=== COMPLETE ===")
            logger.info(f"Batch processing completed in {total_time:.1f} seconds")
            
        else:
            # Single processing mode
            logger.info("=== SINGLE PROCESSING MODE ===")
            
            # Determine input file path
            if args.input:
                input_file = args.input
                tissue, combination = "unknown", "unknown"
                metadata_file = str(Path(args.input).parent / 'selection_metadata.pkl')
            else:
                input_file = find_data_file(args.data_dir, args.tissue, args.combination, logger)
                tissue, combination = args.tissue, args.combination
                metadata_file = str(Path(input_file).parent / 'selection_metadata.pkl')
            
            # Load data
            df = load_data(input_file, logger)
            metadata = load_metadata(metadata_file, logger)
            
            # Generate output file names if not provided
            if not args.output:
                if args.tissue and args.combination:
                    output_name = f"{tissue}_{combination}_top_{args.top}_gene_pairs.txt"
                else:
                    input_path = Path(args.input)
                    output_name = f"{input_path.stem}_top_{args.top}_gene_pairs.txt"
                args.output = output_name
            
            if not args.csv:
                if args.tissue and args.combination:
                    csv_name = f"{tissue}_{combination}_top_{args.top}_gene_pairs.csv"
                else:
                    input_path = Path(args.input)
                    csv_name = f"{input_path.stem}_top_{args.top}_gene_pairs.csv"
                args.csv = csv_name
            
            # Create report
            report = create_output_report(df, metadata, tissue, combination, args.top)
            
            # Save to file
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {args.output}")
            
            # Export to CSV
            export_to_csv(df, args.top, args.csv, logger)
            
            # Display summary to console
            print("\nTop Gene Pairs Summary:")
            print(f"Input file: {input_file}")
            print(f"Tissue: {tissue}")
            print(f"Combination: {combination}")
            print(f"Total gene pairs: {len(df):,}")
            print(f"Text output file: {args.output}")
            print(f"CSV output file: {args.csv}")
            print(f"\nFirst {min(5, args.top)} gene pairs:")
            print(format_gene_pair_output(df, min(5, args.top)))
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 