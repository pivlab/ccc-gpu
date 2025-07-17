#!/usr/bin/env python3
"""
Gene Pair Counter CLI Script

This script processes multiple gene pair intersection datasets and counts
how many gene pairs exist for each unique combination of boolean indicators.
It saves individual count files for each tissue and generates accumulated bar plots.

The dataset should have columns:
- Pearson (high), Pearson (low), Spearman (high), Spearman (low), 
  Clustermatch (high), Clustermatch (low) - boolean columns
- ccc, pearson, spearman - numeric columns

Usage:
    python gene_pair_counter.py input_directory output_base_name

Output:
    - Individual tissue count files: {tissue}_counts.pkl
    - Accumulated bar plots (if requested)
    - Intermediate processing files

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import logging
import argparse
import pickle
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Initialize logger (will be configured later)
logger = None

def create_timestamped_folder() -> Path:
    """
    Create a timestamped folder for outputs.
    
    Returns:
        Path to the created folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = Path(timestamp)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path

def setup_logging(log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file. If None, uses default name based on output file.
    """
    if log_file is None:
        log_file = 'gene_pair_analysis.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the gene pair dataset from a pickle file.
    
    Args:
        file_path: Path to the input pickle file
        
    Returns:
        DataFrame with multi-index gene pairs
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Load pickle file
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
            
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        logger.info(f"Index type: {type(df.index)}")
        if hasattr(df.index, 'names'):
            logger.info(f"Multi-index levels: {df.index.names}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset has the expected structure.
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if validation passes
    """
    logger.info("Validating data structure...")
    
    # Expected boolean columns
    expected_bool_cols = [
        'Pearson (high)', 'Pearson (low)', 
        'Spearman (high)', 'Spearman (low)',
        'Clustermatch (high)', 'Clustermatch (low)'
    ]
    
    # Expected numeric columns
    expected_numeric_cols = ['ccc', 'pearson', 'spearman']
    
    # Check if all expected columns exist
    missing_cols = []
    for col in expected_bool_cols + expected_numeric_cols:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
        return False
    
    # Check if boolean columns are actually boolean
    for col in expected_bool_cols:
        if not df[col].dtype == 'bool':
            logger.warning(f"Column '{col}' is not boolean type. Current type: {df[col].dtype}")
            # Try to convert to boolean
            try:
                df[col] = df[col].astype(bool)
                logger.info(f"Successfully converted '{col}' to boolean")
            except Exception as e:
                logger.error(f"Failed to convert '{col}' to boolean: {e}")
                return False
    
    # Check if numeric columns are numeric
    for col in expected_numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' is not numeric. Current type: {df[col].dtype}")
    
    # Check for multi-index
    if not isinstance(df.index, pd.MultiIndex):
        logger.error("Data does not have a multi-index structure")
        return False
    
    logger.info("Data validation passed!")
    return True

def process_chunk(chunk_data: Tuple[pd.DataFrame, int]) -> pd.DataFrame:
    """
    Process a chunk of data to count gene pairs by indicators.
    
    Args:
        chunk_data: Tuple of (dataframe_chunk, chunk_index)
        
    Returns:
        DataFrame with counts for each indicator combination in this chunk
    """
    chunk_df, chunk_idx = chunk_data
    
    # Define the boolean indicator columns
    indicator_cols = [
        'Pearson (high)', 'Pearson (low)', 
        'Spearman (high)', 'Spearman (low)',
        'Clustermatch (high)', 'Clustermatch (low)'
    ]
    
    # Group by the indicator columns and count
    grouped = chunk_df.groupby(indicator_cols).size().reset_index(name='gene_pair_count')
    
    return grouped

def count_gene_pairs_by_indicators(df: pd.DataFrame, n_threads: int = None) -> pd.DataFrame:
    """
    Count gene pairs for each unique combination of boolean indicators using parallel processing.
    
    Args:
        df: Input DataFrame with gene pairs
        n_threads: Number of threads to use (default: CPU count)
        
    Returns:
        DataFrame with counts for each indicator combination
    """
    logger.info("Starting gene pair counting by indicator combinations...")
    
    # Define the boolean indicator columns
    indicator_cols = [
        'Pearson (high)', 'Pearson (low)', 
        'Spearman (high)', 'Spearman (low)',
        'Clustermatch (high)', 'Clustermatch (low)'
    ]
    
    logger.info(f"Grouping by indicator columns: {indicator_cols}")
    
    # Determine number of threads
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    # For small datasets, don't use parallelization overhead
    # Increased threshold as multiprocessing overhead can be significant
    if len(df) < 50000:
        logger.info(f"Dataset size ({len(df)}) below parallelization threshold (50,000), using single-threaded processing")
        grouped = df.groupby(indicator_cols).size().reset_index(name='gene_pair_count')
    else:
        logger.info(f"Using {n_threads} threads for parallel processing")
        logger.info(f"Processing {len(df)} gene pairs...")
        
        # For this type of operation, ThreadPoolExecutor is often more efficient than ProcessPoolExecutor
        # as it avoids the overhead of pickling/unpickling data between processes
        use_threads = len(df) < 500000  # Use threads for medium datasets, processes for very large ones
        
        if use_threads:
            logger.info("Using ThreadPoolExecutor for optimal performance")
            executor_class = ThreadPoolExecutor
        else:
            logger.info("Using ProcessPoolExecutor for very large dataset")
            executor_class = ProcessPoolExecutor
        
        # Split DataFrame into chunks
        # Use fewer chunks to reduce overhead
        effective_threads = min(n_threads, max(2, len(df) // 25000))  # At least 25k rows per chunk
        chunk_size = max(1, len(df) // effective_threads)
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunks.append((chunk, i // chunk_size))
        
        logger.info(f"Split data into {len(chunks)} chunks of ~{chunk_size} rows each")
        logger.info(f"Using {effective_threads} effective threads")
        
        # Process chunks in parallel
        chunk_results = []
        with executor_class(max_workers=effective_threads) as executor:
            # Submit all chunks for processing
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future][1]
                try:
                    result = future.result()
                    chunk_results.append(result)
                    logger.info(f"Completed processing chunk {chunk_idx}")
                except Exception as exc:
                    logger.error(f"Chunk {chunk_idx} generated an exception: {exc}")
                    raise
        
        # Combine results from all chunks
        logger.info("Combining results from all chunks...")
        if chunk_results:
            # Concatenate all chunk results
            combined_df = pd.concat(chunk_results, ignore_index=True)
            
            # Group by indicator columns again to sum counts from different chunks
            grouped = combined_df.groupby(indicator_cols)['gene_pair_count'].sum().reset_index()
        else:
            # Empty result
            grouped = pd.DataFrame(columns=indicator_cols + ['gene_pair_count'])
    
    logger.info(f"Found {len(grouped)} unique indicator combinations")
    logger.info(f"Total gene pairs processed: {grouped['gene_pair_count'].sum()}")
    
    # Sort by count (descending) for better readability
    grouped = grouped.sort_values('gene_pair_count', ascending=False)
    
    # Log some statistics
    if len(grouped) > 0:
        logger.info(f"Most common combination has {grouped['gene_pair_count'].max()} gene pairs")
        logger.info(f"Least common combination has {grouped['gene_pair_count'].min()} gene pairs")
        logger.info(f"Average gene pairs per combination: {grouped['gene_pair_count'].mean():.2f}")
    
    return grouped

def display_results(results_df: pd.DataFrame, top_n: int = 10):
    """
    Display the results in a readable format.
    
    Args:
        results_df: DataFrame with counting results
        top_n: Number of top combinations to display
    """
    logger.info(f"Displaying top {top_n} most common indicator combinations:")
    
    logger.info("\n" + "="*80)
    logger.info("GENE PAIR COUNTING RESULTS")
    logger.info("="*80)
    
    for i, (_, row) in enumerate(results_df.head(top_n).iterrows()):
        logger.info(f"\nRank {i+1}:")
        logger.info(f"  Gene pairs: {row['gene_pair_count']}")
        logger.info(f"  Indicators:")
        logger.info(f"    Pearson (high): {row['Pearson (high)']}")
        logger.info(f"    Pearson (low): {row['Pearson (low)']}")
        logger.info(f"    Spearman (high): {row['Spearman (high)']}")
        logger.info(f"    Spearman (low): {row['Spearman (low)']}")
        logger.info(f"    Clustermatch (high): {row['Clustermatch (high)']}")
        logger.info(f"    Clustermatch (low): {row['Clustermatch (low)']}")
        logger.info("-" * 50)

def format_number_with_units(number: int) -> str:
    """
    Format a number with K, M, B units and exactly 3 digits including decimal places.
    
    Args:
        number: Integer to format
        
    Returns:
        Formatted string with units
    """
    if number >= 1_000_000_000:
        # Billions
        formatted = number / 1_000_000_000
        if formatted >= 100:
            return f"{formatted:.0f}B"
        elif formatted >= 10:
            return f"{formatted:.1f}B"
        else:
            return f"{formatted:.2f}B"
    elif number >= 1_000_000:
        # Millions
        formatted = number / 1_000_000
        if formatted >= 100:
            return f"{formatted:.0f}M"
        elif formatted >= 10:
            return f"{formatted:.1f}M"
        else:
            return f"{formatted:.2f}M"
    elif number >= 1_000:
        # Thousands
        formatted = number / 1_000
        if formatted >= 100:
            return f"{formatted:.0f}K"
        elif formatted >= 10:
            return f"{formatted:.1f}K"
        else:
            return f"{formatted:.2f}K"
    else:
        # Less than 1000, no unit needed
        return str(number)

def create_bar_plot(results_df: pd.DataFrame, output_path: str) -> Optional[str]:
    """
    Create a bar plot of gene pair counts by indicator combinations with upset plot-style indicators.
    
    Args:
        results_df: DataFrame with counting results
        output_path: Path to save the plot
        
    Returns:
        Path to saved plot or None if no plot was created
    """
    logger.info("Creating bar plot...")
    
    # Define the specific order for combinations
    # Order: [Spearman (low), Pearson (low), Clustermatch (low), Spearman (high), Pearson (high), Clustermatch (high)]
    ordered_combinations = [
        (False, False, False, True, True, True),
        # agreements on top
        (False, False, False, False, True, True),
        (False, False, False, True, False, True),
        (False, False, False, True, True, False),
        # agreements on bottom
        (False, True, True, False, False, False),
        (True, False, True, False, False, False),
        (True, True, False, False, False, False),
        # full agreements on low:
        (True, True, True, False, False, False),
        # disagreements
        #   ccc
        (False, True, False, False, False, True),
        (True, False, False, False, False, True),
        (True, True, False, False, False, True),
        (True, False, False, False, True, True),
        (False, True, False, True, False, True),
        #   pearson
        (False, False, True, False, True, False),
        (True, False, True, False, True, False),
        (True, False, False, False, True, False),
        (False, False, True, True, True, False),
        #   spearman
        (False, False, True, True, False, False),
        (False, True, True, True, False, False),
        (False, True, False, True, False, False),
    ]
    
    # Convert results to a lookup dictionary
    # The results DataFrame has columns in this order: 
    # ['Pearson (high)', 'Pearson (low)', 'Spearman (high)', 'Spearman (low)', 'Clustermatch (high)', 'Clustermatch (low)', 'gene_pair_count']
    
    results_lookup = {}
    for _, row in results_df.iterrows():
        # Map from DataFrame column order to our desired order
        # DataFrame: [Pearson (high), Pearson (low), Spearman (high), Spearman (low), Clustermatch (high), Clustermatch (low)]
        # Our order: [Spearman (low), Pearson (low), Clustermatch (low), Spearman (high), Pearson (high), Clustermatch (high)]
        key = (
            row['Spearman (low)'],     # 0
            row['Pearson (low)'],      # 1
            row['Clustermatch (low)'], # 2
            row['Spearman (high)'],    # 3
            row['Pearson (high)'],     # 4
            row['Clustermatch (high)'] # 5
        )
        results_lookup[key] = row['gene_pair_count']
    
    # Extract counts for ordered combinations, skipping missing ones
    plot_data = []
    plot_labels = []
    
    for combination in ordered_combinations:
        if combination in results_lookup:
            plot_data.append(results_lookup[combination])
            
            # Create descriptive label
            spearman_low, pearson_low, clustermatch_low, spearman_high, pearson_high, clustermatch_high = combination
            label_parts = []
            
            # Add "high" indicators
            if spearman_high:
                label_parts.append("Sp↑")
            if pearson_high:
                label_parts.append("Pe↑")
            if clustermatch_high:
                label_parts.append("Cl↑")
            
            # Add "low" indicators
            if spearman_low:
                label_parts.append("Sp↓")
            if pearson_low:
                label_parts.append("Pe↓")
            if clustermatch_low:
                label_parts.append("Cl↓")
            
            if not label_parts:
                label_parts.append("None")
            
            plot_labels.append(" ".join(label_parts))
    
    if not plot_data:
        logger.warning("No data to plot")
        return None
    
    # Calculate percentages
    total_count = sum(plot_data)
    percentages = [count / total_count * 100 for count in plot_data]
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Define colors for different categories
    colors = []
    for combination in ordered_combinations:
        if combination not in results_lookup:
            continue
            
        spearman_low, pearson_low, clustermatch_low, spearman_high, pearson_high, clustermatch_high = combination
        
        # Color coding based on pattern
        if (spearman_high, pearson_high, clustermatch_high) == (True, True, True):
            colors.append('#2E8B57')  # Sea green - all high
        elif (spearman_low, pearson_low, clustermatch_low) == (True, True, True):
            colors.append('#DC143C')  # Crimson - all low
        elif sum([spearman_high, pearson_high, clustermatch_high]) > sum([spearman_low, pearson_low, clustermatch_low]):
            colors.append('#4682B4')  # Steel blue - more high
        elif sum([spearman_low, pearson_low, clustermatch_low]) > sum([spearman_high, pearson_high, clustermatch_high]):
            colors.append('#FF6347')  # Tomato - more low
        else:
            colors.append('#9370DB')  # Medium purple - mixed
    
    # Create bar plot
    bars = plt.bar(range(len(plot_data)), plot_data, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title('Gene Pair Counts by Indicator Combinations', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Indicator Combinations', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Gene Pairs', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(plot_labels)), plot_labels, rotation=45, ha='right')
    
    # Add value labels on bars with counts and percentages
    for bar, value, percentage in zip(bars, plot_data, percentages):
        height = bar.get_height()
        # Count on top
        plt.text(bar.get_x() + bar.get_width()/2., height + max(plot_data)*0.01,
                format_number_with_units(value), ha='center', va='bottom', fontweight='bold')
        # Percentage below the count
        plt.text(bar.get_x() + bar.get_width()/2., height + max(plot_data)*0.04,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='gray')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    legend_elements = [
        patches.Patch(color='#2E8B57', label='All High'),
        patches.Patch(color='#DC143C', label='All Low'),
        patches.Patch(color='#4682B4', label='More High'),
        patches.Patch(color='#FF6347', label='More Low'),
        patches.Patch(color='#9370DB', label='Mixed')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_path.replace('.pkl', '_barplot.svg')
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Bar plot saved to {plot_path}")
    return plot_path

def save_results(results_df: pd.DataFrame, output_path: str):
    """
    Save the results to a pickle file.
    
    Args:
        results_df: DataFrame with counting results
        output_path: Path to save the output pickle file
    """
    try:
        logger.info(f"Saving results to {output_path}")
        
        # Save as pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(results_df, f)
            
        logger.info(f"Results successfully saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Count gene pairs by indicator combinations from multiple intersection files with parallel processing support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python gene_pair_counter.py --data-dir /path/to/intersections output_counts.pkl
    python gene_pair_counter.py --data-dir /data/intersections results.pkl --top-n 5
    python gene_pair_counter.py --data-dir /data/intersections results.pkl --log-file analysis.log
    python gene_pair_counter.py --data-dir /data/intersections results.pkl --plot
    python gene_pair_counter.py --data-dir /data/intersections results.pkl --threads 8
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to directory containing gene_pair_intersections*.pkl files'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output pickle file for results'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=25,
        help='Number of top combinations to display (default: 25)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: gene_pair_analysis.log)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Skip displaying results to console'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate bar plot of results (SVG format)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=None,
        help='Number of threads to use for parallel processing (default: CPU count)'
    )
    
    return parser.parse_args()

def find_intersection_files(data_dir: str) -> List[Path]:
    """
    Find all gene_pair_intersections*.pkl files in the specified directory.
    
    Args:
        data_dir: Path to directory containing intersection files
        
    Returns:
        List of Path objects for found files, sorted by name
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data directory is not a directory: {data_dir}")
    
    # Find all matching files
    pattern = "gene_pair_intersections*.pkl"
    intersection_files = list(data_path.glob(pattern))
    
    if not intersection_files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {data_dir}")
    
    # Sort files by name for consistent processing order
    intersection_files.sort()
    
    return intersection_files

def accumulate_counts(existing_counts: pd.DataFrame, new_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Accumulate counts from a new file into existing counts.
    
    Args:
        existing_counts: DataFrame with existing accumulated counts
        new_counts: DataFrame with new counts to add
        
    Returns:
        DataFrame with accumulated counts
    """
    if existing_counts is None or existing_counts.empty:
        return new_counts.copy()
    
    # Merge the dataframes, adding counts for matching combinations
    # and keeping unique combinations from both
    merged = pd.merge(
        existing_counts, 
        new_counts, 
        on=['Pearson (high)', 'Pearson (low)', 'Spearman (high)', 'Spearman (low)', 
            'Clustermatch (high)', 'Clustermatch (low)'],
        how='outer',
        suffixes=('_existing', '_new')
    )
    
    # Fill NaN values with 0 and sum the counts
    merged['gene_pair_count_existing'] = merged['gene_pair_count_existing'].fillna(0)
    merged['gene_pair_count_new'] = merged['gene_pair_count_new'].fillna(0)
    merged['gene_pair_count'] = merged['gene_pair_count_existing'] + merged['gene_pair_count_new']
    
    # Keep only the necessary columns
    result = merged[['Pearson (high)', 'Pearson (low)', 'Spearman (high)', 'Spearman (low)', 
                    'Clustermatch (high)', 'Clustermatch (low)', 'gene_pair_count']]
    
    return result

def extract_tissue_name(file_path: Path) -> str:
    """
    Extract tissue name from intersection file path.
    
    Args:
        file_path: Path to intersection file
        
    Returns:
        Tissue name extracted from filename
    """
    # Expected format: gene_pair_intersections-gtex_v8-{tissue}-var_pc_log2.pkl
    filename = file_path.name
    
    # Remove extension and prefix
    name_parts = filename.replace('.pkl', '').split('-')
    
    if len(name_parts) >= 3:
        # Extract tissue name (3rd part after splitting by '-')
        tissue_name = name_parts[2]
        return tissue_name
    
    # Fallback: use filename without extension
    return filename.replace('.pkl', '')

def process_multiple_files(data_dir: str, output_folder: Path, output_file: str, 
                          args) -> pd.DataFrame:
    """
    Process multiple intersection files and accumulate results.
    
    Args:
        data_dir: Directory containing intersection files
        output_folder: Folder for output files
        output_file: Base name for output file
        args: Command line arguments
        
    Returns:
        DataFrame with accumulated results
    """
    # Find all intersection files
    intersection_files = find_intersection_files(data_dir)
    
    logger.info(f"Found {len(intersection_files)} intersection files to process")
    logger.info("Files to process:")
    for i, file_path in enumerate(intersection_files, 1):
        logger.info(f"  {i:2d}. {file_path.name}")
    
    # Initialize accumulator
    accumulated_counts = None
    processed_files = 0
    total_gene_pairs = 0
    
    # Process each file
    for i, file_path in enumerate(intersection_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING FILE {i}/{len(intersection_files)}: {file_path.name}")
        logger.info(f"{'='*80}")
        
        try:
            # Load and process current file
            logger.info(f"Loading data from {file_path}")
            df = load_data(str(file_path))
            
            if df is None or df.empty:
                logger.warning(f"File {file_path.name} is empty or invalid, skipping")
                continue
            
            # Validate data structure
            if not validate_data(df):
                logger.error(f"Data validation failed for {file_path.name}, skipping")
                continue
            
            logger.info(f"File contains {len(df)} gene pairs")
            logger.info(f"Unique gene pairs: {df.index.nunique()}")
            
            # Count gene pairs for this file
            logger.info("Counting gene pairs by indicator combinations...")
            file_counts = count_gene_pairs_by_indicators(df, n_threads=args.threads)
            
            if file_counts is None or file_counts.empty:
                logger.warning(f"No counts generated for {file_path.name}, skipping")
                continue
            
            # Extract tissue name and save individual tissue counts
            tissue_name = extract_tissue_name(file_path)
            tissue_count_file = output_folder / f"{tissue_name}_counts.pkl"
            logger.info(f"Saving tissue-specific counts to {tissue_count_file}")
            with open(tissue_count_file, 'wb') as f:
                pickle.dump(file_counts, f)
            
            # Accumulate counts for plotting
            logger.info("Accumulating counts with previous files...")
            accumulated_counts = accumulate_counts(accumulated_counts, file_counts)
            
            processed_files += 1
            total_gene_pairs += len(df)
            
            # Log progress
            logger.info(f"File {i} processed successfully")
            logger.info(f"New combinations found: {len(file_counts)}")
            logger.info(f"Total combinations so far: {len(accumulated_counts)}")
            logger.info(f"Total gene pairs processed: {total_gene_pairs:,}")
            logger.info(f"Tissue-specific counts saved to {tissue_count_file}")
            
            # Save intermediate results for plotting
            intermediate_file = output_folder / f"intermediate_counts_after_{i:02d}_files.pkl"
            logger.info(f"Saving intermediate results to {intermediate_file}")
            with open(intermediate_file, 'wb') as f:
                pickle.dump(accumulated_counts, f)
            
            # Generate intermediate plot if requested
            if args.plot:
                logger.info("Generating intermediate plot...")
                intermediate_plot_file = str(output_folder / f"intermediate_plot_after_{i:02d}_files.pkl")
                plot_path = create_bar_plot(accumulated_counts, intermediate_plot_file)
                if plot_path:
                    logger.info(f"Intermediate plot saved to {plot_path}")
                else:
                    logger.warning("Failed to generate intermediate plot")
            
            # Display current top combinations
            if not args.no_display:
                logger.info(f"\nAll {len(accumulated_counts)} combinations so far:")
                display_results(accumulated_counts, top_n=len(accumulated_counts))
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            logger.error(f"Skipping file and continuing with next...")
            continue
    
    if accumulated_counts is None or accumulated_counts.empty:
        logger.error("No valid data processed from any files")
        return None
    
    logger.info(f"\n{'='*80}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successfully processed {processed_files}/{len(intersection_files)} files")
    logger.info(f"Total gene pairs processed: {total_gene_pairs:,}")
    logger.info(f"Total unique combinations: {len(accumulated_counts)}")
    logger.info(f"Individual tissue count files saved in {output_folder}")
    
    return accumulated_counts

def main():
    """
    Main function to orchestrate the gene pair counting process from multiple files.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create timestamped folder for outputs
    output_folder = create_timestamped_folder()
    
    # Setup output paths in the timestamped folder
    output_filename = Path(args.output_file).name
    output_file_path = output_folder / output_filename
    
    # Setup logging
    log_file = args.log_file
    if log_file is None:
        # Create log file name based on output file in timestamped folder
        log_file = output_folder / f"{Path(output_filename).stem}_analysis.log"
    else:
        # Put custom log file in timestamped folder too
        log_file = output_folder / Path(log_file).name
    
    setup_logging(str(log_file))
    
    # Get logger after setup
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Gene Pair Counter CLI Script - Multiple Files Mode")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Plot base name: {output_file_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Generate plots: {args.plot}")
    logger.info(f"Threads: {args.threads if args.threads else 'auto (CPU count)'}")
    logger.info(f"Top N results: {args.top_n}")
    
    try:
        # Validate data directory exists
        if not Path(args.data_dir).exists():
            logger.error(f"Data directory does not exist: {args.data_dir}")
            sys.exit(1)
        
        # Process multiple files and accumulate results
        results = process_multiple_files(args.data_dir, output_folder, str(output_file_path), args)
        
        if results is None or results.empty:
            logger.error("No valid results obtained from any files. Exiting.")
            sys.exit(1)
        
        # Display final results (unless --no-display is specified)
        if not args.no_display:
            logger.info(f"\n{'='*80}")
            logger.info("FINAL RESULTS - TOP COMBINATIONS")
            logger.info(f"{'='*80}")
            display_results(results, top_n=args.top_n)
        
        # Generate final plot if requested
        final_plot_path = None
        if args.plot:
            logger.info("Generating final comprehensive plot...")
            final_plot_path = create_bar_plot(results, str(output_file_path))
            if final_plot_path:
                logger.info(f"Final plot saved to {final_plot_path}")
        
        logger.info("Gene pair counting from multiple files completed successfully!")
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE SUMMARY")
        print(f"{'='*80}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output folder: {output_folder}")
        print(f"Log file: {log_file}")
        if final_plot_path:
            print(f"Final plot: {final_plot_path}")
        
        # Count output files
        tissue_count_files = list(output_folder.glob("*_counts.pkl"))
        intermediate_files = list(output_folder.glob("intermediate_*.pkl"))
        intermediate_plots = list(output_folder.glob("intermediate_*.svg"))
        
        print(f"Tissue-specific count files: {len(tissue_count_files)}")
        print(f"Intermediate count files: {len(intermediate_files)}")
        print(f"Intermediate plots: {len(intermediate_plots)}")
        print(f"Unique indicator combinations: {len(results)}")
        print(f"Processing completed with timestamped outputs!")
        
        # List tissue-specific files
        if tissue_count_files:
            print(f"\nTissue-specific count files created:")
            for tissue_file in sorted(tissue_count_files):
                print(f"  - {tissue_file.name}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 