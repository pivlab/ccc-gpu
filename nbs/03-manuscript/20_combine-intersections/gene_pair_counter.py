#!/usr/bin/env python3
"""
Gene Pair Counter CLI Script

This script processes a dataset with gene pairs (multi-index) and counts
how many gene pairs exist for each unique combination of boolean indicators.

The dataset should have columns:
- Pearson (high), Pearson (low), Spearman (high), Spearman (low), 
  Clustermatch (high), Clustermatch (low) - boolean columns
- ccc, pearson, spearman - numeric columns

Usage:
    python gene_pair_counter.py input_file.pkl output_file.pkl

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
    
    print("\n" + "="*80)
    print("GENE PAIR COUNTING RESULTS")
    print("="*80)
    
    for i, (_, row) in enumerate(results_df.head(top_n).iterrows()):
        print(f"\nRank {i+1}:")
        print(f"  Gene pairs: {row['gene_pair_count']}")
        print(f"  Indicators:")
        print(f"    Pearson (high): {row['Pearson (high)']}")
        print(f"    Pearson (low): {row['Pearson (low)']}")
        print(f"    Spearman (high): {row['Spearman (high)']}")
        print(f"    Spearman (low): {row['Spearman (low)']}")
        print(f"    Clustermatch (high): {row['Clustermatch (high)']}")
        print(f"    Clustermatch (low): {row['Clustermatch (low)']}")
        print("-" * 50)

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
        description="Count gene pairs by indicator combinations with parallel processing support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python gene_pair_counter.py input_data.pkl output_counts.pkl
    python gene_pair_counter.py data.pkl results.pkl --top-n 5
    python gene_pair_counter.py input.pkl output.pkl --log-file analysis.log
    python gene_pair_counter.py input.pkl output.pkl --plot
    python gene_pair_counter.py input.pkl output.pkl --threads 8
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input pickle file containing gene pair data'
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

def main():
    """
    Main function to orchestrate the gene pair counting process.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create timestamped folder for outputs
    output_folder = create_timestamped_folder()
    
    # Setup output paths in the timestamped folder
    input_filename = Path(args.input_file).name
    output_filename = Path(args.output_file).name
    
    # Update output paths to use timestamped folder
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
    
    logger.info("Starting Gene Pair Counter CLI Script")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Output file: {output_file_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Generate plot: {args.plot}")
    logger.info(f"Threads: {args.threads if args.threads else 'auto (CPU count)'}")
    
    try:
        # Validate input file exists
        if not Path(args.input_file).exists():
            logger.error(f"Input file does not exist: {args.input_file}")
            sys.exit(1)
        
        # Load data
        df = load_data(args.input_file)
        
        # Validate data structure
        if not validate_data(df):
            logger.error("Data validation failed. Exiting.")
            sys.exit(1)
        
        # Display basic info about the dataset
        logger.info(f"Dataset contains {len(df)} gene pairs")
        logger.info(f"Unique gene pairs: {df.index.nunique()}")
        
        # Count gene pairs by indicator combinations
        results = count_gene_pairs_by_indicators(df, n_threads=args.threads)
        
        # Display results (unless --no-display is specified)
        if not args.no_display:
            display_results(results, top_n=args.top_n)
        
        # Save results
        save_results(results, str(output_file_path))
        
        # Generate plot if requested
        plot_path = None
        if args.plot:
            plot_path = create_bar_plot(results, str(output_file_path))
        
        logger.info("Gene pair counting completed successfully!")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Input file: {args.input_file}")
        print(f"Output folder: {output_folder}")
        print(f"Output file: {output_file_path}")
        print(f"Log file: {log_file}")
        if plot_path:
            print(f"Plot file: {plot_path}")
        print(f"Total gene pairs processed: {len(df)}")
        print(f"Unique indicator combinations: {len(results)}")
        print(f"Processing method: {'Parallel' if len(df) >= 50000 else 'Single-threaded'}")
        if len(df) >= 50000:
            print(f"Threading strategy: {'Threads' if len(df) < 500000 else 'Processes'}")
        print(f"Results saved successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 