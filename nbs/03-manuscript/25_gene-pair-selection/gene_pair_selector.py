#!/usr/bin/env python3
"""
Gene Pair Selector CLI Script

This script processes a gene intersection file and allows users to select
a specific combination of boolean indicators, then filters and sorts the data
based on CCC distance to Pearson and Spearman values.

The dataset should have columns:
- Pearson (high), Pearson (low), Spearman (high), Spearman (low), 
  Clustermatch (high), Clustermatch (low) - boolean columns
- ccc, pearson, spearman - numeric columns

Usage:
    python gene_pair_selector.py input_file.pkl output_dir

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
from datetime import datetime

# Initialize logger (will be configured later)
logger = None

def setup_logging(log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file. If None, uses default name.
    """
    if log_file is None:
        log_file = 'gene_pair_selector.log'
    
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

def get_available_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get all available combinations of boolean indicators and their counts.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with combinations and their counts
    """
    logger.info("Getting available combinations...")
    
    # Define the boolean indicator columns
    indicator_cols = [
        'Pearson (high)', 'Pearson (low)', 
        'Spearman (high)', 'Spearman (low)',
        'Clustermatch (high)', 'Clustermatch (low)'
    ]
    
    # Group by the indicator columns and count
    combinations = df.groupby(indicator_cols).size().reset_index(name='gene_pair_count')
    
    # Sort by count in descending order
    combinations = combinations.sort_values('gene_pair_count', ascending=False)
    
    logger.info(f"Found {len(combinations)} unique combinations")
    
    return combinations

def display_combinations(combinations: pd.DataFrame):
    """
    Display available combinations to the user.
    
    Args:
        combinations: DataFrame with combinations and counts
    """
    logger.info("\nAvailable combinations:")
    logger.info("=" * 80)
    
    # Column headers
    logger.info(f"{'Index':<5} {'Count':<10} {'Pearson(H)':<10} {'Pearson(L)':<10} {'Spearman(H)':<11} {'Spearman(L)':<11} {'Clustermatch(H)':<15} {'Clustermatch(L)':<15}")
    logger.info("-" * 80)
    
    for idx, row in combinations.iterrows():
        logger.info(f"{idx:<5} {row['gene_pair_count']:<10} {str(row['Pearson (high)']):<10} {str(row['Pearson (low)']):<10} {str(row['Spearman (high)']):<11} {str(row['Spearman (low)']):<11} {str(row['Clustermatch (high)']):<15} {str(row['Clustermatch (low)']):<15}")

def get_predefined_combinations() -> List[Tuple[bool, bool, bool, bool, bool, bool]]:
    """
    Get predefined combinations for batch processing.
    
    Returns:
        List of tuples representing combinations in the order:
        [Spearman (low), Pearson (low), Clustermatch (low), Spearman (high), Pearson (high), Clustermatch (high)]
    """
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
    
    return ordered_combinations

def convert_combination_format(combination_tuple: Tuple[bool, bool, bool, bool, bool, bool]) -> Dict[str, bool]:
    """
    Convert combination from user format to dataframe column format.
    
    Args:
        combination_tuple: Tuple in format [Spearman (low), Pearson (low), Clustermatch (low), 
                          Spearman (high), Pearson (high), Clustermatch (high)]
        
    Returns:
        Dictionary with column names as keys
    """
    spearman_low, pearson_low, clustermatch_low, spearman_high, pearson_high, clustermatch_high = combination_tuple
    
    return {
        'Pearson (high)': pearson_high,
        'Pearson (low)': pearson_low,
        'Spearman (high)': spearman_high,
        'Spearman (low)': spearman_low,
        'Clustermatch (high)': clustermatch_high,
        'Clustermatch (low)': clustermatch_low
    }

def filter_data_by_combination(df: pd.DataFrame, combination: Dict[str, bool]) -> pd.DataFrame:
    """
    Filter data based on the chosen combination.
    
    Args:
        df: Input DataFrame
        combination: Dictionary with the chosen combination values
        
    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering data by chosen combination...")
    
    # Define the boolean indicator columns
    indicator_cols = [
        'Pearson (high)', 'Pearson (low)', 
        'Spearman (high)', 'Spearman (low)',
        'Clustermatch (high)', 'Clustermatch (low)'
    ]
    
    # Create filter mask
    mask = pd.Series(True, index=df.index)
    
    for col in indicator_cols:
        mask = mask & (df[col] == combination[col])
    
    filtered_df = df[mask].copy()
    
    logger.info(f"Filtered data from {len(df)} to {len(filtered_df)} gene pairs")
    
    # Log the chosen combination
    logger.info("Chosen combination:")
    for col in indicator_cols:
        logger.info(f"  {col}: {combination[col]}")
    
    return filtered_df

def calculate_ccc_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CCC distance to Pearson and Spearman values.
    
    Args:
        df: Input DataFrame with ccc, pearson, spearman columns
        
    Returns:
        DataFrame with additional distance columns
    """
    logger.info("Calculating CCC distance to Pearson and Spearman...")
    
    df = df.copy()
    
    # Calculate absolute differences
    df['ccc_pearson_diff'] = abs(df['ccc'] - df['pearson'])
    df['ccc_spearman_diff'] = abs(df['ccc'] - df['spearman'])
    
    # Calculate combined distance (sum of both differences)
    df['ccc_combined_distance'] = df['ccc_pearson_diff'] + df['ccc_spearman_diff']
    
    # Calculate mean distance
    df['ccc_mean_distance'] = (df['ccc_pearson_diff'] + df['ccc_spearman_diff']) / 2
    
    # Calculate max distance
    df['ccc_max_distance'] = df[['ccc_pearson_diff', 'ccc_spearman_diff']].max(axis=1)
    
    logger.info("Distance calculation completed")
    
    return df

def sort_by_distance(df: pd.DataFrame, sort_by: str = 'combined') -> pd.DataFrame:
    """
    Sort data by CCC distance.
    
    Args:
        df: Input DataFrame with distance columns
        sort_by: Which distance metric to sort by ('combined', 'mean', 'max')
        
    Returns:
        Sorted DataFrame
    """
    logger.info(f"Sorting by CCC {sort_by} distance...")
    
    distance_col_map = {
        'combined': 'ccc_combined_distance',
        'mean': 'ccc_mean_distance',
        'max': 'ccc_max_distance'
    }
    
    if sort_by not in distance_col_map:
        logger.warning(f"Unknown sort_by value: {sort_by}. Using 'combined' instead.")
        sort_by = 'combined'
    
    distance_col = distance_col_map[sort_by]
    
    # Sort by distance (ascending - smallest distance first)
    sorted_df = df.sort_values(distance_col, ascending=True)
    
    logger.info(f"Data sorted by {distance_col}")
    
    return sorted_df

def save_data(df: pd.DataFrame, file_path: str, description: str = "data"):
    """
    Save DataFrame to pickle file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        description: Description for logging
    """
    try:
        logger.info(f"Saving {description} to {file_path}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
        
        logger.info(f"Successfully saved {description} with {len(df)} rows")
        
    except Exception as e:
        logger.error(f"Error saving {description}: {e}")
        raise

def print_summary_statistics(df: pd.DataFrame, title: str):
    """
    Print summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        title: Title for the summary
    """
    logger.info(f"\n{title}")
    logger.info("=" * 60)
    logger.info(f"Total gene pairs: {len(df)}")
    
    # Statistics for numeric columns
    numeric_cols = ['ccc', 'pearson', 'spearman']
    for col in numeric_cols:
        if col in df.columns:
            logger.info(f"{col.upper()} statistics:")
            logger.info(f"  Mean: {df[col].mean():.4f}")
            logger.info(f"  Std:  {df[col].std():.4f}")
            logger.info(f"  Min:  {df[col].min():.4f}")
            logger.info(f"  Max:  {df[col].max():.4f}")
    
    # Distance statistics (if available)
    distance_cols = ['ccc_pearson_diff', 'ccc_spearman_diff', 'ccc_combined_distance']
    for col in distance_cols:
        if col in df.columns:
            logger.info(f"{col} statistics:")
            logger.info(f"  Mean: {df[col].mean():.4f}")
            logger.info(f"  Std:  {df[col].std():.4f}")
            logger.info(f"  Min:  {df[col].min():.4f}")
            logger.info(f"  Max:  {df[col].max():.4f}")

def display_predefined_combinations():
    """
    Display the predefined combinations for reference.
    """
    combinations = get_predefined_combinations()
    
    logger.info("\nPredefined combinations available:")
    logger.info("=" * 100)
    logger.info(f"{'Index':<5} {'Spearman(L)':<11} {'Pearson(L)':<10} {'Clustermatch(L)':<15} {'Spearman(H)':<11} {'Pearson(H)':<10} {'Clustermatch(H)':<15}")
    logger.info("-" * 100)
    
    for idx, combo in enumerate(combinations):
        spearman_low, pearson_low, clustermatch_low, spearman_high, pearson_high, clustermatch_high = combo
        logger.info(f"{idx:<5} {str(spearman_low):<11} {str(pearson_low):<10} {str(clustermatch_low):<15} {str(spearman_high):<11} {str(pearson_high):<10} {str(clustermatch_high):<15}")

def main():
    """
    Main function to run the gene pair selector.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Select and filter gene pairs based on boolean indicator combinations'
    )
    parser.add_argument(
        'input_file',
        help='Path to input pickle file containing gene pair intersections'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--combination-index',
        type=int,
        help='Index of the predefined combination to use (0-19)'
    )
    parser.add_argument(
        '--sort-by',
        choices=['combined', 'mean', 'max'],
        default='combined',
        help='Distance metric to sort by (default: combined)'
    )
    parser.add_argument(
        '--log-file',
        help='Path to log file (default: gene_pair_selector.log)'
    )
    parser.add_argument(
        '--list-combinations',
        action='store_true',
        help='List all available predefined combinations and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # If user wants to list combinations, show them and exit
    if args.list_combinations:
        display_predefined_combinations()
        return 0
    
    # Check if combination-index is provided when not listing combinations
    if args.combination_index is None:
        logger.error("--combination-index is required when not using --list-combinations")
        parser.print_help()
        return 1
    
    logger.info("Starting Gene Pair Selector (Non-Interactive Mode)")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Combination index: {args.combination_index}")
    logger.info(f"Sort by: {args.sort_by}")
    
    try:
        # Get predefined combinations
        predefined_combinations = get_predefined_combinations()
        
        # Validate combination index
        if not (0 <= args.combination_index < len(predefined_combinations)):
            logger.error(f"Invalid combination index: {args.combination_index}. Must be between 0 and {len(predefined_combinations)-1}")
            logger.info("Use --list-combinations to see all available combinations")
            return 1
        
        # Get chosen combination
        chosen_combination_tuple = predefined_combinations[args.combination_index]
        chosen_combination = convert_combination_format(chosen_combination_tuple)
        
        logger.info(f"Using predefined combination {args.combination_index}:")
        logger.info(f"  [Spearman(L), Pearson(L), Clustermatch(L), Spearman(H), Pearson(H), Clustermatch(H)]")
        logger.info(f"  {chosen_combination_tuple}")
        
        # Load data
        df = load_data(args.input_file)
        
        # Validate data
        if not validate_data(df):
            logger.error("Data validation failed!")
            return 1
        
        # Filter data by chosen combination
        filtered_df = filter_data_by_combination(df, chosen_combination)
        
        if len(filtered_df) == 0:
            logger.warning("No gene pairs found for the chosen combination!")
            logger.info("Available combinations in the data:")
            available_combinations = get_available_combinations(df)
            display_combinations(available_combinations)
            return 1
        
        # Save filtered data
        output_dir = Path(args.output_dir)
        filtered_cache_path = output_dir / 'filtered_data_cache.pkl'
        save_data(filtered_df, filtered_cache_path, "filtered data")
        
        # Print summary for filtered data
        print_summary_statistics(filtered_df, "Filtered Data Summary")
        
        # Calculate CCC distance
        distance_df = calculate_ccc_distance(filtered_df)
        
        # Sort by distance
        sorted_df = sort_by_distance(distance_df, args.sort_by)
        
        # Save sorted data
        sorted_cache_path = output_dir / 'sorted_data_cache.pkl'
        save_data(sorted_df, sorted_cache_path, "sorted data")
        
        # Print summary for sorted data
        print_summary_statistics(sorted_df, "Sorted Data Summary")
        
        # Save metadata about the selection
        metadata = {
            'input_file': args.input_file,
            'output_dir': str(output_dir),
            'sort_by': args.sort_by,
            'chosen_combination_index': args.combination_index,
            'chosen_combination_tuple': chosen_combination_tuple,
            'chosen_combination': chosen_combination,
            'original_data_count': len(df),
            'filtered_data_count': len(filtered_df),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_dir / 'selection_metadata.pkl'
        save_data(metadata, metadata_path, "selection metadata")
        
        logger.info("Gene pair selection completed successfully!")
        logger.info(f"Filtered data saved to: {filtered_cache_path}")
        logger.info(f"Sorted data saved to: {sorted_cache_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 