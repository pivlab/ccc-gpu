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
import warnings

# Suppress specific overflow warnings during controlled operations
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in cast')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in reduce')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')

# Initialize logger (will be configured later)
logger = None

def setup_logging(log_file: str = None, output_dir: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file. If None, uses default timestamped name.
        output_dir: Output directory for log file. If None, uses current directory.
    """
    global logger
    
    if log_file is None:
        # Generate timestamped log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_dir:
            log_file = Path(output_dir) / f'gene_pair_selector_{timestamp}.log'
        else:
            log_file = Path(f'gene_pair_selector_{timestamp}.log')
    
    # Ensure parent directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Initialize the global logger
    logger = logging.getLogger(__name__)

def clean_numeric_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric data by handling inf and NaN values.
    
    Args:
        df: Input DataFrame
        columns: List of numeric columns to clean
        
    Returns:
        DataFrame with cleaned numeric data
    """
    logger.info("Cleaning numeric data...")
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in data, skipping...")
            continue
            
        # Convert to float64 for better precision
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast=None).astype('float64')
        
        # Count problematic values before cleaning
        inf_count = np.isinf(df[col]).sum()
        nan_count = df[col].isna().sum()
        
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in column {col}")
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in column {col}")
        
        # Replace inf and -inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Final count of NaN values
        final_nan_count = df[col].isna().sum()
        logger.info(f"Column {col}: {final_nan_count} NaN values after cleaning")
    
    return df

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
        
        # Clean numeric data immediately after loading
        numeric_cols = ['ccc', 'pearson', 'spearman']
        df = clean_numeric_data(df, numeric_cols)
        
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
        else:
            # Check for data quality issues
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} NaN values ({nan_count/len(df)*100:.1f}%)")
    
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

def get_combination_name(combination_tuple: Tuple[bool, bool, bool, bool, bool, bool]) -> str:
    """
    Convert combination tuple to descriptive name.
    
    Args:
        combination_tuple: Tuple in format [Spearman (low), Pearson (low), Clustermatch (low), 
                          Spearman (high), Pearson (high), Clustermatch (high)]
        
    Returns:
        Descriptive name like "c-high-p-low-s-low"
    """
    spearman_low, pearson_low, clustermatch_low, spearman_high, pearson_high, clustermatch_high = combination_tuple
    
    parts = []
    
    # Clustermatch
    if clustermatch_high and not clustermatch_low:
        parts.append("c-high")
    elif clustermatch_low and not clustermatch_high:
        parts.append("c-low")
    elif clustermatch_high and clustermatch_low:
        parts.append("c-both")
    else:
        parts.append("c-none")
    
    # Pearson
    if pearson_high and not pearson_low:
        parts.append("p-high")
    elif pearson_low and not pearson_high:
        parts.append("p-low")
    elif pearson_high and pearson_low:
        parts.append("p-both")
    else:
        parts.append("p-none")
    
    # Spearman
    if spearman_high and not spearman_low:
        parts.append("s-high")
    elif spearman_low and not spearman_high:
        parts.append("s-low")
    elif spearman_high and spearman_low:
        parts.append("s-both")
    else:
        parts.append("s-none")
    
    return "-".join(parts)

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
    
    # Ensure all columns are float64 for precision
    for col in ['ccc', 'pearson', 'spearman']:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Remove rows where any of the required columns have NaN values
    original_length = len(df)
    df = df.dropna(subset=['ccc', 'pearson', 'spearman'])
    dropped_rows = original_length - len(df)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with NaN values in required columns")
    
    if len(df) == 0:
        logger.error("No valid data remaining after removing NaN values")
        raise ValueError("No valid data remaining after cleaning")
    
    # Calculate absolute differences using numpy for better control
    logger.info("Calculating absolute differences...")
    
    # Use numpy.abs for more robust calculation
    df['ccc_pearson_diff'] = np.abs(df['ccc'].values - df['pearson'].values, dtype=np.float64)
    df['ccc_spearman_diff'] = np.abs(df['ccc'].values - df['spearman'].values, dtype=np.float64)
    
    # Calculate combined distance (sum of both differences)
    logger.info("Calculating combined distance...")
    df['ccc_combined_distance'] = df['ccc_pearson_diff'] + df['ccc_spearman_diff']
    
    # Calculate mean distance
    logger.info("Calculating mean distance...")
    df['ccc_mean_distance'] = (df['ccc_pearson_diff'] + df['ccc_spearman_diff']) / 2.0
    
    # Calculate max distance
    logger.info("Calculating max distance...")
    df['ccc_max_distance'] = np.maximum(df['ccc_pearson_diff'], df['ccc_spearman_diff'])
    
    # Verify no infinite values were created
    distance_cols = ['ccc_pearson_diff', 'ccc_spearman_diff', 'ccc_combined_distance', 'ccc_mean_distance', 'ccc_max_distance']
    for col in distance_cols:
        if col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in {col} after calculation")
                # Replace with NaN and then handle
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Remove any rows that have infinite distance values
    before_inf_filter = len(df)
    df = df.dropna(subset=distance_cols)
    after_inf_filter = len(df)
    
    if before_inf_filter > after_inf_filter:
        logger.warning(f"Removed {before_inf_filter - after_inf_filter} rows with infinite distance values")
    
    logger.info(f"Distance calculation completed for {len(df)} gene pairs")
    
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
    
    if distance_col not in df.columns:
        logger.error(f"Distance column {distance_col} not found in data")
        raise ValueError(f"Distance column {distance_col} not found")
    
    # Check for NaN values in the sort column
    nan_count = df[distance_col].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in {distance_col}, these will be sorted to the end")
    
    # Sort by distance (descending - largest distance first)
    # NaN values will be placed at the end
    sorted_df = df.sort_values(distance_col, ascending=False, na_position='last')
    
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
            # Use robust statistics that handle NaN values
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                logger.info(f"{col.upper()} statistics (n={len(valid_data)}):")
                logger.info(f"  Mean: {valid_data.mean():.6f}")
                logger.info(f"  Std:  {valid_data.std():.6f}")
                logger.info(f"  Min:  {valid_data.min():.6f}")
                logger.info(f"  Max:  {valid_data.max():.6f}")
                logger.info(f"  Median: {valid_data.median():.6f}")
            else:
                logger.info(f"{col.upper()}: No valid data")
    
    # Distance statistics (if available)
    distance_cols = ['ccc_pearson_diff', 'ccc_spearman_diff', 'ccc_combined_distance', 'ccc_mean_distance', 'ccc_max_distance']
    for col in distance_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                logger.info(f"{col} statistics (n={len(valid_data)}):")
                logger.info(f"  Mean: {valid_data.mean():.6f}")
                logger.info(f"  Std:  {valid_data.std():.6f}")
                logger.info(f"  Min:  {valid_data.min():.6f}")
                logger.info(f"  Max:  {valid_data.max():.6f}")
                logger.info(f"  Median: {valid_data.median():.6f}")
            else:
                logger.info(f"{col}: No valid data")

def display_predefined_combinations():
    """
    Display the predefined combinations for reference.
    """
    combinations = get_predefined_combinations()
    
    logger.info("\nPredefined combinations available:")
    logger.info("=" * 130)
    logger.info(f"{'Index':<5} {'Name':<25} {'Spearman(L)':<11} {'Pearson(L)':<10} {'Clustermatch(L)':<15} {'Spearman(H)':<11} {'Pearson(H)':<10} {'Clustermatch(H)':<15}")
    logger.info("-" * 130)
    
    for idx, combo in enumerate(combinations):
        spearman_low, pearson_low, clustermatch_low, spearman_high, pearson_high, clustermatch_high = combo
        combo_name = get_combination_name(combo)
        logger.info(f"{idx:<5} {combo_name:<25} {str(spearman_low):<11} {str(pearson_low):<10} {str(clustermatch_low):<15} {str(spearman_high):<11} {str(pearson_high):<10} {str(clustermatch_high):<15}")

def get_available_tissues() -> List[str]:
    """
    Get list of available tissues for selection.
    
    Returns:
        List of tissue names
    """
    tissues = [
        "adipose_subcutaneous",
        "adipose_visceral_omentum",
        "adrenal_gland",
        "artery_aorta",
        "artery_coronary",
        "artery_tibial",
        "bladder",
        "brain_amygdala",
        "brain_anterior_cingulate_cortex_ba24",
        "brain_caudate_basal_ganglia",
        "brain_cerebellar_hemisphere",
        "brain_cerebellum",
        "brain_cortex",
        "brain_frontal_cortex_ba9",
        "brain_hippocampus",
        "brain_hypothalamus",
        "brain_nucleus_accumbens_basal_ganglia",
        "brain_putamen_basal_ganglia",
        "brain_spinal_cord_cervical_c1",
        "brain_substantia_nigra",
        "breast_mammary_tissue",
        "cells_cultured_fibroblasts",
        "cells_ebvtransformed_lymphocytes",
        "cervix_ectocervix",
        "cervix_endocervix",
        "colon_sigmoid",
        "colon_transverse",
        "esophagus_gastroesophageal_junction",
        "esophagus_mucosa",
        "esophagus_muscularis",
        "fallopian_tube",
        "heart_atrial_appendage",
        "heart_left_ventricle",
        "kidney_cortex",
        "kidney_medulla",
        "liver",
        "lung",
        "minor_salivary_gland",
        "muscle_skeletal",
        "nerve_tibial",
        "ovary",
        "pancreas",
        "pituitary",
        "prostate",
        "skin_not_sun_exposed_suprapubic",
        "skin_sun_exposed_lower_leg",
        "small_intestine_terminal_ileum",
        "spleen",
        "stomach",
        "testis",
        "thyroid",
        "uterus",
        "vagina",
        "whole_blood"
    ]
    return tissues

def find_intersection_file(data_dir: str, tissue: str) -> str:
    """
    Find the intersection file for a specific tissue.
    
    Args:
        data_dir: Directory containing intersection files
        tissue: Tissue name
        
    Returns:
        Path to the intersection file
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Look for files matching the pattern: gene_pair_intersections*{tissue}*.pkl
    pattern = f"gene_pair_intersections*{tissue}*.pkl"
    matching_files = list(data_path.glob(pattern))
    
    if not matching_files:
        # Try alternative pattern without asterisks
        pattern = f"*{tissue}*.pkl"
        matching_files = list(data_path.glob(pattern))
    
    if not matching_files:
        # List available files for debugging
        all_files = list(data_path.glob("*.pkl"))
        logger.error(f"No intersection file found for tissue: {tissue}")
        logger.error(f"Available files in {data_dir}:")
        for file in all_files:
            logger.error(f"  {file.name}")
        raise FileNotFoundError(f"No intersection file found for tissue: {tissue}")
    
    if len(matching_files) > 1:
        logger.warning(f"Multiple files found for tissue {tissue}:")
        for file in matching_files:
            logger.warning(f"  {file}")
        logger.warning(f"Using first match: {matching_files[0]}")
    
    selected_file = str(matching_files[0])
    logger.info(f"Found intersection file for {tissue}: {selected_file}")
    
    return selected_file

def main():
    """
    Main function to run the gene pair selector.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Select and filter gene pairs based on boolean indicator combinations'
    )
    parser.add_argument(
        '--data-dir',
        help='Directory containing gene pair intersection files'
    )
    parser.add_argument(
        '--tissue',
        choices=get_available_tissues(),
        help='Tissue to analyze'
    )
    parser.add_argument(
        '--output',
        help='Output directory to save results (combination-specific subdirectory will be created)'
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
    parser.add_argument(
        '--list-tissues',
        action='store_true',
        help='List all available tissues and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging - we'll update this later with the output directory
    global logger
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # If user wants to list tissues, show them and exit
    if args.list_tissues:
        logger.info("Available tissues:")
        for tissue in get_available_tissues():
            logger.info(f"  {tissue}")
        return 0
    
    # If user wants to list combinations, show them and exit
    if args.list_combinations:
        display_predefined_combinations()
        return 0
    
    # Check if required arguments are provided when not listing
    if not args.data_dir or not args.tissue or not args.output:
        logger.error("--data-dir, --tissue, and --output are required when not using --list-combinations or --list-tissues")
        parser.print_help()
        return 1
    
    # Check if combination-index is provided when not listing
    if args.combination_index is None:
        logger.error("--combination-index is required when not using --list-combinations or --list-tissues")
        parser.print_help()
        return 1
    
    logger.info("Starting Gene Pair Selector (Non-Interactive Mode)")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Tissue: {args.tissue}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Combination index: {args.combination_index}")
    logger.info(f"Sort by: {args.sort_by}")
    
    try:
        # Find the intersection file for the specified tissue
        input_file = find_intersection_file(args.data_dir, args.tissue)
        
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
        combination_name = get_combination_name(chosen_combination_tuple)
        
        # Setup logging with proper output directory
        output_dir = Path(args.output) / combination_name
        if args.log_file:
            setup_logging(args.log_file)
        else:
            setup_logging(output_dir=str(output_dir))
        logger = logging.getLogger(__name__)
        
        logger.info(f"Using predefined combination {args.combination_index}: {combination_name}")
        logger.info(f"  [Spearman(L), Pearson(L), Clustermatch(L), Spearman(H), Pearson(H), Clustermatch(H)]")
        logger.info(f"  {chosen_combination_tuple}")
        
        # Load data
        df = load_data(input_file)
        
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
            'data_dir': args.data_dir,
            'tissue': args.tissue,
            'input_file': input_file,
            'output_dir': str(output_dir),
            'sort_by': args.sort_by,
            'chosen_combination_index': args.combination_index,
            'chosen_combination_name': combination_name,
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