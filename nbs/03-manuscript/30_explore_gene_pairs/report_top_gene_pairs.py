#!/usr/bin/env python3
"""
Display Top Gene Pairs CLI Script

This script loads processed gene pair data from the gene_pair_selector output
and displays the top data lines with metadata information.

Usage:
    python report_top_gene_pairs.py --input /path/to/sorted_data_cache.pkl --output results.txt
    python report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --top 100
    python report_top_gene_pairs.py --data-dir /path/to/results --tissue liver --combination c-low-p-high-s-low

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

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the gene pair dataset from a pickle file.
    
    Args:
        file_path: Path to the input pickle file
        
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
        logger.error(f"Error loading data: {e}")
        raise

def load_metadata(metadata_path: str) -> dict:
    """
    Load metadata from pickle file if it exists.
    
    Args:
        metadata_path: Path to the metadata pickle file
        
    Returns:
        Metadata dictionary or empty dict if file doesn't exist
    """
    try:
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
            return metadata
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
        return {}

def find_data_file(data_dir: str, tissue: str, combination: str) -> str:
    """
    Find the sorted data cache file based on tissue and combination.
    
    Args:
        data_dir: Base data directory
        tissue: Tissue name
        combination: Combination name
        
    Returns:
        Path to the sorted data cache file
    """
    # Look for the combination directory
    base_path = Path(data_dir)
    
    # Try different directory patterns
    possible_patterns = [
        f"{tissue}_{combination}",
        f"{tissue}_combination_*"
    ]
    
    # Search for matching directories
    for pattern in possible_patterns:
        if '*' in pattern:
            # Handle wildcard pattern - look for tissue_combination_* directories
            for dir_path in base_path.glob(pattern):
                candidate = dir_path / combination / "sorted_data_cache.pkl"
                if candidate.exists():
                    return str(candidate)
        else:
            # Direct pattern
            candidate = base_path / pattern / "sorted_data_cache.pkl"
            if candidate.exists():
                return str(candidate)
    
    # Also try direct combination directory
    candidate = base_path / combination / "sorted_data_cache.pkl"
    if candidate.exists():
        return str(candidate)
    
    # List available directories for debugging
    available_dirs = [d.name for d in base_path.iterdir() if d.is_dir()]
    raise FileNotFoundError(f"Could not find sorted data cache for tissue '{tissue}' and combination '{combination}' in {data_dir}. Available directories: {available_dirs}")

def format_gene_pair_output(df: pd.DataFrame, top_n: int = 30) -> str:
    """
    Format the top gene pairs for text output.
    
    Args:
        df: DataFrame with gene pair data
        top_n: Number of top rows to display
        
    Returns:
        Formatted string representation
    """
    top_df = df.head(top_n)
    
    output_lines = []
    output_lines.append(f"Top {top_n} Gene Pairs (from original sorted data)")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    # Add column headers
    if hasattr(df.index, 'names') and df.index.names == ['gene1', 'gene2']:
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
        if hasattr(df.index, 'names') and df.index.names == ['gene1', 'gene2']:
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

def export_to_csv(df: pd.DataFrame, top_n: int, output_path: str) -> None:
    """
    Export the top gene pairs to a CSV file.
    
    Args:
        df: DataFrame with gene pair data
        top_n: Number of top rows to export
        output_path: Path to save the CSV file
    """
    top_df = df.head(top_n)
    
    # Reset index to make gene pairs into columns if they're multi-index
    if hasattr(df.index, 'names') and df.index.names == ['gene1', 'gene2']:
        top_df = top_df.reset_index()
    else:
        top_df = top_df.reset_index()
        top_df.rename(columns={'index': 'gene_pair_index'}, inplace=True)
    
    # Save to CSV
    top_df.to_csv(output_path, index=False)
    logger.info(f"CSV exported to: {output_path}")

def create_output_report(df: pd.DataFrame, metadata: dict, args: argparse.Namespace) -> str:
    """
    Create a complete output report with metadata and bottom gene pairs.
    
    Args:
        df: DataFrame with gene pair data
        metadata: Metadata dictionary
        args: Command line arguments
        
    Returns:
        Complete formatted report
    """
    report_lines = []
    
    # Header
    report_lines.append("Gene Pair Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Metadata section
    report_lines.append("METADATA")
    report_lines.append("-" * 20)
    
    if args.tissue:
        report_lines.append(f"Tissue: {args.tissue}")
    if args.combination:
        report_lines.append(f"Combination: {args.combination}")
    if args.input:
        report_lines.append(f"Input File: {args.input}")
    
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
    top_output = format_gene_pair_output(df, args.top)
    report_lines.append(top_output)
    
    return "\n".join(report_lines)

def main():
    global logger
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Display top gene pairs from processed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct file input
  python report_top_gene_pairs.py --input /path/to/sorted_data_cache.pkl --output results.txt

  # Using tissue and combination
  python report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --top 100

  # With custom CSV output
  python report_top_gene_pairs.py --tissue whole_blood --combination c-high-p-low-s-low --csv my_results.csv

  # With custom data directory
  python report_top_gene_pairs.py --data-dir /path/to/results --tissue liver --combination c-low-p-high-s-low
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', 
                           help='Direct path to sorted_data_cache.pkl file')
    input_group.add_argument('--tissue', 
                           help='Tissue name (requires --combination)')
    
    parser.add_argument('--combination', 
                        help='Combination name (e.g., c-high-p-low-s-low)')
    
    parser.add_argument('--data-dir', 
                        default='/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection',
                        help='Base directory for gene pair selection results (default: %(default)s)')
    
    parser.add_argument('--output', 
                        help='Output text file path (default: auto-generated based on tissue/combination)')
    
    parser.add_argument('--csv', 
                        help='Output CSV file path (default: auto-generated based on tissue/combination)')
    
    parser.add_argument('--top', 
                        type=int, 
                        default=30,
                        help='Number of top gene pairs to display (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.tissue and not args.combination:
        parser.error("--combination is required when using --tissue")
    
    try:
        # Determine input file path
        if args.input:
            input_file = args.input
            metadata_file = str(Path(args.input).parent / 'selection_metadata.pkl')
        else:
            input_file = find_data_file(args.data_dir, args.tissue, args.combination)
            metadata_file = str(Path(input_file).parent / 'selection_metadata.pkl')
        
        # Load data
        df = load_data(input_file)
        metadata = load_metadata(metadata_file)
        
        # Generate output file names if not provided
        if not args.output:
            if args.tissue and args.combination:
                output_name = f"{args.tissue}_{args.combination}_top_{args.top}_gene_pairs.txt"
            else:
                input_path = Path(args.input)
                output_name = f"{input_path.stem}_top_{args.top}_gene_pairs.txt"
            args.output = output_name
        
        if not args.csv:
            if args.tissue and args.combination:
                csv_name = f"{args.tissue}_{args.combination}_top_{args.top}_gene_pairs.csv"
            else:
                input_path = Path(args.input)
                csv_name = f"{input_path.stem}_top_{args.top}_gene_pairs.csv"
            args.csv = csv_name
        
        # Create report
        report = create_output_report(df, metadata, args)
        
        # Save to file
        with open(args.output, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to: {args.output}")
        
        # Export to CSV
        export_to_csv(df, args.top, args.csv)
        
        # Also display summary to console
        print("\nTop Gene Pairs Summary:")
        print(f"Input file: {input_file}")
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