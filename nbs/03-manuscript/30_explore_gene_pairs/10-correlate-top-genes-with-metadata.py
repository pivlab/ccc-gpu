#!/usr/bin/env python3
"""
CLI tool for combining top gene pairs across tissues and combination categories.
Processes results from report_top_gene_pairs.py across all tissues and combinations.
"""

import argparse
import sys
import logging
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd


def setup_logging(output_dir):
    """Set up logging configuration."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs" / f"gene_pair_combination_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "gene_pair_combination.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger, log_file


def discover_tissues_and_combinations(data_dir):
    """Discover all available tissues and their combinations."""
    logger = logging.getLogger(__name__)
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info(f"Scanning data directory: {data_dir}")
    
    tissues = []
    combinations = set()
    
    # Scan for tissue directories
    for tissue_dir in data_dir.iterdir():
        if tissue_dir.is_dir():
            tissues.append(tissue_dir.name)
            
            # Scan for combination directories within this tissue
            for combo_dir in tissue_dir.iterdir():
                if combo_dir.is_dir():
                    combinations.add(combo_dir.name)
    
    tissues.sort()
    combinations = sorted(list(combinations))
    
    logger.info(f"Discovered {len(tissues)} tissues")
    logger.info(f"Discovered {len(combinations)} combinations: {combinations}")
    
    return tissues, combinations


def find_top_gene_files(data_dir, top_n):
    """Find all top gene pairs files matching the pattern."""
    logger = logging.getLogger(__name__)
    data_dir = Path(data_dir)
    
    file_pattern = f"top_{top_n}_gene_pairs.csv"
    logger.info(f"Looking for files matching pattern: {file_pattern}")
    
    # Structure: {combination: {tissue: file_path}}
    found_files = defaultdict(dict)
    missing_files = []
    
    tissues, combinations = discover_tissues_and_combinations(data_dir)
    
    for tissue in tissues:
        for combination in combinations:
            expected_path = data_dir / tissue / combination / file_pattern
            
            if expected_path.exists():
                found_files[combination][tissue] = expected_path
            else:
                missing_files.append(str(expected_path))
    
    # Log statistics
    total_expected = len(tissues) * len(combinations)
    total_found = sum(len(combo_files) for combo_files in found_files.values())
    
    logger.info(f"File discovery summary:")
    logger.info(f"  Expected files: {total_expected}")
    logger.info(f"  Found files: {total_found}")
    logger.info(f"  Missing files: {len(missing_files)}")
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} files:")
        for missing_file in missing_files[:10]:  # Show first 10
            logger.warning(f"  {missing_file}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")
    
    # Log found files by combination
    for combination in sorted(found_files.keys()):
        tissues_with_files = len(found_files[combination])
        logger.info(f"Combination '{combination}': {tissues_with_files} tissues")
    
    return dict(found_files), tissues, combinations


def load_and_combine_combination(combination, tissue_files, output_dir):
    """Load and combine all tissue files for a specific combination."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing combination: {combination}")
    logger.info(f"  Files to process: {len(tissue_files)}")
    
    combined_dfs = []
    load_errors = []
    total_rows = 0
    
    for tissue, file_path in tissue_files.items():
        try:
            logger.info(f"  Loading {tissue}: {file_path.name}")
            df = pd.read_csv(file_path)
            
            # Add tissue column
            df['tissue'] = tissue
            
            # Reorder columns to put tissue first
            cols = ['tissue'] + [col for col in df.columns if col != 'tissue']
            df = df[cols]
            
            combined_dfs.append(df)
            total_rows += len(df)
            
            logger.info(f"    Loaded {len(df)} rows")
            
        except Exception as e:
            load_errors.append(f"{tissue}: {str(e)}")
            logger.error(f"  Failed to load {tissue}: {e}")
    
    if not combined_dfs:
        logger.error(f"No files successfully loaded for combination: {combination}")
        return None
    
    # Combine all dataframes
    logger.info(f"Combining {len(combined_dfs)} dataframes...")
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    logger.info(f"Total rows: {total_rows}")
    
    if load_errors:
        logger.warning(f"Load errors for combination {combination}:")
        for error in load_errors:
            logger.warning(f"  {error}")
    
    # Save combined results
    output_file_pkl = output_dir / f"combined_{combination}_top_gene_pairs.pkl"
    output_file_csv = output_dir / f"combined_{combination}_top_gene_pairs.csv"
    
    combined_df.to_pickle(output_file_pkl)
    combined_df.to_csv(output_file_csv, index=False)
    
    logger.info(f"Saved combined results:")
    logger.info(f"  Pickle: {output_file_pkl}")
    logger.info(f"  CSV: {output_file_csv}")
    
    return combined_df, len(load_errors)


def generate_summary_report(results_summary, output_dir, top_n):
    """Generate a comprehensive summary report."""
    logger = logging.getLogger(__name__)
    
    summary_file = output_dir / "combination_summary_report.txt"
    
    with open(summary_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("TOP GENE PAIRS COMBINATION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Top N Gene Pairs: {top_n}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write("\n")
        
        # Overall statistics
        total_combinations = len(results_summary)
        total_tissues_processed = sum(stats['tissues_processed'] for stats in results_summary.values())
        total_rows = sum(stats['total_rows'] for stats in results_summary.values())
        total_errors = sum(stats['load_errors'] for stats in results_summary.values())
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Combinations processed: {total_combinations}\n")
        f.write(f"Total tissues processed: {total_tissues_processed}\n")
        f.write(f"Total gene pairs combined: {total_rows:,}\n")
        f.write(f"Total load errors: {total_errors}\n")
        f.write("\n")
        
        # By combination
        f.write("BY COMBINATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Combination':<25} {'Tissues':<10} {'Rows':<12} {'Errors':<8} {'Success Rate':<12}\n")
        f.write("-" * 70 + "\n")
        
        for combination in sorted(results_summary.keys()):
            stats = results_summary[combination]
            success_rate = ((stats['tissues_processed'] - stats['load_errors']) / stats['tissues_processed'] * 100) if stats['tissues_processed'] > 0 else 0
            
            f.write(f"{combination:<25} {stats['tissues_processed']:<10} {stats['total_rows']:<12,} {stats['load_errors']:<8} {success_rate:<12.1f}%\n")
        
        f.write("\n")
        
        # Files created
        f.write("OUTPUT FILES CREATED\n")
        f.write("-" * 40 + "\n")
        for combination in sorted(results_summary.keys()):
            f.write(f"Combination: {combination}\n")
            f.write(f"  - combined_{combination}_top_gene_pairs.pkl\n")
            f.write(f"  - combined_{combination}_top_gene_pairs.csv\n")
        
        f.write(f"\nSummary report: combination_summary_report.txt\n")
        f.write(f"Log file: Available in logs/ subdirectory\n")
    
    logger.info(f"Summary report saved to: {summary_file}")


def main():
    """Main function to process top gene pairs across tissues and combinations."""
    parser = argparse.ArgumentParser(
        description="Combine top gene pairs across tissues and combination categories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--top",
        type=int,
        help="Number of top gene pairs to match (e.g., 1000 for top_1000_gene_pairs.csv)",
    )
    
    parser.add_argument(
        "--data-dir",
        default="/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_selection",
        help="Directory containing gene pair selection results",
    )
    
    parser.add_argument(
        "--output-dir",
        default="/pividori_lab/haoyu_projects/ccc-gpu/results/top_gene_pair_correlation",
        help="Directory to save combined results",
    )
    
    parser.add_argument(
        "--list-combinations",
        action="store_true",
        help="List available combination categories and exit",
    )
    
    parser.add_argument(
        "--list-tissues",
        action="store_true",
        help="List available tissues and exit",
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logger, log_file = setup_logging(output_dir)
        
        logger.info(f"Starting top gene pairs combination analysis")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Top N gene pairs: {args.top}")
        
        # Discover tissues and combinations
        tissues, combinations = discover_tissues_and_combinations(args.data_dir)
        
        # If user wants to list combinations
        if args.list_combinations:
            logger.info("Available combination categories:")
            for combo in combinations:
                logger.info(f"  {combo}")
            return
        
        # If user wants to list tissues
        if args.list_tissues:
            logger.info("Available tissues:")
            for tissue in tissues:
                logger.info(f"  {tissue}")
            return
        
        # Validate required arguments for main processing
        if args.top is None:
            logger.error("--top argument is required for processing (not needed for --list-* commands)")
            sys.exit(1)
        
        # Find all top gene files
        found_files, tissues, combinations = find_top_gene_files(args.data_dir, args.top)
        
        if not found_files:
            logger.error("No matching top gene pairs files found")
            sys.exit(1)
        
        # Process each combination
        results_summary = {}
        total_start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING COMBINATIONS")
        logger.info(f"{'='*60}")
        
        for i, combination in enumerate(sorted(combinations), 1):
            logger.info(f"\n[{i}/{len(combinations)}] Processing combination: {combination}")
            
            if combination not in found_files:
                logger.warning(f"No files found for combination: {combination}")
                continue
            
            tissue_files = found_files[combination]
            
            combo_start_time = time.time()
            combined_df, load_errors = load_and_combine_combination(
                combination, tissue_files, output_dir
            )
            combo_end_time = time.time()
            combo_runtime = combo_end_time - combo_start_time
            
            if combined_df is not None:
                # Store summary statistics
                results_summary[combination] = {
                    'tissues_processed': len(tissue_files),
                    'total_rows': len(combined_df),
                    'load_errors': load_errors,
                    'runtime': combo_runtime,
                    'unique_genes': len(set(combined_df['gene1'].tolist() + combined_df['gene2'].tolist())) if 'gene1' in combined_df.columns and 'gene2' in combined_df.columns else 0
                }
                
                logger.info(f"Combination {combination} completed:")
                logger.info(f"  Runtime: {combo_runtime:.2f} seconds ({combo_runtime/60:.2f} minutes)")
                logger.info(f"  Total rows: {len(combined_df):,}")
                logger.info(f"  Unique genes: {results_summary[combination]['unique_genes']:,}")
            else:
                logger.error(f"Failed to process combination: {combination}")
        
        total_end_time = time.time()
        total_runtime = total_end_time - total_start_time
        
        # Generate summary report
        generate_summary_report(results_summary, output_dir, args.top)
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        logger.info(f"Combinations processed: {len(results_summary)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Log file: {log_file}")
        
        # Show which files were created
        logger.info("\nOutput files created:")
        for combination in sorted(results_summary.keys()):
            logger.info(f"  combined_{combination}_top_gene_pairs.pkl")
            logger.info(f"  combined_{combination}_top_gene_pairs.csv")
        logger.info(f"  combination_summary_report.txt")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 