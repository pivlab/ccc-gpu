#!/usr/bin/env python3
"""
CLI tool for combining top gene pairs across tissues and combination categories.
Processes results from report_top_gene_pairs.py across all tissues and combinations.

For each gene pair, computes fresh correlations with GTEx metadata using metadata_corr_cli.py
and extracts top metadata variables correlated with each gene individually and common to both genes.
"""

import argparse
import sys
import logging
import time
import subprocess
import tempfile
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np


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





def call_metadata_correlation_cli(gene1_symbol, gene2_symbol, output_dir, temp_dir):
    """
    Call the metadata correlation CLI script for a gene pair.
    
    Args:
        gene1_symbol: First gene symbol
        gene2_symbol: Second gene symbol  
        output_dir: Output directory for results
        temp_dir: Temporary directory for intermediate files
        
    Returns:
        Tuple of (gene1_results_df, gene2_results_df) or (None, None) if failed
    """
    logger = logging.getLogger(__name__)
    
    # Path to the metadata correlation CLI script
    cli_script = Path("nbs/03-manuscript/05-metadata_correlation/metadata_corr_cli.py")
    if not cli_script.exists():
        # Try relative path from current working directory
        cli_script = Path("../05-metadata_correlation/metadata_corr_cli.py")
        if not cli_script.exists():
            # Try absolute path
            cli_script = Path("/home/zhhaoyu/_projs/ccc-gpu/nbs/03-manuscript/05-metadata_correlation/metadata_corr_cli.py")
    
    if not cli_script.exists():
        logger.error(f"Cannot find metadata correlation CLI script: {cli_script}")
        return None, None
    
    # Create temporary output directory for this gene pair
    pair_temp_dir = temp_dir / f"{gene1_symbol}_{gene2_symbol}"
    pair_temp_dir.mkdir(exist_ok=True)
    
    try:
        # Construct command to run the CLI
        cmd = [
            sys.executable, str(cli_script),
            gene1_symbol, gene2_symbol,
            "--output-dir", str(pair_temp_dir),
            "--expr-data-dir", "/pividori_lab/haoyu_projects/ccc-gpu/data/gtex/gene_selection/all",
            "--permutations", "10000",  # Reduce permutations for speed
            "--n-jobs", "4"  # Reduce parallel jobs to avoid overwhelming system
        ]
        
        # Run the CLI script
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout per gene pair
        )
        
        if result.returncode != 0:
            logger.warning(f"CLI failed for {gene1_symbol}-{gene2_symbol}: {result.stderr}")
            return None, None
        
        # Load results
        gene1_file = pair_temp_dir / f"{gene1_symbol}_all_tissues_correlation_results.pkl"
        gene2_file = pair_temp_dir / f"{gene2_symbol}_all_tissues_correlation_results.pkl"
        
        gene1_df = None
        gene2_df = None
        
        if gene1_file.exists():
            gene1_df = pd.read_pickle(gene1_file)
            gene1_df = gene1_df[gene1_df['status'] == 'success']  # Only successful results
            
        if gene2_file.exists():
            gene2_df = pd.read_pickle(gene2_file)
            gene2_df = gene2_df[gene2_df['status'] == 'success']  # Only successful results
            
        return gene1_df, gene2_df
        
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout processing {gene1_symbol}-{gene2_symbol}")
        return None, None
    except Exception as e:
        logger.warning(f"Error processing {gene1_symbol}-{gene2_symbol}: {e}")
        return None, None


def get_top_metadata_correlations_from_cli(gene_df, top_n=5, alpha=0.05):
    """
    Extract top metadata correlations from CLI results.
    
    Args:
        gene_df: DataFrame from CLI output for a specific gene
        top_n: Number of top correlations to return
        alpha: P-value significance threshold
    
    Returns:
        DataFrame with top correlations
    """
    if gene_df is None or len(gene_df) == 0:
        return pd.DataFrame()
    
    # Filter significant results
    significant_df = gene_df[gene_df['p_value'] <= alpha].copy()
    
    if len(significant_df) == 0:
        return pd.DataFrame()
    
    # Add absolute CCC value for ranking
    significant_df['abs_ccc'] = significant_df['ccc_value'].abs()
    
    # Sort by absolute CCC value and take top N
    top_correlations = significant_df.sort_values('abs_ccc', ascending=False).head(top_n)
    
    return top_correlations[['ccc_value', 'p_value']].reset_index()


def get_common_metadata_correlations_from_cli(gene1_df, gene2_df, gene1_symbol, gene2_symbol, top_n=5, alpha=0.05):
    """
    Find common metadata correlations between two genes using CLI results.
    
    Args:
        gene1_df: CLI results for gene 1
        gene2_df: CLI results for gene 2  
        gene1_symbol: Symbol for gene 1
        gene2_symbol: Symbol for gene 2
        top_n: Number of top common correlations
        alpha: P-value significance threshold
        
    Returns:
        DataFrame with top common correlations
    """
    if gene1_df is None or gene2_df is None or len(gene1_df) == 0 or len(gene2_df) == 0:
        return pd.DataFrame()
    
    # Filter by p-value significance
    sig_df1 = gene1_df[gene1_df["p_value"] <= alpha][["ccc_value"]].reset_index()
    sig_df2 = gene2_df[gene2_df["p_value"] <= alpha][["ccc_value"]].reset_index()
    
    if len(sig_df1) == 0 or len(sig_df2) == 0:
        return pd.DataFrame()
    
    # Reset index to get metadata_column as a regular column
    if 'metadata_column' not in sig_df1.columns:
        sig_df1 = sig_df1.reset_index()
    if 'metadata_column' not in sig_df2.columns:
        sig_df2 = sig_df2.reset_index()
    
    # Merge on metadata column
    merged = pd.merge(sig_df1, sig_df2, on="metadata_column", suffixes=(f"_{gene1_symbol}", f"_{gene2_symbol}"))
    
    if len(merged) == 0:
        return pd.DataFrame()
    
    # Compute score = min(abs(ccc1), abs(ccc2))
    merged["score"] = merged.apply(
        lambda row: min(abs(row[f"ccc_value_{gene1_symbol}"]), abs(row[f"ccc_value_{gene2_symbol}"])),
        axis=1
    )
    
    # Sort and pick top-n
    topn = merged.sort_values("score", ascending=False).head(top_n)
    return topn


def process_gene_pairs_with_metadata_correlations(combined_df, temp_dir, top_n_metadata=5):
    """
    Process gene pairs with metadata correlations from CLI.
    
    Args:
        combined_df: Combined gene pairs dataframe
        temp_dir: Temporary directory for CLI outputs
        top_n_metadata: Number of top metadata correlations per gene
        
    Returns:
        Enhanced dataframe with metadata correlation columns
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Computing metadata correlations for {len(combined_df)} gene pairs")
    logger.info(f"Top N metadata correlations: {top_n_metadata}")
    
    # Create temporary directory
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize enhanced dataframe with new columns
    enhanced_df = combined_df.copy()
    
    # Gene 1 correlations columns
    for i in range(1, top_n_metadata + 1):
        enhanced_df[f'gene1_top{i}_metadata'] = ''
        enhanced_df[f'gene1_top{i}_ccc'] = np.nan
        enhanced_df[f'gene1_top{i}_pvalue'] = np.nan
    
    # Gene 2 correlations columns  
    for i in range(1, top_n_metadata + 1):
        enhanced_df[f'gene2_top{i}_metadata'] = ''
        enhanced_df[f'gene2_top{i}_ccc'] = np.nan
        enhanced_df[f'gene2_top{i}_pvalue'] = np.nan
    
    # Common correlations columns
    for i in range(1, top_n_metadata + 1):
        enhanced_df[f'common_top{i}_metadata'] = ''
    
    # Process each gene pair
    total_pairs = len(enhanced_df)
    processed_pairs = 0
    failed_pairs = 0
    cli_success_pairs = 0
    
    for idx, row in enhanced_df.iterrows():
        try:
            gene1_symbol = row['Gene 1 Symbol']
            gene2_symbol = row['Gene 2 Symbol']
            
            # Progress logging every 10 pairs (since CLI calls are expensive)
            if (idx + 1) % 10 == 0:
                logger.info(f"Processing gene pair {idx + 1:,}/{total_pairs:,} ({((idx + 1)/total_pairs*100):.1f}%) - Current: {gene1_symbol}-{gene2_symbol}")
            
            # Call CLI for this gene pair
            gene1_cli_df, gene2_cli_df = call_metadata_correlation_cli(
                gene1_symbol, gene2_symbol, temp_dir, temp_dir
            )
            
            # Process results if CLI succeeded
            if gene1_cli_df is not None or gene2_cli_df is not None:
                cli_success_pairs += 1
                
                # Get top correlations for gene 1
                if gene1_cli_df is not None:
                    gene1_top = get_top_metadata_correlations_from_cli(gene1_cli_df, top_n_metadata)
                    
                    # Fill gene 1 correlation data
                    for i, (_, corr_row) in enumerate(gene1_top.iterrows()):
                        if i < top_n_metadata:
                            enhanced_df.loc[idx, f'gene1_top{i+1}_metadata'] = corr_row['metadata_column']
                            enhanced_df.loc[idx, f'gene1_top{i+1}_ccc'] = corr_row['ccc_value']
                            enhanced_df.loc[idx, f'gene1_top{i+1}_pvalue'] = corr_row['p_value']
                
                # Get top correlations for gene 2
                if gene2_cli_df is not None:
                    gene2_top = get_top_metadata_correlations_from_cli(gene2_cli_df, top_n_metadata)
                    
                    # Fill gene 2 correlation data
                    for i, (_, corr_row) in enumerate(gene2_top.iterrows()):
                        if i < top_n_metadata:
                            enhanced_df.loc[idx, f'gene2_top{i+1}_metadata'] = corr_row['metadata_column']
                            enhanced_df.loc[idx, f'gene2_top{i+1}_ccc'] = corr_row['ccc_value']
                            enhanced_df.loc[idx, f'gene2_top{i+1}_pvalue'] = corr_row['p_value']
                
                # Get common correlations if both genes have results
                if gene1_cli_df is not None and gene2_cli_df is not None:
                    common_top = get_common_metadata_correlations_from_cli(
                        gene1_cli_df, gene2_cli_df, gene1_symbol, gene2_symbol, top_n_metadata
                    )
                    
                    # Fill common correlation data
                    for i, (_, common_row) in enumerate(common_top.iterrows()):
                        if i < top_n_metadata:
                            enhanced_df.loc[idx, f'common_top{i+1}_metadata'] = common_row['metadata_column']
            
            processed_pairs += 1
            
        except Exception as e:
            failed_pairs += 1
            logger.warning(f"Failed to process gene pair at index {idx}: {gene1_symbol}-{gene2_symbol}: {e}")
            continue
    
    logger.info(f"Metadata correlation processing completed:")
    logger.info(f"  Total pairs: {total_pairs:,}")
    logger.info(f"  Processed successfully: {processed_pairs:,}")
    logger.info(f"  CLI succeeded for: {cli_success_pairs:,}")
    logger.info(f"  Failed: {failed_pairs:,}")
    logger.info(f"  Success rate: {(processed_pairs/total_pairs*100):.1f}%")
    logger.info(f"  CLI success rate: {(cli_success_pairs/total_pairs*100):.1f}%")
    
    # Clean up temporary directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    return enhanced_df


def load_and_enhance_combination(combination, tissue_files, output_dir, top_n_metadata=5, temp_base_dir=None):
    """Load, combine, and enhance tissue files with metadata correlations."""
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
        return None, len(load_errors)
    
    # Combine all dataframes
    logger.info(f"Combining {len(combined_dfs)} dataframes...")
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    logger.info(f"Total rows: {total_rows}")
    
    if load_errors:
        logger.warning(f"Load errors for combination {combination}:")
        for error in load_errors:
            logger.warning(f"  {error}")
    
    # Compute metadata correlations for all gene pairs
    logger.info(f"Computing metadata correlations...")
    temp_dir = temp_base_dir / f"correlations_{combination}"
    enhanced_df = process_gene_pairs_with_metadata_correlations(combined_df, temp_dir, top_n_metadata)
    combined_df = enhanced_df
    logger.info(f"Enhanced dataframe shape: {combined_df.shape}")
    
    # Save combined results
    output_file_pkl = output_dir / f"combined_{combination}_top_gene_pairs_with_metadata.pkl"
    output_file_csv = output_dir / f"combined_{combination}_top_gene_pairs_with_metadata.csv"
    
    combined_df.to_pickle(output_file_pkl)
    combined_df.to_csv(output_file_csv, index=False)
    
    logger.info(f"Saved combined results:")
    logger.info(f"  Pickle: {output_file_pkl}")
    logger.info(f"  CSV: {output_file_csv}")
    
    return combined_df, len(load_errors)


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
            f.write(f"  - combined_{combination}_top_gene_pairs_with_metadata.pkl\n")
            f.write(f"  - combined_{combination}_top_gene_pairs_with_metadata.csv\n")
        
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
    
    parser.add_argument(
        "--top-metadata-correlations",
        type=int,
        default=5,
        help="Number of top metadata correlations to extract for each gene.",
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
        
        # Create temporary base directory for metadata correlations
        temp_base_dir = output_dir / "temp_metadata_correlations"
        temp_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Temporary directory for metadata correlations: {temp_base_dir}")
        
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
            combined_df, load_errors = load_and_enhance_combination(
                combination, tissue_files, output_dir,
                top_n_metadata=args.top_metadata_correlations,
                temp_base_dir=temp_base_dir
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
            logger.info(f"  combined_{combination}_top_gene_pairs_with_metadata.pkl")
            logger.info(f"  combined_{combination}_top_gene_pairs_with_metadata.csv")
        logger.info(f"  combination_summary_report.txt")
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_base_dir)
            logger.info(f"Cleaned up temporary directory: {temp_base_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_base_dir}: {e}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 