#!/usr/bin/env python3
"""
CLI tool for combining top gene pairs across tissues and combination categories.
Processes results from report_top_gene_pairs.py across all tissues and combinations.

For each gene pair, computes fresh correlations with GTEx metadata using metadata_corr_cli.py
and extracts top metadata variables correlated with each gene individually and common to both genes.

Output columns added per gene pair:
- Gene 1: top N metadata correlations (gene1_top{1-N}_metadata, gene1_top{1-N}_ccc, gene1_top{1-N}_pvalue)
- Gene 2: top N metadata correlations (gene2_top{1-N}_metadata, gene2_top{1-N}_ccc, gene2_top{1-N}_pvalue)
- Common: comma-separated list (common_metadata) + detailed top 5 (25 new columns):
  * top{1-5}_common_metadata_name: metadata variable name
  * top{1-5}_common_metadata_gene1_ccc: Gene 1's CCC value for this metadata
  * top{1-5}_common_metadata_gene1_pvalue: Gene 1's p-value for this metadata
  * top{1-5}_common_metadata_gene2_ccc: Gene 2's CCC value for this metadata
  * top{1-5}_common_metadata_gene2_pvalue: Gene 2's p-value for this metadata
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
import multiprocessing as mp
from multiprocessing import Queue
import pickle


def setup_logging(output_dir, combination=None):
    """Set up logging configuration."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent
    
    if combination:
        # Separate log directory for each combination
        log_dir = script_dir / "logs" / f"10-correlate-top-genes-with-metadata_{timestamp}" / combination
        log_filename = f"combination_{combination}.log"
    else:
        # Main log directory
        log_dir = script_dir / "logs" / f"10-correlate-top-genes-with-metadata_{timestamp}"
        log_filename = "10-correlate-top-genes-with-metadata_main.log"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename
    
    # Clear any existing handlers for this process
    logger = logging.getLogger(__name__ if not combination else f"{__name__}.{combination}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
    
    # File handler keeps DEBUG level for detailed CLI output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for INFO level (only for main process)
    if not combination:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    return logger, log_file





def call_metadata_correlation_cli(gene1_symbol, gene2_symbol, tissue, output_dir, temp_dir, args):
    """
    Call the metadata correlation CLI script for a gene pair in a specific tissue.
    
    Args:
        gene1_symbol: First gene symbol
        gene2_symbol: Second gene symbol
        tissue: Tissue name to process (used with --include flag)
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
            cli_script = Path("/home/haoyu/_database/projs/ccc-gpu/nbs/03-manuscript/05-metadata_correlation/metadata_corr_cli.py")
    
    if not cli_script.exists():
        logger.error(f"Cannot find metadata correlation CLI script: {cli_script}")
        return None, None
    
    # Create tissue-specific temporary directory, then gene pair directory
    tissue_temp_dir = temp_dir / tissue
    tissue_temp_dir.mkdir(parents=True, exist_ok=True)
    
    pair_temp_dir = tissue_temp_dir / f"{gene1_symbol}_{gene2_symbol}"
    pair_temp_dir.mkdir(exist_ok=True)
    
    try:
        # Construct command to run the CLI
        cmd = [
            sys.executable, str(cli_script),
            gene1_symbol, gene2_symbol,
            "--output-dir", str(pair_temp_dir),
            # "--expr-data-dir", "/pividori_lab/haoyu_projects/ccc-gpu/data/gtex/gene_selection/all",
            "--expr-data-dir", "/mnt/data/proj_data/ccc-gpu/data/gtex/gene_selection/all",
            "--data-dir", "/mnt/data/proj_data/ccc-gpu/data/gtex",
            "--include", tissue,  # Only process the specific tissue for this gene pair
            "--permutations", str(args.permutations),  # Configurable permutations
            "--n-jobs", str(args.n_jobs),  # Configurable parallel jobs
            "--quiet",  # Reduce verbosity for batch processing
            "--no-csv-output",  # Skip CSV generation, only need pickle files
            "--no-individual-logs"  # Skip individual tissue logs
        ]
        
        # Run the CLI script
        logger.info(f"Running metadata correlation CLI for {gene1_symbol}-{gene2_symbol} in tissue {tissue}")
        logger.debug(f"Output directory: {pair_temp_dir}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800  # 30 minute timeout per gene pair - needed for 10K permutations
        )
        
        # Log CLI output for debugging
        if result.stdout:
            logger.debug(f"CLI stdout for {gene1_symbol}-{gene2_symbol}:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"CLI stderr for {gene1_symbol}-{gene2_symbol}:\n{result.stderr}")
        
        if result.returncode != 0:
            logger.warning(f"CLI failed for {gene1_symbol}-{gene2_symbol} in {tissue} (exit code {result.returncode}): {result.stderr}")
            return None, None
        else:
            logger.info(f"CLI succeeded for {gene1_symbol}-{gene2_symbol} in {tissue}")
        
        # Load results
        gene1_file = pair_temp_dir / f"{gene1_symbol}_all_tissues_correlation_results.pkl"
        gene2_file = pair_temp_dir / f"{gene2_symbol}_all_tissues_correlation_results.pkl"
        
        gene1_df = None
        gene2_df = None
        
        if gene1_file.exists():
            gene1_df = pd.read_pickle(gene1_file)
            total_gene1 = len(gene1_df)
            gene1_df = gene1_df[gene1_df['status'] == 'success']  # Only successful results
            logger.info(f"Loaded {gene1_symbol} results: {len(gene1_df)}/{total_gene1} successful correlations")
        else:
            logger.warning(f"No results file found for {gene1_symbol}: {gene1_file}")
            
        if gene2_file.exists():
            gene2_df = pd.read_pickle(gene2_file)
            total_gene2 = len(gene2_df)
            gene2_df = gene2_df[gene2_df['status'] == 'success']  # Only successful results
            logger.info(f"Loaded {gene2_symbol} results: {len(gene2_df)}/{total_gene2} successful correlations")
        else:
            logger.warning(f"No results file found for {gene2_symbol}: {gene2_file}")
        
        # Log CLI output files created (for debugging)
        cli_output_files = list(pair_temp_dir.glob("*"))
        logger.debug(f"CLI created {len(cli_output_files)} files for {gene1_symbol}-{gene2_symbol}: {[f.name for f in cli_output_files]}")
            
        return gene1_df, gene2_df
        
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout processing {gene1_symbol}-{gene2_symbol} in {tissue}")
        return None, None
    except Exception as e:
        logger.warning(f"Error processing {gene1_symbol}-{gene2_symbol} in {tissue}: {e}")
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
    
    # Filter by p-value significance and include p-values in the result
    sig_df1 = gene1_df[gene1_df["p_value"] <= alpha][["ccc_value", "p_value"]].reset_index()
    sig_df2 = gene2_df[gene2_df["p_value"] <= alpha][["ccc_value", "p_value"]].reset_index()
    
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


def process_gene_pairs_with_metadata_correlations(combined_df, temp_dir, args, top_n_metadata=5):
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
    
    # Critical start logs
    logger.info(f"STARTING metadata correlation processing for {len(combined_df):,} gene pairs")
    logger.info(f"   Configuration: top-{top_n_metadata} metadata correlations per gene")
    logger.info(f"   Permutations: {args.permutations}, Parallel jobs: {args.n_jobs}")
    
    # Estimate processing time
    estimated_time_minutes = len(combined_df) * 1.5 / 60  # Rough estimate: 1.5 min per pair
    logger.info(f"   Estimated processing time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.1f} hours)")
    
    # Create temporary directory
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"   Temporary directory: {temp_dir}")
    
    # Analyze tissues involved
    unique_tissues = combined_df['tissue'].nunique() if 'tissue' in combined_df.columns else 0
    if unique_tissues > 0:
        tissue_counts = combined_df['tissue'].value_counts()
        logger.info(f"   Processing {unique_tissues} tissue(s): {dict(tissue_counts.head(5))}")
    
    import time
    start_time = time.time()
    
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
    
    # Common correlations column (single column with comma-separated values)
    enhanced_df['common_metadata'] = ''
    
    # Top 5 common metadata detailed columns (25 new columns)
    for i in range(1, 6):  # top 1 to top 5
        enhanced_df[f'top{i}_common_metadata_name'] = ''
        enhanced_df[f'top{i}_common_metadata_gene1_ccc'] = np.nan  
        enhanced_df[f'top{i}_common_metadata_gene1_pvalue'] = np.nan
        enhanced_df[f'top{i}_common_metadata_gene2_ccc'] = np.nan
        enhanced_df[f'top{i}_common_metadata_gene2_pvalue'] = np.nan
    
    # Process each gene pair
    total_pairs = len(enhanced_df)
    processed_pairs = 0
    failed_pairs = 0
    cli_success_pairs = 0
    
    # Progress tracking
    last_progress_time = time.time()
    
    # Track common metadata fields across all pairs
    # common_metadata_counter = {}  # Commented out for performance
    
    logger.info(f"PROCESSING STARTED: {total_pairs:,} gene pairs to process")
    
    for idx, row in enhanced_df.iterrows():
        try:
            gene1_symbol = row['Gene 1 Symbol']
            gene2_symbol = row['Gene 2 Symbol']
            tissue = row['tissue']
            
            # Enhanced progress logging with time estimates
            current_time = time.time()
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed_time = current_time - start_time
                progress_pct = (idx + 1) / total_pairs
                
                if progress_pct > 0:
                    estimated_total_time = elapsed_time / progress_pct
                    remaining_time = estimated_total_time - elapsed_time
                    
                    logger.info(f"Progress: {idx + 1:,}/{total_pairs:,} ({progress_pct*100:.1f}%) | "
                               f"Elapsed: {elapsed_time/60:.1f}m | ETA: {remaining_time/60:.1f}m | "
                               f"Current: {gene1_symbol}-{gene2_symbol} ({tissue})")
                
                # Critical milestones
                if idx + 1 in [50, 100, 250, 500] or (idx + 1) % 500 == 0:
                    success_rate = (cli_success_pairs / (idx + 1)) * 100 if idx >= 0 else 0
                    logger.info(f"MILESTONE: Processed {idx + 1:,} pairs | "
                               f"CLI success rate: {success_rate:.1f}% | "
                               f"Failures: {failed_pairs}")
                
                last_progress_time = current_time
            
            # Call CLI for this gene pair (only for the specific tissue)
            gene1_cli_df, gene2_cli_df = call_metadata_correlation_cli(
                gene1_symbol, gene2_symbol, tissue, temp_dir, temp_dir, args
            )
            
            # Process results if CLI succeeded
            if gene1_cli_df is not None or gene2_cli_df is not None:
                cli_success_pairs += 1
                
                # Get top correlations for gene 1
                if gene1_cli_df is not None:
                    gene1_top = get_top_metadata_correlations_from_cli(gene1_cli_df, top_n_metadata)
                    
                    # Log gene 1 top correlations - Commented out for performance
                    # if len(gene1_top) > 0:
                    #     gene1_metadata_fields = [row['metadata_column'] for _, row in gene1_top.iterrows()]
                    #     logger.debug(f"   {gene1_symbol} top metadata: {gene1_metadata_fields[:top_n_metadata]}")
                    
                    # Fill gene 1 correlation data
                    for i, (_, corr_row) in enumerate(gene1_top.iterrows()):
                        if i < top_n_metadata:
                            enhanced_df.loc[idx, f'gene1_top{i+1}_metadata'] = corr_row['metadata_column']
                            enhanced_df.loc[idx, f'gene1_top{i+1}_ccc'] = corr_row['ccc_value']
                            enhanced_df.loc[idx, f'gene1_top{i+1}_pvalue'] = corr_row['p_value']
                
                # Get top correlations for gene 2
                if gene2_cli_df is not None:
                    gene2_top = get_top_metadata_correlations_from_cli(gene2_cli_df, top_n_metadata)
                    
                    # Log gene 2 top correlations - Commented out for performance
                    # if len(gene2_top) > 0:
                    #     gene2_metadata_fields = [row['metadata_column'] for _, row in gene2_top.iterrows()]
                    #     logger.debug(f"   {gene2_symbol} top metadata: {gene2_metadata_fields[:top_n_metadata]}")
                    
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
                    
                    # Log common metadata correlations (most important!) - Commented out for performance
                    # if len(common_top) > 0:
                    #     common_metadata_fields = [row['metadata_column'] for _, row in common_top.iterrows()]
                    #     common_scores = [f"{row['metadata_column']}({row['score']:.3f})" for _, row in common_top.iterrows()]
                    #     logger.info(f"   COMMON metadata for {gene1_symbol}-{gene2_symbol}: {common_metadata_fields}")
                    #     logger.debug(f"   COMMON with scores: {common_scores}")
                    #     
                    #     # Count common metadata fields for summary
                    #     for field in common_metadata_fields:
                    #         common_metadata_counter[field] = common_metadata_counter.get(field, 0) + 1
                    # else:
                    #     logger.debug(f"   No common metadata found for {gene1_symbol}-{gene2_symbol}")
                    
                    # Fill common correlation data as comma-separated list AND detailed columns
                    if len(common_top) > 0:
                        common_metadata_list = []
                        
                        # Fill detailed columns for top 5 common metadata
                        for i, (_, common_row) in enumerate(common_top.iterrows()):
                            if i < top_n_metadata:
                                common_metadata_list.append(common_row['metadata_column'])
                            
                            # Fill detailed columns (up to 5 entries)
                            if i < 5:  # Only fill top 5 detailed columns
                                col_num = i + 1
                                metadata_name = common_row['metadata_column']
                                
                                # Fill the detailed columns
                                enhanced_df.loc[idx, f'top{col_num}_common_metadata_name'] = metadata_name
                                enhanced_df.loc[idx, f'top{col_num}_common_metadata_gene1_ccc'] = common_row[f'ccc_value_{gene1_symbol}']
                                enhanced_df.loc[idx, f'top{col_num}_common_metadata_gene1_pvalue'] = common_row[f'p_value_{gene1_symbol}']
                                enhanced_df.loc[idx, f'top{col_num}_common_metadata_gene2_ccc'] = common_row[f'ccc_value_{gene2_symbol}']
                                enhanced_df.loc[idx, f'top{col_num}_common_metadata_gene2_pvalue'] = common_row[f'p_value_{gene2_symbol}']
                        
                        # Create comma-separated string for backward compatibility
                        enhanced_df.loc[idx, 'common_metadata'] = ','.join(common_metadata_list)
                    else:
                        enhanced_df.loc[idx, 'common_metadata'] = ''
                        # Clear all detailed columns if no common metadata
                        for i in range(1, 6):
                            enhanced_df.loc[idx, f'top{i}_common_metadata_name'] = ''
                            enhanced_df.loc[idx, f'top{i}_common_metadata_gene1_ccc'] = np.nan
                            enhanced_df.loc[idx, f'top{i}_common_metadata_gene1_pvalue'] = np.nan
                            enhanced_df.loc[idx, f'top{i}_common_metadata_gene2_ccc'] = np.nan
                            enhanced_df.loc[idx, f'top{i}_common_metadata_gene2_pvalue'] = np.nan
            
            processed_pairs += 1
            
        except Exception as e:
            failed_pairs += 1
            logger.warning(f"FAILED gene pair at index {idx}: {gene1_symbol}-{gene2_symbol} in {tissue}: {e}")
            
            # Critical warning for high failure rates
            if failed_pairs > 0 and (idx + 1) >= 20:  # Check after at least 20 attempts
                failure_rate = (failed_pairs / (idx + 1)) * 100
                if failure_rate > 20:  # Warning if >20% failure rate
                    logger.warning(f"HIGH FAILURE RATE: {failure_rate:.1f}% ({failed_pairs}/{idx + 1}) - Review configuration!")
                elif failure_rate > 50:  # Critical if >50% failure rate  
                    logger.error(f"CRITICAL FAILURE RATE: {failure_rate:.1f}% - Consider stopping and debugging!")
            
            continue
    
    # Final completion logs with enhanced statistics
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_pair = total_time / total_pairs if total_pairs > 0 else 0
    
    logger.info(f"METADATA CORRELATION PROCESSING COMPLETED!")
    logger.info(f"   FINAL STATISTICS:")
    logger.info(f"   ├─ Total pairs processed: {total_pairs:,}")
    logger.info(f"   ├─ Successfully processed: {processed_pairs:,}")
    logger.info(f"   ├─ CLI successes: {cli_success_pairs:,}")
    logger.info(f"   ├─ Failed pairs: {failed_pairs:,}")
    logger.info(f"   ├─ Overall success rate: {(processed_pairs/total_pairs*100):.1f}%")
    logger.info(f"   ├─ CLI success rate: {(cli_success_pairs/total_pairs*100):.1f}%")
    logger.info(f"   └─ Processing time: {total_time/60:.1f} minutes ({avg_time_per_pair:.1f}s per pair)")
    
    # Critical warnings for poor performance
    cli_success_rate = (cli_success_pairs/total_pairs*100) if total_pairs > 0 else 0
    if cli_success_rate < 50:
        logger.error(f"CRITICAL: Very low CLI success rate ({cli_success_rate:.1f}%)! Check configuration and data quality.")
    elif cli_success_rate < 80:
        logger.warning(f"WARNING: Low CLI success rate ({cli_success_rate:.1f}%). Consider reviewing failed pairs.")
    else:
        logger.info(f"GOOD: CLI success rate is acceptable ({cli_success_rate:.1f}%)")
    
    # Tissue-specific summary if applicable
    if 'tissue' in combined_df.columns:
        tissue_summary = combined_df['tissue'].value_counts()
        logger.info(f"   Processed tissues: {dict(tissue_summary)}")
    
    # Most frequent common metadata fields summary - Commented out for performance
    # if common_metadata_counter:
    #     # Sort by frequency and get top 10
    #     sorted_common = sorted(common_metadata_counter.items(), key=lambda x: x[1], reverse=True)
    #     top_common = sorted_common[:10]
    #     
    #     logger.info(f"   TOP COMMON METADATA FIELDS (across all gene pairs):")
    #     for i, (field, count) in enumerate(top_common, 1):
    #         pct = (count / cli_success_pairs) * 100 if cli_success_pairs > 0 else 0
    #         logger.info(f"      {i:2d}. {field}: {count:,} pairs ({pct:.1f}%)")
    #         
    #     total_unique_common = len(common_metadata_counter)
    #     logger.info(f"   Total unique common metadata fields found: {total_unique_common}")
    # else:
    #     logger.warning(f"   No common metadata fields found across any gene pairs")
    
    # Clean up temporary directory (organized by tissue)
    try:
        import shutil
        # Log directory structure for debugging
        if temp_dir.exists():
            tissue_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            logger.debug(f"Cleaning up temp directory with {len(tissue_dirs)} tissue subdirectories")
            for tissue_dir in tissue_dirs:
                gene_pair_dirs = [d for d in tissue_dir.iterdir() if d.is_dir()]
                logger.debug(f"  {tissue_dir.name}: {len(gene_pair_dirs)} gene pairs")
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    logger.info(f"process_gene_pairs_with_metadata_correlations COMPLETED - returning enhanced dataframe with {len(enhanced_df.columns)} columns")
    return enhanced_df


def process_combination_worker(combination, tissue_files, output_dir, args, top_n_metadata=5, temp_base_dir=None, result_queue=None):
    """Worker function to process a single combination in a separate process."""
    try:
        # Set up logging for this combination
        logger, log_file = setup_logging(output_dir, combination)
        
        logger.info(f"WORKER STARTED for combination: {combination}")
        logger.info(f"Process ID: {os.getpid()}")
        logger.info(f"Log file: {log_file}")
        
        # Call the actual processing function
        combined_df, load_errors = load_and_enhance_combination(
            combination, tissue_files, output_dir, args, top_n_metadata, temp_base_dir, logger
        )
        
        # Prepare result
        if combined_df is not None:
            result = {
                'combination': combination,
                'success': True,
                'tissues_processed': len(tissue_files),
                'total_rows': len(combined_df),
                'load_errors': load_errors,
                'unique_genes': len(set(combined_df['gene1'].tolist() + combined_df['gene2'].tolist())) if 'gene1' in combined_df.columns and 'gene2' in combined_df.columns else 0,
                'log_file': str(log_file)
            }
        else:
            result = {
                'combination': combination,
                'success': False,
                'error': 'Processing failed',
                'log_file': str(log_file)
            }
        
        logger.info(f"WORKER COMPLETED for combination: {combination}")
        
        if result_queue is not None:
            result_queue.put(result)
        return result
        
    except Exception as e:
        error_result = {
            'combination': combination,
            'success': False,
            'error': str(e),
            'log_file': str(log_file) if 'log_file' in locals() else 'N/A'
        }
        if result_queue is not None:
            result_queue.put(error_result)
        return error_result


def load_and_enhance_combination(combination, tissue_files, output_dir, args, top_n_metadata=5, temp_base_dir=None, logger=None):
    """Load, combine, and enhance tissue files with metadata correlations."""
    if logger is None:
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
    
    # Conditionally compute metadata correlations
    if args.combine_only:
        logger.info(f"Skipping metadata correlations (--combine-only flag enabled)")
        # Use original combined dataframe without metadata enhancement
        enhanced_df = combined_df
        output_suffix = "_combined"
    else:
        logger.info(f"Computing metadata correlations...")
        temp_dir = temp_base_dir / f"correlations_{combination}"
        enhanced_df = process_gene_pairs_with_metadata_correlations(combined_df, temp_dir, args, top_n_metadata)
        logger.info(f"Enhanced dataframe shape: {enhanced_df.shape}")
        output_suffix = "_with_metadata"
    
    # Update combined_df reference
    combined_df = enhanced_df
    
    # Save combined results with appropriate suffix including top_n
    output_file_pkl = output_dir / f"combined_{combination}_top_{args.top}_gene_pairs{output_suffix}.pkl"
    output_file_csv = output_dir / f"combined_{combination}_top_{args.top}_gene_pairs{output_suffix}.csv"
    
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
    
    # Prioritize 'c-high-p-low-s-low' combination first, then sort the rest
    combinations_list = list(combinations)
    priority_combination = 'c-high-p-low-s-low'
    
    if priority_combination in combinations_list:
        combinations_list.remove(priority_combination)
        combinations_list.sort()
        combinations = [priority_combination] + combinations_list
    else:
        combinations = sorted(combinations_list)
    
    logger.info(f"Discovered {len(tissues)} tissues")
    logger.info(f"Discovered {len(combinations)} combinations: {combinations}")
    logger.info(f"Processing order: {priority_combination} will be processed first")
    
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


def process_combinations_sequential(combinations, found_files, output_dir, args, temp_base_dir, logger):
    """Process combinations sequentially (original behavior)."""
    results_summary = {}
    
    for i, combination in enumerate(combinations, 1):
        logger.info(f"\n[{i}/{len(combinations)}] Processing combination: {combination}")
        
        if combination not in found_files:
            logger.warning(f"No files found for combination: {combination}")
            continue
        
        tissue_files = found_files[combination]
        
        # Filter to specific tissue if requested (for debugging)
        if args.tissue:
            if args.tissue in tissue_files:
                tissue_files = {args.tissue: tissue_files[args.tissue]}
                logger.info(f"  Filtering to single tissue: {args.tissue} (debugging mode)")
            else:
                logger.warning(f"  Requested tissue '{args.tissue}' not found in combination '{combination}'")
                logger.warning(f"  Available tissues: {list(tissue_files.keys())}")
                continue
        
        combo_start_time = time.time()
        combined_df, load_errors = load_and_enhance_combination(
            combination, tissue_files, output_dir, args,
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
    
    return results_summary


def process_combinations_parallel(combinations, found_files, output_dir, args, temp_base_dir, logger):
    """Process combinations in parallel using multiprocessing."""
    results_summary = {}
    
    # Filter combinations that have files
    valid_combinations = [c for c in combinations if c in found_files]
    
    if not valid_combinations:
        logger.error("No valid combinations found")
        return results_summary
    
    logger.info(f"Starting parallel processing of {len(valid_combinations)} combinations")
    logger.info(f"Maximum parallel workers: {args.max_parallel_combinations}")
    
    # Prepare arguments for each worker
    worker_args = []
    for combination in valid_combinations:
        tissue_files = found_files[combination]
        
        # Filter to specific tissue if requested (for debugging)
        if args.tissue:
            if args.tissue in tissue_files:
                tissue_files = {args.tissue: tissue_files[args.tissue]}
                logger.info(f"  Filtering combination '{combination}' to single tissue: {args.tissue}")
            else:
                logger.warning(f"  Skipping combination '{combination}': tissue '{args.tissue}' not found")
                continue
        
        worker_args.append((
            combination, tissue_files, output_dir, args, 
            args.top_metadata_correlations, temp_base_dir
        ))
    
    if not worker_args:
        logger.error("No valid worker arguments after filtering")
        return results_summary
    
    # Start processing with multiprocessing
    logger.info(f"Launching {len(worker_args)} parallel workers")
    
    try:
        # Use multiprocessing Pool
        max_workers = min(args.max_parallel_combinations, len(worker_args))
        with mp.Pool(processes=max_workers) as pool:
            # Start all workers
            async_results = []
            for worker_arg in worker_args:
                result = pool.apply_async(process_combination_worker, worker_arg)
                async_results.append((worker_arg[0], result))  # (combination, async_result)
            
            # Monitor progress
            completed = 0
            total = len(async_results)
            
            while completed < total:
                time.sleep(5)  # Check every 5 seconds
                
                current_completed = sum(1 for _, result in async_results if result.ready())
                if current_completed > completed:
                    completed = current_completed
                    logger.info(f"Progress: {completed}/{total} combinations completed")
                    
                    # Log which combinations are done
                    for combination, result in async_results:
                        if result.ready() and combination not in results_summary:
                            try:
                                worker_result = result.get()
                                if worker_result['success']:
                                    results_summary[combination] = {
                                        'tissues_processed': worker_result['tissues_processed'],
                                        'total_rows': worker_result['total_rows'],
                                        'load_errors': worker_result['load_errors'],
                                        'runtime': 0,  # Will be updated later
                                        'unique_genes': worker_result['unique_genes'],
                                        'log_file': worker_result['log_file']
                                    }
                                    logger.info(f"✓ Combination '{combination}' completed successfully")
                                    logger.info(f"  Rows: {worker_result['total_rows']:,}")
                                    logger.info(f"  Unique genes: {worker_result['unique_genes']:,}")
                                    logger.info(f"  Log file: {worker_result['log_file']}")
                                else:
                                    logger.error(f"✗ Combination '{combination}' failed: {worker_result.get('error', 'Unknown error')}")
                                    logger.error(f"  Log file: {worker_result.get('log_file', 'N/A')}")
                            except Exception as e:
                                logger.error(f"✗ Error getting result for combination '{combination}': {e}")
            
            # Ensure all workers complete
            pool.close()
            pool.join()
            
        logger.info(f"All {total} combination workers completed")
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        # Fall back to sequential processing
        logger.info("Falling back to sequential processing")
        return process_combinations_sequential(combinations, found_files, output_dir, args, temp_base_dir, logger)
    
    return results_summary


def generate_summary_report(results_summary, output_dir, top_n, combine_only=False, tissue=None):
    """Generate a comprehensive summary report."""
    logger = logging.getLogger(__name__)
    
    # Build filename with top_n and optionally tissue
    filename_parts = ["combination_summary_report", f"top_{top_n}"]
    if tissue:
        filename_parts.append(tissue)
    summary_filename = "_".join(filename_parts) + ".txt"
    summary_file = output_dir / summary_filename
    
    with open(summary_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("TOP GENE PAIRS COMBINATION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Top N Gene Pairs: {top_n}\n")
        f.write(f"Processing Mode: {'Combine Only (No Metadata Correlations)' if combine_only else 'Full Processing with Metadata Correlations'}\n")
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
        file_suffix = "_combined" if combine_only else "_with_metadata"
        for combination in sorted(results_summary.keys()):
            f.write(f"Combination: {combination}\n")
            f.write(f"  - combined_{combination}_top_{top_n}_gene_pairs{file_suffix}.pkl\n")
            f.write(f"  - combined_{combination}_top_{top_n}_gene_pairs{file_suffix}.csv\n")
            if 'log_file' in results_summary[combination]:
                f.write(f"  - Log file: {results_summary[combination]['log_file']}\n")
        
        f.write(f"\nSummary report: {summary_filename}\n")
        f.write(f"Main log file: Available in logs/ subdirectory\n")
        f.write(f"Individual combination logs: Available in logs/*/combination_*.log\n")
    
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
    
    # Performance tuning arguments
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical significance testing (higher = more accurate but slower).",
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16,
        help="Number of parallel jobs for correlation computation.",
    )
    
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine gene pairs across tissues without computing metadata correlations (much faster).",
    )
    
    parser.add_argument(
        "--tissue",
        type=str,
        help="Process only a specific tissue (for debugging). If not provided, all tissues will be processed.",
    )
    
    parser.add_argument(
        "--combinations",
        nargs="*",
        default=["c-high-p-low-s-low"],
        help="List of combination categories to process. Default: ['c-high-p-low-s-low']. Use --list-combinations to see available options.",
    )
    
    parser.add_argument(
        "--parallel-combinations",
        action="store_true",
        help="Process different combinations in parallel (recommended for faster processing).",
    )
    
    parser.add_argument(
        "--max-parallel-combinations",
        type=int,
        default=5,
        help="Maximum number of combinations to process in parallel (default: 5).",
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging (main process)
        logger, log_file = setup_logging(output_dir)
        
        logger.info(f"Starting top gene pairs combination analysis")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Top N gene pairs: {args.top}")
        logger.info(f"Mode: {'Combine only (no metadata correlations)' if args.combine_only else 'Full processing with metadata correlations'}")
        logger.info(f"Parallel processing: {'Enabled' if args.parallel_combinations else 'Disabled'}")
        if args.parallel_combinations:
            logger.info(f"Max parallel combinations: {args.max_parallel_combinations}")
        if args.tissue:
            logger.info(f"Tissue filtering: Only processing tissue '{args.tissue}' (debugging mode)")
        else:
            logger.info(f"Tissue filtering: Processing all tissues")
        
        # Discover tissues and combinations
        tissues, combinations = discover_tissues_and_combinations(args.data_dir)
        
        # Filter combinations if specific ones are requested
        available_combinations = set(combinations)
        requested_combinations = set(args.combinations) if args.combinations else available_combinations
        
        # Check if requested combinations exist
        invalid_combinations = requested_combinations - available_combinations
        if invalid_combinations:
            logger.error(f"Invalid combination(s) requested: {list(invalid_combinations)}")
            logger.error(f"Available combinations: {list(available_combinations)}")
            sys.exit(1)
        
        # Filter to only requested combinations
        original_count = len(combinations)
        combinations = [c for c in combinations if c in requested_combinations]
        
        if args.combinations and args.combinations != ["c-high-p-low-s-low"]:
            logger.info(f"Combination filtering: Processing {len(combinations)} selected combination(s): {combinations}")
        elif args.combinations == ["c-high-p-low-s-low"]:
            logger.info(f"Combination filtering: Using default - processing only 'c-high-p-low-s-low'")
        else:
            logger.info(f"Combination filtering: Processing all {len(combinations)} combinations")
        
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
        
        # Create temporary base directory for metadata correlations (only if needed)
        temp_base_dir = None
        if not args.combine_only:
            temp_base_dir = output_dir / "temp_metadata_correlations"
            temp_base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Temporary directory for metadata correlations: {temp_base_dir}")
        else:
            logger.info(f"Skipping temporary directory creation (combine-only mode)")
        
        # Find all top gene files
        found_files, tissues, combinations = find_top_gene_files(args.data_dir, args.top)
        
        if not found_files:
            logger.error("No matching top gene pairs files found")
            sys.exit(1)
        
        # Process each combination (parallel or sequential)
        results_summary = {}
        total_start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING COMBINATIONS")
        logger.info(f"{'='*60}")
        logger.info(f"Parallel processing: {'Enabled' if args.parallel_combinations else 'Disabled'}")
        
        if args.parallel_combinations:
            results_summary = process_combinations_parallel(
                combinations, found_files, output_dir, args, temp_base_dir, logger
            )
        else:
            results_summary = process_combinations_sequential(
                combinations, found_files, output_dir, args, temp_base_dir, logger
            )
        
        total_end_time = time.time()
        total_runtime = total_end_time - total_start_time
        
        # Generate summary report
        generate_summary_report(results_summary, output_dir, args.top, args.combine_only, args.tissue)
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        logger.info(f"Combinations processed: {len(results_summary)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Main log file: {log_file}")
        if args.parallel_combinations:
            logger.info(f"Individual combination log files:")
            for combination in sorted(results_summary.keys()):
                if 'log_file' in results_summary[combination]:
                    logger.info(f"  {combination}: {results_summary[combination]['log_file']}")
                else:
                    logger.info(f"  {combination}: Log file not available")
        
        # Show which files were created
        logger.info("\nOutput files created:")
        file_suffix = "_combined" if args.combine_only else "_with_metadata"
        for combination in sorted(results_summary.keys()):
            logger.info(f"  combined_{combination}_top_{args.top}_gene_pairs{file_suffix}.pkl")
            logger.info(f"  combined_{combination}_top_{args.top}_gene_pairs{file_suffix}.csv")
        
        # Build summary report filename with same logic as in generate_summary_report
        summary_filename_parts = ["combination_summary_report", f"top_{args.top}"]
        if args.tissue:
            summary_filename_parts.append(args.tissue)
        summary_filename = "_".join(summary_filename_parts) + ".txt"
        logger.info(f"  {summary_filename}")
        
        # Clean up temporary directory (only if it was created)
        if temp_base_dir is not None:
            try:
                import shutil
                shutil.rmtree(temp_base_dir)
                logger.info(f"Cleaned up temporary directory: {temp_base_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_base_dir}: {e}")
        else:
            logger.info("No temporary directory to clean up (combine-only mode)")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    mp.freeze_support()
    main() 