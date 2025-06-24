#!/usr/bin/env python3
"""
CLI tool for exploring gene expression correlations with metadata.
Converted from 00-data-exploration.ipynb
"""

import argparse
import sys
import warnings
import re
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from ccc.coef import ccc

# Suppress specific NumPy warnings
warnings.filterwarnings('ignore', message='invalid value encountered in cast')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')


def find_expression_files(expr_data_dir, include_patterns=None, exclude_patterns=None):
    """Find expression files matching include/exclude patterns."""
    expr_data_dir = Path(expr_data_dir)
    
    if not expr_data_dir.exists():
        raise FileNotFoundError(f"Expression data directory not found: {expr_data_dir}")
    
    # Find all .pkl files with the expected pattern
    pattern = re.compile(r'gtex_v8_data_(.+)-var_pc_log2\.pkl$')
    all_files = []
    
    for file_path in expr_data_dir.glob("*.pkl"):
        match = pattern.match(file_path.name)
        if match:
            tissue_name = match.group(1)
            all_files.append((file_path, tissue_name))
    
    if not all_files:
        raise FileNotFoundError(f"No matching expression files found in {expr_data_dir}")
    
    # Apply include patterns
    if include_patterns:
        filtered_files = []
        for file_path, tissue_name in all_files:
            for pattern in include_patterns:
                if re.search(pattern.lower(), tissue_name.lower()) or re.search(pattern.lower(), file_path.name.lower()):
                    filtered_files.append((file_path, tissue_name))
                    break
        all_files = filtered_files
    
    # Apply exclude patterns
    if exclude_patterns:
        filtered_files = []
        for file_path, tissue_name in all_files:
            excluded = False
            for pattern in exclude_patterns:
                if re.search(pattern.lower(), tissue_name.lower()) or re.search(pattern.lower(), file_path.name.lower()):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append((file_path, tissue_name))
        all_files = filtered_files
    
    print(f"Found {len(all_files)} expression files to process:")
    for file_path, tissue_name in all_files:
        print(f"  {tissue_name}: {file_path.name}")
    
    return all_files


def load_metadata_and_gene_map():
    """Load metadata and gene mapping files."""
    # Define paths
    MANUSCRIPT_DIR = Path("/mnt/data/projs/manuscripts/ccc-gpu/")
    METADATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gtex_metadata")
    
    # File paths
    METADATA_FILE = METADATA_DIR / "gtex_v8-sample_metadata.pkl"
    GENE_MAP_FILE = MANUSCRIPT_DIR / "data" / "gtex_gene_id_symbol_mappings.pkl"
    
    # Check if files exist
    for file_path in [METADATA_FILE, GENE_MAP_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print("Loading metadata and gene mapping files...")
    
    # Load data
    gtex_metadata = pd.read_pickle(METADATA_FILE)
    gene_map = pd.read_pickle(GENE_MAP_FILE)
    
    print(f"Loaded metadata: {gtex_metadata.shape}")
    print(f"Loaded gene mapping: {gene_map.shape}")
    
    return gtex_metadata, gene_map


def setup_tissue_logger(gene_symbol, tissue_name, output_dir):
    """Set up a logger for a specific gene-tissue combination."""
    logger_name = f"tissue_{gene_symbol}_{tissue_name}"
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = output_dir / f"{gene_symbol}_{tissue_name}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, log_file


def setup_summary_logger(gene_symbols, output_dir):
    """Set up a logger for the main function summary."""
    logger_name = "summary"
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create file handler
    genes_connected = "_".join(gene_symbols)
    log_file = output_dir / f"{genes_connected}_summary.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, log_file


def log_and_print(message, logger=None):
    """Print message and log it if logger is provided."""
    print(message)
    if logger:
        logger.info(message)


def get_gene_id(gene_symbol, gene_map):
    """Get gene ID from gene symbol."""
    matches = gene_map.loc[gene_map["gene_symbol"] == gene_symbol, "gene_ens_id"]
    
    if len(matches) == 0:
        raise ValueError(f"Gene symbol '{gene_symbol}' not found in gene mapping")
    elif len(matches) > 1:
        print(f"Warning: Multiple matches found for '{gene_symbol}': {matches.tolist()}")
        print(f"Using first match: {matches.iloc[0]}")
    
    return matches.iloc[0]


def compute_correlations_for_tissue(gene_symbol, tissue_name, expr_file_path, gtex_metadata, gene_map, 
                                  output_dir, pvalue_n_perms=1000000, n_jobs=1):
    """Compute correlation between gene expression and all metadata columns for a specific tissue."""
    
    # Set up logging for this tissue
    logger, log_file = setup_tissue_logger(gene_symbol, tissue_name, output_dir)
    
    log_and_print(f"\n{'='*60}", logger)
    log_and_print(f"Processing tissue: {tissue_name}", logger)
    log_and_print(f"File: {expr_file_path.name}", logger)
    log_and_print(f"Log file: {log_file}", logger)
    log_and_print(f"{'='*60}", logger)
    
    # Load expression data
    log_and_print("Loading expression data...", logger)
    expr_data = pd.read_pickle(expr_file_path)
    log_and_print(f"Expression data shape: {expr_data.shape}", logger)
    
    # Get gene ID
    gene_id = get_gene_id(gene_symbol, gene_map)
    log_and_print(f"Gene ID for {gene_symbol}: {gene_id}", logger)
    
    # Check if gene exists in this tissue
    if gene_id not in expr_data.index:
        log_and_print(f"Warning: Gene ID '{gene_id}' not found in {tissue_name} expression data", logger)
        return None, gene_id
    
    # Get sample IDs from expression data
    sample_ids = expr_data.columns
    log_and_print(f"Number of samples: {len(sample_ids)}", logger)
    
    # Get gene expression data
    gene_expr_row = expr_data.loc[gene_id]
    
    # Get metadata for these samples (only for samples that exist in both datasets)
    common_samples = sample_ids.intersection(gtex_metadata.index)
    if len(common_samples) == 0:
        log_and_print(f"Warning: No common samples found between {tissue_name} expression data and metadata", logger)
        return None, gene_id
    
    log_and_print(f"Common samples: {len(common_samples)}", logger)
    
    # Filter to common samples
    gene_expr_filtered = gene_expr_row.loc[common_samples]
    sample_metadata = gtex_metadata.loc[common_samples]
    
    log_and_print(f"Computing CCC between {gene_symbol} expression and all metadata columns...", logger)
    log_and_print(f"Using {pvalue_n_perms} permutations and {n_jobs} jobs", logger)
    log_and_print(f"Processing {len(sample_metadata.columns)} metadata columns...", logger)
    
    # Initialize results
    results = []
    
    # Iterate through all metadata columns
    for i, column in enumerate(sample_metadata.columns, 1):
        log_and_print(f"Processing column {i}/{len(sample_metadata.columns)}: {column}", logger)
        
        try:
            metadata_vector = sample_metadata[column]
            
            # Skip columns with all NaN values
            if metadata_vector.isna().all():
                log_and_print(f"  Skipping {column}: all values are NaN", logger)
                results.append({
                    'metadata_column': column,
                    'ccc_value': np.nan,
                    'p_value': np.nan,
                    'status': 'all_nan'
                })
                continue
            
            # Skip columns with only one unique value (after removing NaN)
            unique_values = metadata_vector.dropna().nunique()
            if unique_values <= 1:
                log_and_print(f"  Skipping {column}: only {unique_values} unique value(s)", logger)
                results.append({
                    'metadata_column': column,
                    'ccc_value': np.nan,
                    'p_value': np.nan,
                    'status': 'insufficient_variation'
                })
                continue
            
            # Compute CCC (suppress numpy warnings during computation)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ccc_val, ccc_pval = ccc(gene_expr_filtered, metadata_vector, 
                                       pvalue_n_perms=pvalue_n_perms, n_jobs=n_jobs)
            
            results.append({
                'metadata_column': column,
                'ccc_value': ccc_val,
                'p_value': ccc_pval,
                'status': 'success'
            })
            
            log_and_print(f"  CCC: {ccc_val:.6f}, p-value: {ccc_pval:.2e}", logger)
            
        except Exception as e:
            log_and_print(f"  Error processing {column}: {e}", logger)
            results.append({
                'metadata_column': column,
                'ccc_value': np.nan,
                'p_value': np.nan,
                'status': f'error: {str(e)}'
            })
    
    # Convert to DataFrame with metadata column names as index
    results_df = pd.DataFrame(results)
    results_df.set_index('metadata_column', inplace=True)
    
    # Add tissue information
    results_df['tissue'] = tissue_name
    results_df['gene_symbol'] = gene_symbol
    results_df['gene_id'] = gene_id
    results_df['n_samples'] = len(common_samples)
    
    # Log completion
    successful_analyses = results_df[results_df['status'] == 'success']
    log_and_print(f"\nCompleted processing {tissue_name}:", logger)
    log_and_print(f"  Total metadata columns: {len(results_df)}", logger)
    log_and_print(f"  Successful analyses: {len(successful_analyses)}", logger)
    log_and_print(f"  Skipped/Failed: {len(results_df) - len(successful_analyses)}", logger)
    
    # Close the logger
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    
    return results_df, gene_id


def main():
    parser = argparse.ArgumentParser(
        description="Analyze gene expression correlations with metadata using CCC across multiple tissues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "gene_symbols",
        nargs="+",
        help="Gene symbol(s) to analyze (e.g., RASSF2 TP53 BRCA1)"
    )
    
    parser.add_argument(
        "--expr-data-dir",
        default="/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8/gene_selection/all",
        help="Directory containing expression data files"
    )
    
    parser.add_argument(
        "--include",
        nargs="*",
        help="Include only tissues matching these patterns (fuzzy match on tissue name)"
    )
    
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Exclude tissues matching these patterns (fuzzy match on tissue name)"
    )
    
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000000,
        # default=100000,
        help="Number of permutations for p-value calculation"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=24,
        help="Number of parallel jobs for computation"
    )
    
    parser.add_argument(
        "--list-metadata-columns",
        action="store_true",
        help="List available metadata columns and exit"
    )
    
    parser.add_argument(
        "--list-tissues",
        action="store_true",
        help="List available tissue files and exit"
    )
    
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output files (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up summary logger
        summary_logger, summary_log_file = setup_summary_logger(args.gene_symbols, output_dir)
        
        log_and_print(f"Output directory: {output_dir.absolute()}", summary_logger)
        log_and_print(f"Summary log file: {summary_log_file}", summary_logger)
        log_and_print(f"Gene symbols to analyze: {', '.join(args.gene_symbols)}", summary_logger)
        
        # Find expression files
        expression_files = find_expression_files(
            args.expr_data_dir,
            include_patterns=args.include,
            exclude_patterns=args.exclude
        )
        
        # If user wants to list tissues
        if args.list_tissues:
            log_and_print(f"Available expression files in {args.expr_data_dir}:", summary_logger)
            for file_path, tissue_name in expression_files:
                log_and_print(f"  {tissue_name}: {file_path.name}", summary_logger)
            return
        
        # Load metadata and gene mapping
        gtex_metadata, gene_map = load_metadata_and_gene_map()
        
        # If user wants to list metadata columns
        if args.list_metadata_columns:
            log_and_print("Available metadata columns:", summary_logger)
            for col in sorted(gtex_metadata.columns):
                log_and_print(f"  {col}", summary_logger)
            return
        
        # Process each gene symbol
        all_genes_results = {}
        total_start_time = time.time()
        
        for gene_idx, gene_symbol in enumerate(args.gene_symbols, 1):
            log_and_print(f"\n{'='*100}", summary_logger)
            log_and_print(f"PROCESSING GENE {gene_idx}/{len(args.gene_symbols)}: {gene_symbol}", summary_logger)
            log_and_print(f"{'='*100}", summary_logger)
            
            # Process each tissue for this gene
            all_results = {}
            gene_id = None
            tissue_runtimes = {}
            gene_start_time = time.time()
            
            for i, (expr_file_path, tissue_name) in enumerate(expression_files, 1):
                log_and_print(f"\n[{i}/{len(expression_files)}] Starting processing for {gene_symbol} in {tissue_name}...", summary_logger)
                tissue_start_time = time.time()
                
                try:
                    results_df, current_gene_id = compute_correlations_for_tissue(
                        gene_symbol,
                        tissue_name,
                        expr_file_path,
                        gtex_metadata,
                        gene_map,
                        output_dir,
                        args.permutations,
                        args.n_jobs
                    )
                
                    tissue_end_time = time.time()
                    tissue_runtime = tissue_end_time - tissue_start_time
                    tissue_runtimes[tissue_name] = tissue_runtime
                    
                    if results_df is not None:
                        all_results[tissue_name] = results_df
                        gene_id = current_gene_id
                        
                        # Save individual tissue results
                        output_file = output_dir / f"{gene_symbol}_{tissue_name}_correlation_results.pkl"
                        log_file = output_dir / f"{gene_symbol}_{tissue_name}.log"
                        results_df.to_pickle(output_file)
                        log_and_print(f"Results for {gene_symbol} in {tissue_name} saved to: {output_file}", summary_logger)
                        log_and_print(f"Log file for {gene_symbol} in {tissue_name} saved to: {log_file}", summary_logger)
                        log_and_print(f"Runtime for {gene_symbol} in {tissue_name}: {tissue_runtime:.2f} seconds ({tissue_runtime/60:.2f} minutes)", summary_logger)
                    else:
                        log_file = output_dir / f"{gene_symbol}_{tissue_name}.log"
                        log_and_print(f"No results generated for {gene_symbol} in {tissue_name}", summary_logger)
                        log_and_print(f"Log file for {gene_symbol} in {tissue_name} saved to: {log_file}", summary_logger)
                        log_and_print(f"Runtime for {gene_symbol} in {tissue_name}: {tissue_runtime:.2f} seconds ({tissue_runtime/60:.2f} minutes)", summary_logger)
                        
                except Exception as e:
                    tissue_end_time = time.time()
                    tissue_runtime = tissue_end_time - tissue_start_time
                    tissue_runtimes[tissue_name] = tissue_runtime
                    log_file = output_dir / f"{gene_symbol}_{tissue_name}.log"
                    log_and_print(f"Error processing {gene_symbol} in {tissue_name}: {e}", summary_logger)
                    log_and_print(f"Log file for {gene_symbol} in {tissue_name} saved to: {log_file}", summary_logger)
                    log_and_print(f"Runtime for {gene_symbol} in {tissue_name} (failed): {tissue_runtime:.2f} seconds ({tissue_runtime/60:.2f} minutes)", summary_logger)
                    continue
            
            # Gene-level summary
            gene_end_time = time.time()
            gene_runtime = gene_end_time - gene_start_time
            
            if not all_results:
                log_and_print(f"No successful analyses completed for {gene_symbol}.", summary_logger)
                log_and_print(f"Runtime for {gene_symbol}: {gene_runtime:.2f} seconds ({gene_runtime/60:.2f} minutes)", summary_logger)
                continue
                
            # Store results for this gene
            all_genes_results[gene_symbol] = {
                'results': all_results,
                'gene_id': gene_id,
                'tissue_runtimes': tissue_runtimes,
                'gene_runtime': gene_runtime
            }
            
            # Save combined results for this gene
            combined_results = pd.concat(all_results.values(), ignore_index=False)
            combined_output_file = output_dir / f"{gene_symbol}_all_tissues_correlation_results.pkl"
            combined_results.to_pickle(combined_output_file)
            combined_csv_file = output_dir / f"{gene_symbol}_all_tissues_correlation_results.csv"
            combined_results.to_csv(combined_csv_file)
            
            log_and_print(f"Combined results for {gene_symbol} saved to: {combined_output_file}", summary_logger)
            log_and_print(f"Combined results (CSV) for {gene_symbol} saved to: {combined_csv_file}", summary_logger)
            log_and_print(f"Runtime for {gene_symbol}: {gene_runtime:.2f} seconds ({gene_runtime/60:.2f} minutes)", summary_logger)
        
        total_end_time = time.time()
        total_runtime = total_end_time - total_start_time
        
        if not all_genes_results:
            log_and_print("No successful analyses completed for any gene.", summary_logger)
            return
        
        # Create overall summary
        log_and_print(f"\n{'='*100}", summary_logger)
        log_and_print("OVERALL RESULTS SUMMARY", summary_logger)
        log_and_print(f"{'='*100}", summary_logger)
        log_and_print(f"Gene symbols processed: {', '.join(all_genes_results.keys())}", summary_logger)
        log_and_print(f"Total genes: {len(all_genes_results)}", summary_logger)
        log_and_print(f"Permutations: {args.permutations:,}", summary_logger)
        log_and_print(f"Tissues per gene: {len(expression_files)}", summary_logger)
        
        # Combine all results across genes
        all_combined_results = []
        for gene_symbol, gene_data in all_genes_results.items():
            gene_combined = pd.concat(gene_data['results'].values(), ignore_index=False)
            all_combined_results.append(gene_combined)
        
        mega_combined_results = pd.concat(all_combined_results, ignore_index=False)
        
        # Save mega combined results
        mega_output_file = output_dir / f"all_genes_all_tissues_correlation_results.pkl"
        mega_combined_results.to_pickle(mega_output_file)
        log_and_print(f"All genes combined results saved to: {mega_output_file}", summary_logger)
        
        # Also save as CSV for easy viewing
        mega_csv_file = output_dir / f"all_genes_all_tissues_correlation_results.csv"
        mega_combined_results.to_csv(mega_csv_file)
        log_and_print(f"All genes combined results (CSV) saved to: {mega_csv_file}", summary_logger)
        
        # List all log files created
        log_and_print(f"\nLog files created:", summary_logger)
        for gene_symbol in all_genes_results.keys():
            for tissue_name in [name for _, name in expression_files]:
                log_file = output_dir / f"{gene_symbol}_{tissue_name}.log"
                if log_file.exists():
                    log_and_print(f"  {gene_symbol} - {tissue_name}: {log_file}", summary_logger)
        
        # Show summary statistics across all genes and tissues
        successful_analyses = mega_combined_results[mega_combined_results['status'] == 'success']
        if len(successful_analyses) > 0:
            log_and_print(f"\nTotal successful analyses across all genes and tissues: {len(successful_analyses)}", summary_logger)
            
            log_and_print(f"\n{'='*100}", summary_logger)
            log_and_print("TOP CORRELATIONS ACROSS ALL GENES AND TISSUES (by absolute CCC value)", summary_logger)
            log_and_print(f"{'='*100}", summary_logger)
            
            # Sort by absolute CCC value (descending)
            top_results = successful_analyses.reindex(
                successful_analyses['ccc_value'].abs().sort_values(ascending=False).index
            )
            
            # Display top results
            log_and_print(f"{'Gene':<12} {'Tissue':<20} {'Metadata Column':<25} {'CCC Value':<12} {'P-value':<12} {'Significance':<15}", summary_logger)
            log_and_print("-" * 110, summary_logger)
            
            for idx, row in top_results.head(30).iterrows():
                gene = row['gene_symbol']
                tissue = row['tissue']
                ccc_val = row['ccc_value']
                p_val = row['p_value']
                
                # Determine significance
                if p_val < 0.001:
                    significance = "***"
                elif p_val < 0.01:
                    significance = "**"
                elif p_val < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                log_and_print(f"{gene:<12} {tissue:<20} {idx:<25} {ccc_val:>10.6f}  {p_val:>10.2e}  {significance:<15}", summary_logger)
            
            # Summary by gene
            log_and_print(f"\n{'='*100}", summary_logger)
            log_and_print("SUMMARY BY GENE", summary_logger)
            log_and_print(f"{'='*100}", summary_logger)
            
            for gene_symbol, gene_data in all_genes_results.items():
                gene_combined = pd.concat(gene_data['results'].values(), ignore_index=False)
                gene_successful = gene_combined[gene_combined['status'] == 'success']
                
                log_and_print(f"\nGene: {gene_symbol} (ID: {gene_data['gene_id']})", summary_logger)
                log_and_print(f"  Tissues processed: {len(gene_data['results'])}", summary_logger)
                log_and_print(f"  Successful analyses: {len(gene_successful)}", summary_logger)
                
                if len(gene_successful) > 0:
                    mean_ccc = gene_successful['ccc_value'].abs().mean()
                    max_ccc = gene_successful['ccc_value'].abs().max()
                    log_and_print(f"  Mean |CCC|: {mean_ccc:.6f}", summary_logger)
                    log_and_print(f"  Max |CCC|: {max_ccc:.6f}", summary_logger)
                    
                    # Top correlation for this gene
                    top_corr = gene_successful.reindex(gene_successful['ccc_value'].abs().sort_values(ascending=False).index).iloc[0]
                    log_and_print(f"  Top correlation: {top_corr.name} in {top_corr['tissue']} (CCC: {top_corr['ccc_value']:.6f}, p: {top_corr['p_value']:.2e})", summary_logger)
                
                log_and_print(f"  Runtime: {gene_data['gene_runtime']:.2f} seconds ({gene_data['gene_runtime']/60:.2f} minutes)", summary_logger)
            
            # Summary by tissue across all genes
            log_and_print(f"\n{'='*100}", summary_logger)
            log_and_print("SUMMARY BY TISSUE (across all genes)", summary_logger)
            log_and_print(f"{'='*100}", summary_logger)
            
            tissue_summary = {}
            for gene_symbol, gene_data in all_genes_results.items():
                for tissue_name, tissue_results in gene_data['results'].items():
                    if tissue_name not in tissue_summary:
                        tissue_summary[tissue_name] = []
                    tissue_summary[tissue_name].append(tissue_results)
            
            log_and_print(f"{'Tissue':<25} {'N Genes':<10} {'Successful':<12} {'Mean |CCC|':<12} {'Max |CCC|':<12}", summary_logger)
            log_and_print("-" * 75, summary_logger)
            
            for tissue_name in sorted(tissue_summary.keys()):
                tissue_all_genes = pd.concat(tissue_summary[tissue_name], ignore_index=False)
                tissue_successful = tissue_all_genes[tissue_all_genes['status'] == 'success']
                
                if len(tissue_successful) > 0:
                    mean_ccc = tissue_successful['ccc_value'].abs().mean()
                    max_ccc = tissue_successful['ccc_value'].abs().max()
                    log_and_print(f"{tissue_name:<25} {len(tissue_summary[tissue_name]):<10} {len(tissue_successful):<12} {mean_ccc:<12.6f} {max_ccc:<12.6f}", summary_logger)
                else:
                    log_and_print(f"{tissue_name:<25} {len(tissue_summary[tissue_name]):<10} {'0':<12} {'N/A':<12} {'N/A':<12}", summary_logger)
        
        # Runtime summary
        log_and_print(f"\n{'='*100}", summary_logger)
        log_and_print("RUNTIME SUMMARY", summary_logger)
        log_and_print(f"{'='*100}", summary_logger)
        log_and_print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)", summary_logger)
        log_and_print(f"Average runtime per gene: {total_runtime/len(args.gene_symbols):.2f} seconds", summary_logger)
        log_and_print(f"Total gene-tissue combinations: {len(args.gene_symbols) * len(expression_files)}", summary_logger)
        
        # Runtime by gene
        log_and_print(f"\nRuntime by gene:", summary_logger)
        log_and_print(f"{'Gene':<15} {'Runtime (sec)':<15} {'Runtime (min)':<15} {'Tissues':<10} {'Successful':<12}", summary_logger)
        log_and_print("-" * 75, summary_logger)
        
        for gene_symbol, gene_data in all_genes_results.items():
            successful_tissues = len(gene_data['results'])
            log_and_print(f"{gene_symbol:<15} {gene_data['gene_runtime']:<15.2f} {gene_data['gene_runtime']/60:<15.2f} {len(expression_files):<10} {successful_tissues:<12}", summary_logger)
        
        # Aggregate tissue runtime statistics across all genes
        all_tissue_runtimes = {}
        for gene_symbol, gene_data in all_genes_results.items():
            for tissue_name, runtime in gene_data['tissue_runtimes'].items():
                if tissue_name not in all_tissue_runtimes:
                    all_tissue_runtimes[tissue_name] = []
                all_tissue_runtimes[tissue_name].append(runtime)
        
        if all_tissue_runtimes:
            log_and_print(f"\nAverage runtime by tissue (across all genes):", summary_logger)
            log_and_print(f"{'Tissue':<25} {'Avg Runtime (sec)':<18} {'Avg Runtime (min)':<18} {'N Runs':<8} {'Min':<10} {'Max':<10}", summary_logger)
            log_and_print("-" * 95, summary_logger)
            
            tissue_avg_runtimes = []
            for tissue_name in sorted(all_tissue_runtimes.keys()):
                runtimes = all_tissue_runtimes[tissue_name]
                avg_runtime = np.mean(runtimes)
                min_runtime = np.min(runtimes)
                max_runtime = np.max(runtimes)
                tissue_avg_runtimes.append((tissue_name, avg_runtime))
                
                log_and_print(f"{tissue_name:<25} {avg_runtime:<18.2f} {avg_runtime/60:<18.2f} {len(runtimes):<8} {min_runtime:<10.2f} {max_runtime:<10.2f}", summary_logger)
            
            # Find fastest and slowest tissues (by average)
            tissue_avg_runtimes.sort(key=lambda x: x[1])
            fastest_tissue = tissue_avg_runtimes[0]
            slowest_tissue = tissue_avg_runtimes[-1]
            
            log_and_print(f"\nFastest tissue (avg): {fastest_tissue[0]} ({fastest_tissue[1]:.2f} seconds)", summary_logger)
            log_and_print(f"Slowest tissue (avg): {slowest_tissue[0]} ({slowest_tissue[1]:.2f} seconds)", summary_logger)
            log_and_print(f"Speed ratio: {slowest_tissue[1]/fastest_tissue[1]:.1f}x", summary_logger)
        
        # Final message about summary log
        log_and_print(f"\nSummary log saved to: {summary_log_file}", summary_logger)
        
        # Close the summary logger
        for handler in summary_logger.handlers:
            handler.close()
            summary_logger.removeHandler(handler)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 