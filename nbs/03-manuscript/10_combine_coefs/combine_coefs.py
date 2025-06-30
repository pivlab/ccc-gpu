#!/usr/bin/env python3
"""
Combine Coefficients CLI Tool

This tool combines correlation coefficients from different methods (CCC-GPU, Pearson, Spearman)
into a single dataframe for a specified tissue type and gene selection strategy.

Features:
- Memory optimization: Automatically converts correlation matrices from float32 to float16 for ~50% memory reduction
- Comprehensive logging with detailed memory usage tracking
- Robust error handling and input validation
- Configurable parameters via command-line arguments

Author: Generated from 10-combine_coefs.ipynb
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Import CCC utilities
try:
    from ccc import conf
    from ccc.utils import get_upper_triag
except ImportError as e:
    print(f"Error importing CCC modules: {e}")
    print("Please ensure the CCC package is properly installed")
    sys.exit(1)


@dataclass
class Config:
    """Configuration class for the combine coefficients tool."""

    top_n_genes: str = "all"
    data_dir: Path = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
    gtex_tissue: str = "whole_blood"
    gene_selection_strategy: str = "var_pc_log2"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")  # Will be resolved relative to script location
    temp_dir: Path = (
        None  # Will be auto-generated in similarity_matrices folder if None
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.data_dir = Path(self.data_dir)

        # Make log_dir relative to script location if it's a relative path
        if not self.log_dir.is_absolute():
            script_dir = Path(__file__).parent
            self.log_dir = script_dir / self.log_dir
        else:
            self.log_dir = Path(self.log_dir)

        # Set up temp_dir: if None, create in similarity_matrices folder with pattern
        if self.temp_dir is None:
            # Create cache directory in similarity_matrices folder
            cache_dir_name = (
                f"gtex_v8_data_{self.gtex_tissue}-{self.gene_selection_strategy}-cache"
            )
            self.temp_dir = (
                self.data_dir
                / "similarity_matrices"
                / self.top_n_genes
                / cache_dir_name
            )
        else:
            # Handle user-provided temp_dir (make relative to script location if needed)
            if not self.temp_dir.is_absolute():
                script_dir = Path(__file__).parent
                self.temp_dir = script_dir / self.temp_dir
            else:
                self.temp_dir = Path(self.temp_dir)

        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


class CorrelationProcessor:
    """Processor class for handling correlation matrix operations."""

    def __init__(self, config: Config):
        """
        Initialize the correlation processor.

        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set up paths
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Set up input and output file paths based on configuration."""
        self.gene_selection_dir = (
            self.config.data_dir / "gene_selection" / self.config.top_n_genes
        )
        self.similarity_matrices_dir = (
            self.config.data_dir / "similarity_matrices" / self.config.top_n_genes
        )

        # Input gene expression data file
        self.input_gene_expr_file = (
            self.gene_selection_dir
            / f"gtex_v8_data_{self.config.gtex_tissue}-{self.config.gene_selection_strategy}.pkl"
        )

        # Template for correlation matrix files
        self.similarity_matrix_template = (
            self.similarity_matrices_dir
            / "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"
        )

        # Output file
        self.output_file = Path(
            str(self.similarity_matrix_template).format(
                tissue=self.config.gtex_tissue,
                gene_sel_strategy=self.config.gene_selection_strategy,
                corr_method="all",
            )
        )

        self.logger.info(f"Input gene expression file: {self.input_gene_expr_file}")
        self.logger.info(f"Output file: {self.output_file}")
        self.logger.info(f"Temporary directory: {self.config.temp_dir}")

    def _get_temp_file_path(self, method: str) -> Path:
        """
        Generate temporary file path for aligned correlation matrix cache.

        Args:
            method: Correlation method ('ccc_gpu', 'pearson', 'spearman')

        Returns:
            Path to temporary file
        """
        temp_filename = f"aligned_{method}_{self.config.gtex_tissue}_{self.config.gene_selection_strategy}.pkl"
        return self.config.temp_dir / temp_filename

    def _save_aligned_matrix(self, series: pd.Series, method: str) -> None:
        """
        Save aligned correlation matrix to temporary file.

        Args:
            series: Aligned correlation series
            method: Correlation method name
        """
        temp_file = self._get_temp_file_path(method)

        # Log detailed information about the matrix being cached
        self.logger.info(f"Caching aligned {method} matrix:")
        self.logger.info(f"  Shape: {series.shape}")
        self.logger.info(f"  Data type: {series.dtype}")
        self.logger.info(
            f"  Memory usage: {series.memory_usage(deep=True) / 1024**2:.2f} MB"
        )

        # Log basic statistics
        self.logger.info(f"  Value range: [{series.min():.6f}, {series.max():.6f}]")
        self.logger.info(f"  Mean: {series.mean():.6f}, Std: {series.std():.6f}")

        # Log first few values
        self.logger.info("  First 5 values:")
        for i, (idx, val) in enumerate(series.head(5).items()):
            self.logger.info(f"    {idx}: {val:.6f}")

        # Log index information
        self.logger.info(f"  Index type: {type(series.index).__name__}")
        if hasattr(series.index, "names") and series.index.names:
            self.logger.info(f"  Index names: {series.index.names}")

        # Save to cache
        series.to_pickle(temp_file)

        # Log cache file information
        cache_file_size = temp_file.stat().st_size / 1024**2
        self.logger.info(f"  Cached to: {temp_file}")
        self.logger.info(f"  Cache file size: {cache_file_size:.2f} MB")

    def _load_aligned_matrix(self, method: str) -> pd.Series:
        """
        Load aligned correlation matrix from temporary file.

        Args:
            method: Correlation method name

        Returns:
            Aligned correlation series
        """
        temp_file = self._get_temp_file_path(method)

        # Log cache file information
        cache_file_size = temp_file.stat().st_size / 1024**2
        self.logger.info(f"Loading aligned {method} matrix from cache:")
        self.logger.info(f"  Cache file: {temp_file}")
        self.logger.info(f"  Cache file size: {cache_file_size:.2f} MB")

        # Load the series
        series = pd.read_pickle(temp_file)

        # Log detailed information about the loaded matrix
        self.logger.info(f"  Loaded shape: {series.shape}")
        self.logger.info(f"  Data type: {series.dtype}")
        self.logger.info(
            f"  Memory usage: {series.memory_usage(deep=True) / 1024**2:.2f} MB"
        )

        # Log basic statistics
        self.logger.info(f"  Value range: [{series.min():.6f}, {series.max():.6f}]")
        self.logger.info(f"  Mean: {series.mean():.6f}, Std: {series.std():.6f}")

        # Verify data integrity
        null_count = series.isnull().sum()
        inf_count = np.isinf(series).sum()
        self.logger.info(f"  Data integrity: {null_count} nulls, {inf_count} infs")

        return series

    def _has_aligned_matrix(self, method: str) -> bool:
        """
        Check if aligned correlation matrix exists in cache.

        Args:
            method: Correlation method name

        Returns:
            True if cached aligned file exists
        """
        temp_file = self._get_temp_file_path(method)
        return temp_file.exists()

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files after successful completion."""
        temp_files_removed = 0
        for method in ["ccc_gpu", "pearson", "spearman"]:
            temp_file = self._get_temp_file_path(method)
            if temp_file.exists():
                temp_file.unlink()
                temp_files_removed += 1

        if temp_files_removed > 0:
            self.logger.info(f"Cleaned up {temp_files_removed} temporary files")

    def clear_cache(self) -> None:
        """Clear all cached temporary files."""
        temp_files_cleared = 0
        for method in ["ccc_gpu", "pearson", "spearman"]:
            temp_file = self._get_temp_file_path(method)
            if temp_file.exists():
                temp_file.unlink()
                temp_files_cleared += 1

        if temp_files_cleared > 0:
            self.logger.info(f"Cleared {temp_files_cleared} cached temporary files")
        else:
            self.logger.info("No cached files to clear")

    def validate_inputs(self) -> None:
        """Validate that all required input files exist."""
        if not self.input_gene_expr_file.exists():
            raise FileNotFoundError(
                f"Gene expression file not found: {self.input_gene_expr_file}"
            )

        # Check correlation matrix files
        for method in ["ccc_gpu", "pearson", "spearman"]:
            corr_file = Path(
                str(self.similarity_matrix_template).format(
                    tissue=self.config.gtex_tissue,
                    gene_sel_strategy=self.config.gene_selection_strategy,
                    corr_method=method,
                )
            )
            if not corr_file.exists():
                raise FileNotFoundError(
                    f"Correlation matrix file not found: {corr_file}"
                )

        self.logger.info("All input files validated successfully")

    def load_gene_mapping(self) -> Dict[str, str]:
        """
        Load gene Ensembl ID to Symbol mapping.

        Returns:
            Dictionary mapping gene Ensembl IDs to gene symbols
        """
        self.logger.info("Loading gene mapping...")
        gene_map_file = self.config.data_dir / "gtex_gene_id_symbol_mappings.pkl"

        if not gene_map_file.exists():
            raise FileNotFoundError(f"Gene mapping file not found: {gene_map_file}")

        gene_map_df = pd.read_pickle(gene_map_file)
        gene_map = gene_map_df.set_index("gene_ens_id")["gene_symbol"].to_dict()

        # Validation check
        if "ENSG00000145309.5" in gene_map:
            expected_symbol = "CABS1"
            if gene_map["ENSG00000145309.5"] != expected_symbol:
                self.logger.warning(
                    f"Gene mapping validation failed: expected {expected_symbol}, "
                    f"got {gene_map['ENSG00000145309.5']}"
                )

        self.logger.info(f"Loaded {len(gene_map)} gene mappings")
        return gene_map

    def load_gene_expression_data(self) -> pd.DataFrame:
        """
        Load gene expression data.

        Returns:
            DataFrame containing gene expression data
        """
        self.logger.info(
            f"Loading gene expression data from {self.input_gene_expr_file}"
        )
        data = pd.read_pickle(self.input_gene_expr_file)
        self.logger.info(f"Loaded gene expression data with shape: {data.shape}")
        return data

    def load_correlation_matrix(
        self, method: str, reference_index: pd.Index
    ) -> pd.DataFrame:
        """
        Load and validate a correlation matrix for a specific method.
        Converts data from float32 to float16 for memory optimization.

        Args:
            method: Correlation method ('ccc_gpu', 'pearson', 'spearman')
            reference_index: Reference index to validate against

        Returns:
            DataFrame containing the correlation matrix in float16 format
        """
        corr_file = Path(
            str(self.similarity_matrix_template).format(
                tissue=self.config.gtex_tissue,
                gene_sel_strategy=self.config.gene_selection_strategy,
                corr_method=method,
            )
        )

        self.logger.info(f"Loading {method} correlation matrix from {corr_file}")
        corr_df = pd.read_pickle(corr_file)

        # Log original data type and memory usage
        original_dtype = (
            corr_df.dtypes.iloc[0] if len(corr_df.dtypes.unique()) == 1 else "mixed"
        )
        original_memory_mb = corr_df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Convert to float16 for memory optimization
        self.logger.info(
            f"Converting {method} matrix from {original_dtype} to float16 for memory optimization"
        )
        corr_df = corr_df.astype(np.float16)

        # Log memory savings
        new_memory_mb = corr_df.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_saved_mb = original_memory_mb - new_memory_mb
        memory_saved_pct = (
            (memory_saved_mb / original_memory_mb) * 100
            if original_memory_mb > 0
            else 0
        )
        self.logger.info(
            f"Memory usage: {original_memory_mb:.1f}MB → {new_memory_mb:.1f}MB "
            f"(saved {memory_saved_mb:.1f}MB, {memory_saved_pct:.1f}%)"
        )

        # Validate index consistency
        if not reference_index.equals(corr_df.index):
            self.logger.warning(f"Index mismatch for {method} correlation matrix")
            # Align indices if possible
            corr_df = corr_df.loc[
                reference_index, reference_index.intersection(corr_df.columns)
            ]

        self.logger.info(
            f"Loaded {method} correlation matrix with shape: {corr_df.shape}, dtype: {corr_df.dtypes.iloc[0]}"
        )
        return corr_df

    def process_correlation_matrix(
        self, corr_df: pd.DataFrame, method: str
    ) -> pd.Series:
        """
        Process correlation matrix by extracting upper triangular part and converting to series.
        Maintains float16 precision throughout processing.

        Args:
            corr_df: Correlation matrix DataFrame (float16)
            method: Correlation method name for logging

        Returns:
            Series containing processed correlation values (float16)
        """
        self.logger.info(
            f"Processing {method} correlation matrix (dtype: {corr_df.dtypes.iloc[0]})"
        )

        # Get upper triangular part (maintains float16)
        upper_triag = get_upper_triag(corr_df)

        # Convert to series and drop NaN values (maintains float16)
        series = upper_triag.unstack().rename_axis((None, None)).dropna()

        # Apply absolute value for Pearson and Spearman (ensure float16 is maintained)
        if method in ["pearson", "spearman"]:
            series = series.abs().astype(np.float16)
            self.logger.info(
                f"Applied absolute value to {method} correlations (maintaining float16)"
            )

        # Ensure final series is float16
        if series.dtype != np.float16:
            series = series.astype(np.float16)
            self.logger.info("Converted final series to float16")

        self.logger.info(
            f"Processed {method} correlation matrix to series with {len(series)} values (dtype: {series.dtype})"
        )
        return series

    def _get_or_process_aligned_matrix(
        self,
        method: str,
        common_indices: pd.Index = None,
        correlation_matrices: Dict[str, pd.DataFrame] = None,
        reference_index: pd.Index = None,
    ) -> pd.Series:
        """
        Get aligned correlation matrix from cache or process and align it.

        Args:
            method: Correlation method name
            common_indices: Common indices to align to (only needed if not cached)
            correlation_matrices: Dictionary of loaded correlation matrices (only needed if not cached)
            reference_index: Reference index for loading correlation matrix (only needed if not cached)

        Returns:
            Aligned correlation series (float16)
        """
        # Check for aligned cache first
        if self._has_aligned_matrix(method):
            self.logger.info(f"Found cached aligned {method} matrix - using directly")
            series = self._load_aligned_matrix(method)

            # Verify dtype and convert if necessary
            if series.dtype != np.float16:
                self.logger.info(
                    f"Converting cached aligned {method} matrix from {series.dtype} to float16"
                )
                series = series.astype(np.float16)

            return series

        # No cache found - process from scratch and align
        self.logger.info(
            f"No cache found for {method}, loading, processing and aligning matrix"
        )

        # Load correlation matrix if not already loaded
        if method not in correlation_matrices:
            correlation_matrices[method] = self.load_correlation_matrix(
                method, reference_index
            )

        # Process the matrix
        processed_series = self.process_correlation_matrix(
            correlation_matrices[method], method
        )

        # Log information about processed matrix before alignment
        self.logger.info(f"Processed {method} matrix before alignment:")
        self.logger.info(f"  Original shape: {processed_series.shape}")
        self.logger.info(f"  Data type: {processed_series.dtype}")
        self.logger.info(
            f"  Memory usage: {processed_series.memory_usage(deep=True) / 1024**2:.2f} MB"
        )

        # Align to common indices
        aligned_series = processed_series.loc[common_indices]

        # Log alignment information
        alignment_ratio = len(aligned_series) / len(processed_series) * 100
        self.logger.info("Alignment summary:")
        self.logger.info(f"  Aligned shape: {aligned_series.shape}")
        self.logger.info(f"  Alignment ratio: {alignment_ratio:.2f}% of original")
        self.logger.info(
            f"  Memory saved: {(processed_series.memory_usage(deep=True) - aligned_series.memory_usage(deep=True)) / 1024**2:.2f} MB"
        )

        # Cache the aligned result
        self._save_aligned_matrix(aligned_series, method)

        return aligned_series

    def combine_correlations(self) -> pd.DataFrame:
        """
        Combine all correlation methods into a single DataFrame.

        Returns:
            DataFrame with columns for each correlation method
        """
        self.logger.info("Starting correlation combination process")

        # Load gene expression data for index reference
        gene_expr_data = self.load_gene_expression_data()
        reference_index = gene_expr_data.index

        # Check cache status for all methods
        cached_methods = []
        uncached_methods = []

        for method in ["ccc_gpu", "pearson", "spearman"]:
            if self._has_aligned_matrix(method):
                cached_methods.append(method)
            else:
                uncached_methods.append(method)

        # Log cache status
        if cached_methods:
            self.logger.info(f"Found aligned cache for: {', '.join(cached_methods)}")
        if uncached_methods:
            self.logger.info(f"Need to load and process: {', '.join(uncached_methods)}")

        # Initialize empty correlation_matrices dict - matrices will be loaded on-demand
        correlation_matrices = {}

        # Process Spearman first to get common indices (Spearman provides the reference)
        if self._has_aligned_matrix("spearman"):
            # Load from cache
            spearman_series = self._get_or_process_aligned_matrix("spearman")
        else:
            # Need to process Spearman - it becomes the reference, so no alignment needed initially
            self.logger.info(
                "No cache found for spearman, loading and processing matrix"
            )
            correlation_matrices["spearman"] = self.load_correlation_matrix(
                "spearman", reference_index
            )
            spearman_series = self.process_correlation_matrix(
                correlation_matrices["spearman"], "spearman"
            )

            # Log information about processed Spearman (which becomes the reference)
            self.logger.info("Processed spearman matrix (reference):")
            self.logger.info(f"  Shape: {spearman_series.shape}")
            self.logger.info(f"  Data type: {spearman_series.dtype}")
            self.logger.info(
                f"  Memory usage: {spearman_series.memory_usage(deep=True) / 1024**2:.2f} MB"
            )
            self.logger.info("  Will be used as alignment reference for other methods")

            # Save as aligned cache (Spearman is the reference, so it's "aligned" to itself)
            self._save_aligned_matrix(spearman_series, "spearman")

        common_indices = spearman_series.index

        self.logger.info(
            f"Using Spearman indices as reference: {len(common_indices)} gene pairs"
        )

        # Build aligned correlations starting with Spearman
        aligned_correlations = {"spearman": spearman_series}

        # For other methods, get aligned matrices
        for method in ["ccc_gpu", "pearson"]:
            aligned_series = self._get_or_process_aligned_matrix(
                method, common_indices, correlation_matrices, reference_index
            )
            aligned_correlations[method] = aligned_series
            self.logger.info(
                f"Got aligned {method} series with {len(aligned_series)} values"
            )

        self.logger.info(f"Combining {len(common_indices)} gene pairs")

        # Create combined DataFrame with explicit float16 dtype
        combined_df = pd.DataFrame(
            {
                "ccc": aligned_correlations["ccc_gpu"].astype(np.float16),
                "pearson": aligned_correlations["pearson"].astype(np.float16),
                "spearman": aligned_correlations["spearman"].astype(np.float16),
            }
        )

        # Validate no NaN values
        # if combined_df.isna().any().any():
        #     raise ValueError("Combined DataFrame contains NaN values")

        # Log final DataFrame information including memory usage
        memory_usage_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.logger.info(
            f"Successfully combined correlations into DataFrame with shape: {combined_df.shape}"
        )
        self.logger.info(
            f"Final DataFrame dtype: {combined_df.dtypes.iloc[0]}, memory usage: {memory_usage_mb:.1f}MB"
        )

        return combined_df

    def save_results(self, combined_df: pd.DataFrame) -> None:
        """
        Save the combined correlation DataFrame to file.
        Logs detailed information about data types, memory usage, and file size.

        Args:
            combined_df: Combined correlation DataFrame to save (float16 optimized)
        """
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Log DataFrame information before saving
        memory_usage_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
        dtypes_info = ", ".join(
            [f"{col}: {dtype}" for col, dtype in combined_df.dtypes.items()]
        )

        self.logger.info(f"Saving combined correlations to {self.output_file}")
        self.logger.info(
            f"Saving DataFrame with shape: {combined_df.shape}, memory: {memory_usage_mb:.1f}MB"
        )
        self.logger.info(f"Data types: {dtypes_info}")

        # Save the DataFrame
        combined_df.to_pickle(self.output_file)

        # Log file size information
        file_size_mb = self.output_file.stat().st_size / (1024 * 1024)
        self.logger.info(f"Results saved successfully. File size: {file_size_mb:.1f}MB")
        self.logger.info(
            "Memory optimization: Using float16 reduces memory usage by ~50% compared to float32"
        )

    def run(self, cleanup_temp: bool = True) -> None:
        """
        Execute the complete correlation combination workflow.

        Args:
            cleanup_temp: Whether to clean up temporary files after successful completion
        """
        try:
            self.logger.info("Starting correlation combination workflow")

            # Validate inputs
            self.validate_inputs()

            # Load gene mapping (for completeness, though not directly used in processing)
            gene_mapping = self.load_gene_mapping()

            # Combine correlations
            combined_df = self.combine_correlations()

            # Save results
            self.save_results(combined_df)

            # Clean up temporary files if requested
            if cleanup_temp:
                self._cleanup_temp_files()

            self.logger.info("Correlation combination workflow completed successfully")

        except Exception as e:
            self.logger.error(f"Error in correlation combination workflow: {str(e)}")
            self.logger.info("Temporary files preserved for debugging/resumption")
            raise


def setup_logging(config: Config) -> None:
    """
    Set up logging configuration.

    Args:
        config: Configuration object containing logging parameters
    """
    # Create log file path
    log_file = (
        config.log_dir
        / f"combine_coefs_{config.gtex_tissue}_{config.gene_selection_strategy}.log"
    )

    # Ensure log directory exists (double check)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create file handler
    file_handler = logging.FileHandler(
        log_file, mode="w"
    )  # 'w' to overwrite existing log
    file_handler.setLevel(getattr(logging, config.log_level.upper()))
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level.upper()))
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.root.setLevel(getattr(logging, config.log_level.upper()))
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Test logging immediately
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log directory: {config.log_dir.absolute()}")
    logger.info(f"Log file: {log_file.absolute()}")

    # Force flush to ensure writes
    file_handler.flush()
    console_handler.flush()


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Combine correlation coefficients from different methods into a single dataframe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--top-n-genes", type=str, default="all", help="Number of top genes to process"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8"),
        help="Base data directory path",
    )

    parser.add_argument(
        "--gtex-tissue",
        type=str,
        default="whole_blood",
        help="GTEx tissue type to process",
    )

    parser.add_argument(
        "--gene-selection-strategy",
        type=str,
        default="var_pc_log2",
        help="Gene selection strategy",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for log files (relative to script location)",
    )

    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Directory for temporary cache files (default: auto-generated in similarity_matrices folder)",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary cache files after completion (useful for debugging)",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache files before processing",
    )

    return parser


def main() -> None:
    """Main entry point for the CLI application."""
    logger = None
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()

        # Create configuration
        config = Config(
            top_n_genes=args.top_n_genes,
            data_dir=args.data_dir,
            gtex_tissue=args.gtex_tissue,
            gene_selection_strategy=args.gene_selection_strategy,
            log_level=args.log_level,
            log_dir=args.log_dir,
            temp_dir=args.temp_dir,
        )

        # Setup logging
        setup_logging(config)

        logger = logging.getLogger(__name__)
        logger.info("Starting combine coefficients CLI tool")
        logger.info(f"Configuration: {config}")

        # Validate tissue parameter
        if not config.gtex_tissue:
            raise ValueError("GTEx tissue must be specified")

        # Create processor
        processor = CorrelationProcessor(config)

        # Clear cache if requested
        if args.clear_cache:
            logger.info("Clearing existing cache files")
            processor.clear_cache()

        # Run processor
        cleanup_temp = not args.no_cleanup
        processor.run(cleanup_temp=cleanup_temp)

        logger.info("CLI tool completed successfully")

    except KeyboardInterrupt:
        if logger:
            logger.info("Operation cancelled by user")
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.error(f"CLI tool failed with error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Ensure all log handlers are properly flushed and closed
        for handler in logging.root.handlers[:]:
            handler.flush()
            if hasattr(handler, "close"):
                handler.close()


if __name__ == "__main__":
    main()
