"""
Script to compute correlation matrices for GTEx v8 gene expression data using GPU-accelerated CCC.
This script processes tissue-specific gene expression data and computes correlation matrices
using the CCC (Canonical Correlation Coefficient) method with GPU acceleration.
"""

import logging
import pandas as pd
from time import time
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from datetime import datetime
import sys
import argparse
from ccc.corr import ccc_gpu, pearson, spearman

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_DIR = SCRIPT_DIR / "logs"  # Directory for log files


# Configure logging
def setup_logging(log_dir: Path = None, method_name: str = None) -> None:
    """Configure logging to write to both file and stdout.

    Args:
        log_dir: Directory to store log files. If None, logs will only go to stdout.
    """
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers = []

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_dir is provided
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"gtex-var_pc_log2-{method_name}.root.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Log file created at: {log_file}")


def get_correlation_method(method_name: str) -> Callable:
    """Get the correlation method function based on the method name.

    Args:
        method_name: Name of the correlation method ('ccc_gpu', 'pearson', or 'spearman')

    Returns:
        Callable: The correlation method function

    Raises:
        ValueError: If method_name is not one of the supported methods
    """
    method_map = {
        "ccc_gpu": lambda x: ccc_gpu(x, n_jobs=N_CPU_CORES),
        "pearson": pearson,
        "spearman": spearman,
    }

    if method_name not in method_map:
        raise ValueError(
            f"Unsupported correlation method: {method_name}. "
            f"Supported methods are: {', '.join(method_map.keys())}"
        )

    method = method_map[method_name]
    method.__name__ = method_name
    return method


# Configuration constants
# Data processing parameters
GENE_SELECTION_STRATEGY = "var_pc_log2"
TOP_N_GENES = "all"
N_CPU_CORES = 24

# File paths
DATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
INPUT_DIR = DATA_DIR / "gene_selection" / "all"
OUTPUT_DIR = DATA_DIR / "similarity_matrices" / TOP_N_GENES


def validate_input_data(data: pd.DataFrame) -> None:
    """Validate the input data format and content.

    Args:
        data: Input DataFrame containing gene expression data

    Raises:
        ValueError: If data validation fails
    """
    if data.empty:
        raise ValueError("Input data is empty")
    if data.isnull().any().any():
        raise ValueError("Input data contains null values")


def process_tissue_data(
    tissue_data_file: Path, correlation_method: Callable
) -> pd.DataFrame:
    """Process a single tissue data file and compute correlations.

    Args:
        tissue_data_file: Path to the tissue data file
        correlation_method: Function to compute correlations

    Returns:
        DataFrame containing the computed correlations
    """
    logging.info(f"Processing tissue data file: {tissue_data_file.stem}")

    # Read and validate data
    data = pd.read_pickle(tissue_data_file)
    validate_input_data(data)

    # Compute correlations
    start_time = time()
    data_corrs = correlation_method(data)
    data_corrs = data_corrs.astype("float32")  # Convert from double to float
    elapsed_time = time() - start_time

    logging.info(
        f"Computed correlations for {tissue_data_file.stem} in {elapsed_time:.2f} seconds"
    )
    return data_corrs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute correlation matrices for GTEx v8 gene expression data"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ccc_gpu", "pearson", "spearman"],
        default="ccc_gpu",
        help="Correlation method to use (default: ccc_gpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup logging
        setup_logging(LOG_DIR, args.method)
        logging.info(
            f"Starting GTEx correlation computation using {args.method} method"
        )

        # Get correlation method
        correlation_method = get_correlation_method(args.method)

        # Validate directories
        if not INPUT_DIR.exists():
            raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {OUTPUT_DIR}")

        # Get input files
        input_files = sorted(list(INPUT_DIR.glob(f"*-{GENE_SELECTION_STRATEGY}.pkl")))
        if not input_files:
            raise FileNotFoundError(f"No .pkl files found in {INPUT_DIR}")

        logging.info(f"Found {len(input_files)} input files")
        logging.info(f"Input files: {input_files}")

        # Process each tissue data file
        for tissue_data_file in tqdm(input_files, ncols=100, desc="Processing tissues"):
            try:
                # Compute correlations
                data_corrs = process_tissue_data(tissue_data_file, correlation_method)

                # Save results
                output_filename = (
                    f"{tissue_data_file.stem}-{correlation_method.__name__}.pkl"
                )
                output_path = OUTPUT_DIR / output_filename
                data_corrs.to_pickle(path=output_path)
                logging.info(f"Saved correlations to {output_path}\n")

            except Exception as e:
                logging.error(f"Error processing {tissue_data_file}: {str(e)}\n")
                continue

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
