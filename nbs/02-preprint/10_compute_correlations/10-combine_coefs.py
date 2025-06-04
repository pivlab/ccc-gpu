#!/usr/bin/env python
# coding: utf-8

"""
Script to combine correlation coefficients from different methods for a specific GTEx tissue.
It combines all coefficient values in one tissue into a single dataframe for easier processing later.
"""

import pandas as pd
from pathlib import Path
import argparse
import logging
from datetime import datetime
import sys
import concurrent.futures
from typing import List
from tqdm import tqdm

from ccc.utils import get_upper_triag


# Configure logging
def setup_logging(log_dir: Path = None) -> None:
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
        log_file = log_dir / f"combine_coefs_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Log file created at: {log_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Combine correlation coefficients from different methods for a specific GTEx tissue"
    )
    parser.add_argument(
        "--tissue",
        type=str,
        required=True,
        help="GTEx tissue to process (e.g., 'whole_blood')",
    )
    parser.add_argument(
        "--gene-selection",
        type=str,
        default="var_pc_log2",
        help="Gene selection strategy (default: var_pc_log2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of genes to process in each batch (default: 1000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )
    return parser.parse_args()


# Configuration constants
TOP_N_GENES = "all"
DATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
GENE_SELECTION_DIR = DATA_DIR / "gene_selection" / TOP_N_GENES
SIMILARITY_MATRICES_DIR = DATA_DIR / "similarity_matrices" / TOP_N_GENES


def process_batch(
    batch_genes: List[str],
    clustermatch_df: pd.DataFrame,
    pearson_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
) -> pd.DataFrame:
    """Process a batch of genes and combine their correlation coefficients.

    Args:
        batch_genes: List of genes to process
        clustermatch_df: CCC-GPU correlation matrix
        pearson_df: Pearson correlation matrix
        spearman_df: Spearman correlation matrix

    Returns:
        DataFrame containing combined coefficients for the batch
    """
    # Get upper triangle and unstack for the batch
    clustermatch_batch = get_upper_triag(clustermatch_df.loc[batch_genes, batch_genes])
    clustermatch_batch = clustermatch_batch.unstack().rename_axis((None, None))

    pearson_batch = get_upper_triag(pearson_df.loc[batch_genes, batch_genes])
    pearson_batch = pearson_batch.unstack().rename_axis((None, None)).abs()

    spearman_batch = get_upper_triag(spearman_df.loc[batch_genes, batch_genes])
    spearman_batch = spearman_batch.unstack().rename_axis((None, None)).abs()

    # Combine methods
    return pd.DataFrame(
        {
            "ccc": clustermatch_batch,
            "pearson": pearson_batch,
            "spearman": spearman_batch,
        }
    ).sort_index()


def save_batch(
    batch_df: pd.DataFrame,
    output_dir: Path,
    batch_num: int,
) -> Path:
    """Save a batch of results to a separate file.

    Args:
        batch_df: DataFrame containing the batch results
        output_dir: Directory to save the batch files
        batch_num: Batch number for the filename

    Returns:
        Path to the saved batch file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save batch to a numbered file
    batch_file = output_dir / f"batch_{batch_num:04d}.pkl"
    batch_df.to_pickle(batch_file)
    logging.info(f"Saved batch {batch_num} with shape {batch_df.shape} to {batch_file}")
    logging.info(f"Batch file exists: {batch_file.exists()}")

    return batch_file


def combine_batch_files(batch_files: List[Path], output_file: Path) -> None:
    """Combine all batch files into a single output file.

    Args:
        batch_files: List of batch file paths in order
        output_file: Path to save the combined results
    """
    logging.info("Combining batch files...")

    # Load and concatenate all batches in order
    dfs = []
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        df = pd.read_pickle(batch_file)
        dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dfs)
    logging.info(f"Combined shape: {combined_df.shape}")

    # Save combined results
    combined_df.to_pickle(output_file)
    logging.info(f"Saved combined results to {output_file}")

    # Clean up batch files
    for batch_file in batch_files:
        batch_file.unlink()
    logging.info("Cleaned up batch files")


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup logging
        setup_logging(Path("logs"))
        logging.info(f"Starting to combine coefficients for tissue '{args.tissue}'")

        # Set up file paths
        input_gene_expr_data_file = (
            GENE_SELECTION_DIR / f"gtex_v8_data_{args.tissue}-{args.gene_selection}.pkl"
        )
        logging.info(f"Input gene expression file: {input_gene_expr_data_file}")

        if not input_gene_expr_data_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_gene_expr_data_file}"
            )

        similarity_matrix_filename_template = (
            "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"
        )
        input_corr_file_template = Path(
            SIMILARITY_MATRICES_DIR / similarity_matrix_filename_template
        )

        output_file = SIMILARITY_MATRICES_DIR / str(input_corr_file_template).format(
            tissue=args.tissue,
            gene_sel_strategy=args.gene_selection,
            corr_method="all",
        )
        logging.info(f"Final output will be saved to: {output_file}")

        # Create temporary directory for batch files
        temp_dir = output_file.parent / f"{output_file.stem}_batches"
        logging.info(f"Creating temporary directory for batch files: {temp_dir}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Temporary directory exists: {temp_dir.exists()}")
        logging.info(f"Temporary directory contents: {list(temp_dir.glob('*.pkl'))}")

        # Load gene mapping
        logging.info("Loading gene mapping...")
        gene_map = pd.read_pickle(DATA_DIR / "gtex_gene_id_symbol_mappings.pkl")
        gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

        # Load gene expression data
        logging.info("Loading gene expression data...")
        data = pd.read_pickle(input_gene_expr_data_file)
        logging.info(f"Gene expression data shape: {data.shape}")

        # Load correlation matrices
        logging.info("Loading correlation matrices...")

        # CCC-GPU
        clustermatch_df = pd.read_pickle(
            str(input_corr_file_template).format(
                tissue=args.tissue,
                gene_sel_strategy=args.gene_selection,
                corr_method="ccc_gpu",
            )
        )
        logging.info(f"CCC-GPU matrix shape: {clustermatch_df.shape}")

        # Pearson
        pearson_df = pd.read_pickle(
            str(input_corr_file_template).format(
                tissue=args.tissue,
                gene_sel_strategy=args.gene_selection,
                corr_method="pearson",
            )
        )
        logging.info(f"Pearson matrix shape: {pearson_df.shape}")

        # Spearman
        spearman_df = pd.read_pickle(
            str(input_corr_file_template).format(
                tissue=args.tissue,
                gene_sel_strategy=args.gene_selection,
                corr_method="spearman",
            )
        )
        logging.info(f"Spearman matrix shape: {spearman_df.shape}")

        # Validate data
        assert data.index.equals(
            clustermatch_df.index
        ), "Index mismatch in CCC-GPU matrix"
        assert data.index.equals(pearson_df.index), "Index mismatch in Pearson matrix"
        assert data.index.equals(spearman_df.index), "Index mismatch in Spearman matrix"

        # Get list of genes and create batches
        genes = list(data.index)
        n_genes = len(genes)
        n_batches = (n_genes + args.batch_size - 1) // args.batch_size
        gene_batches = [
            genes[i : i + args.batch_size] for i in range(0, n_genes, args.batch_size)
        ]

        logging.info(
            f"Processing {n_genes} genes in {n_batches} batches of size {args.batch_size}"
        )

        # Process batches in parallel
        batch_files = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            futures = []
            for i, batch_genes in enumerate(gene_batches):
                future = executor.submit(
                    process_batch, batch_genes, clustermatch_df, pearson_df, spearman_df
                )
                futures.append((i, future))

            # Save results as they complete
            for i, future in tqdm(futures, desc="Processing batches"):
                batch_df = future.result()
                batch_file = save_batch(batch_df, temp_dir, i)
                batch_files.append(batch_file)
                logging.info(f"Current batch files: {[f.name for f in batch_files]}")

        # List all batch files before combining
        logging.info(
            f"All batch files before combining: {[f.name for f in temp_dir.glob('*.pkl')]}"
        )

        # Combine all batch files
        combine_batch_files(batch_files, output_file)

        # Clean up temporary directory
        logging.info(f"Cleaning up temporary directory: {temp_dir}")
        temp_dir.rmdir()
        logging.info("✅ Done!")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
