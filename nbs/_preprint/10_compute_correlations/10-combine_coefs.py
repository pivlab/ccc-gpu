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
    return parser.parse_args()


# Configuration constants
TOP_N_GENES = "all"
DATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
GENE_SELECTION_DIR = DATA_DIR / "gene_selection" / TOP_N_GENES
SIMILARITY_MATRICES_DIR = DATA_DIR / "similarity_matrices" / TOP_N_GENES


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
        logging.info(f"Output file will be: {output_file}")

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

        # Process matrices
        logging.info("Processing correlation matrices...")

        # Get upper triangle and unstack
        clustermatch_df = get_upper_triag(clustermatch_df)
        # clustermatch_df = clustermatch_df.unstack().rename_axis((None, None)).dropna()
        clustermatch_df = clustermatch_df.unstack().rename_axis((None, None))

        pearson_df = get_upper_triag(pearson_df)
        pearson_df = pearson_df.unstack().rename_axis((None, None))
        # Apply abs only to non-NaN values
        pearson_df = pearson_df.fillna(-100).abs().replace(-100, pd.NA)

        spearman_df = get_upper_triag(spearman_df)
        spearman_df = spearman_df.unstack().rename_axis((None, None))
        # Apply abs only to non-NaN values
        spearman_df = spearman_df.fillna(-100).abs().replace(-100, pd.NA)

        # Validate processed data
        assert clustermatch_df.index.equals(
            pearson_df.index
        ), "Index mismatch after processing"
        assert clustermatch_df.index.equals(
            spearman_df.index
        ), "Index mismatch after processing"

        # Combine all methods
        logging.info("Combining all methods...")
        df = pd.DataFrame(
            {
                "ccc": clustermatch_df,
                "pearson": pearson_df,
                "spearman": spearman_df,
            }
        ).sort_index()

        # Final validation
        # assert not df.isna().any().any(), "Found NaN values in final dataframe"
        logging.info(f"Final combined matrix shape: {df.shape}")

        # Save results
        logging.info(f"Saving combined coefficients to {output_file}")
        df.to_pickle(output_file)
        logging.info("Done!")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# In[ ]:
