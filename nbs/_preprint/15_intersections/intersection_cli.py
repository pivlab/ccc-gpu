#!/usr/bin/env python
# coding: utf-8

"""
Analyzes how correlation coefficients intersect on different gene pairs.
Takes the top gene pairs with maximum correlation coefficients according to Pearson,
Spearman and CCC, and the equivalent set with minimum coefficient values, then
compares how these sets intersect each other.

After identifying different intersection sets, plots gene pairs to visualize
what's being captured or not by each coefficient.
"""

import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_indicators
from pathlib import Path
import argparse
import logging
from datetime import datetime
import sys
from typing import Dict, Tuple, Optional

from ccc.plots import MyUpSet


# Global configuration
DEFAULT_DATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
DEFAULT_MANUSCRIPT_DIR = Path("/mnt/data/projs/manuscripts/ccc-gpu")
DEFAULT_Q_DIFF = 0.30
DEFAULT_TOP_N_GENES = "all"
DEFAULT_TISSUE = "whole_blood"
DEFAULT_GENE_SELECTION = "var_pc_log2"

RESULTS_DIR = DEFAULT_MANUSCRIPT_DIR / "results"

# Global state
df: Optional[pd.DataFrame] = None
df_plot: Optional[pd.DataFrame] = None


def setup_paths(
    tissue: str,
    gene_selection: str,
    top_n_genes: str,
    data_dir: Optional[Path] = None,
    manuscript_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Set up all necessary paths for the analysis.

    Args:
        tissue: GTEx tissue to analyze
        gene_selection: Gene selection strategy
        top_n_genes: Number of top genes to analyze
        data_dir: Base data directory
        manuscript_dir: Manuscript directory for saving figures

    Returns:
        Dictionary containing all necessary paths
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    manuscript_dir = manuscript_dir or DEFAULT_MANUSCRIPT_DIR

    paths = {
        "data_dir": data_dir,
        "gene_selection_dir": data_dir / "gene_selection" / top_n_genes,
        "similarity_matrices_dir": data_dir / "similarity_matrices" / top_n_genes,
        "manuscript_dir": manuscript_dir,
        "figure_dir": manuscript_dir / "figures" / "coefs_comp" / f"gtex_{tissue}",
    }

    # Create similarity matrix filename template
    paths["similarity_matrix_filename_template"] = (
        "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"
    )
    paths["input_corr_file_template"] = (
        paths["similarity_matrices_dir"] / paths["similarity_matrix_filename_template"]
    )

    return paths


def load_data(paths: Dict[str, Path], tissue: str, gene_selection: str) -> None:
    """Load correlation data from pickle files.

    Args:
        paths: Dictionary of paths
        tissue: GTEx tissue to analyze
        gene_selection: Gene selection strategy
    """
    global df

    input_corr_file = paths["similarity_matrices_dir"] / str(
        paths["input_corr_file_template"]
    ).format(
        tissue=tissue,
        gene_sel_strategy=gene_selection,
        corr_method="all",
    )

    if not input_corr_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_corr_file}")

    df = pd.read_pickle(input_corr_file)
    logging.info(f"Loaded correlation data with shape: {df.shape}")


def get_quantile_bounds(method_name: str, q_diff: float) -> Tuple[float, float]:
    """Get lower and upper quantile bounds for a correlation method.

    Args:
        method_name: Name of the correlation method
        q_diff: Quantile difference threshold

    Returns:
        Tuple of (lower_quantile, upper_quantile)
    """
    return tuple(df[method_name].quantile([q_diff, 1 - q_diff]))


def prepare_plot_data(q_diff: float) -> None:
    """Prepare data for plotting intersections.

    Args:
        q_diff: Quantile difference threshold
    """
    global df_plot

    # Get quantile bounds for each method
    clustermatch_lq, clustermatch_hq = get_quantile_bounds("ccc", q_diff)
    pearson_lq, pearson_hq = get_quantile_bounds("pearson", q_diff)
    spearman_lq, spearman_hq = get_quantile_bounds("spearman", q_diff)

    # Create boolean masks for high and low values
    masks = {
        "pearson_higher": df["pearson"] >= pearson_hq,
        "pearson_lower": df["pearson"] <= pearson_lq,
        "spearman_higher": df["spearman"] >= spearman_hq,
        "spearman_lower": df["spearman"] <= spearman_lq,
        "clustermatch_higher": df["ccc"] >= clustermatch_hq,
        "clustermatch_lower": df["ccc"] <= clustermatch_lq,
    }

    # Create plot DataFrame
    df_plot = pd.DataFrame(masks)
    df_plot = pd.concat([df_plot, df], axis=1)

    # Rename columns for plotting
    df_plot = df_plot.rename(
        columns={
            "pearson_higher": "Pearson (high)",
            "pearson_lower": "Pearson (low)",
            "spearman_higher": "Spearman (high)",
            "spearman_lower": "Spearman (low)",
            "clustermatch_higher": "CCC (high)",
            "clustermatch_lower": "CCC (low)",
        }
    )

    logging.info(f"Prepared plot data with shape: {df_plot.shape}")


def plot_intersections(
    paths: Dict[str, Path],
    output_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """Plot intersections between correlation methods.

    Args:
        paths: Dictionary of paths
        output_file: Path to save the plot. If None, uses default location.
        log_dir: Directory to save a copy of the plot in logs. If None, no copy is saved.
    """
    if df_plot is None:
        raise ValueError("Plot data not prepared. Call prepare_plot_data() first.")

    # Get categories for plotting
    categories = sorted(
        [x for x in df_plot.columns if " (" in x],
        reverse=True,
        key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],
    )

    # Create intersection data
    gene_pairs_by_cats = from_indicators(
        categories, data=df_plot.copy()
    )  # Use copy to avoid chained assignment

    # Sort by index
    gene_pairs_by_cats = gene_pairs_by_cats.sort_index()

    # Get unique index combinations
    # tmp_index = gene_pairs_by_cats.index.unique().to_frame(False)

    # Define the order of subsets based on the original notebook
    ordered_subsets = [
        # full agreements on high:
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
        (False, True, False, True, False, True),
        (False, True, False, False, False, True),
        (True, False, False, False, False, True),
        (True, True, False, False, False, True),
        #   pearson
        (False, False, True, False, True, False),
        (True, False, False, False, True, False),
        (True, False, True, False, True, False),
        #   spearman
        (False, True, False, True, False, False),
    ]

    # Reorder the subsets
    gene_pairs_by_cats = gene_pairs_by_cats.loc[ordered_subsets]

    # Rename columns and index names
    gene_pairs_by_cats = gene_pairs_by_cats.rename(
        columns={
            "Clustermatch (high)": "CCC (high)",
            "Clustermatch (low)": "CCC (low)",
        }
    )

    gene_pairs_by_cats.index.set_names(
        {
            "Clustermatch (high)": "CCC (high)",
            "Clustermatch (low)": "CCC (low)",
        },
        inplace=True,
    )

    # Create figure
    fig = plt.figure(figsize=(14, 5))

    # Plot using custom UpSet class
    g = MyUpSet(
        gene_pairs_by_cats,
        show_counts=True,
        sort_categories_by=None,
        sort_by=None,
        show_percentages=True,
        element_size=None,
    ).plot(fig)

    # Remove totals
    if "totals" in g:
        g["totals"].remove()

    # Save figure in default location
    if output_file is None:
        output_file = paths["figure_dir"] / "upsetplot.svg"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_file,
        bbox_inches="tight",
        facecolor="white",
    )
    logging.info(f"Saved intersection plot to: {output_file}")

    # Save a copy in logs directory if provided
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_plot_file = log_dir / f"intersection_plot_{timestamp}.svg"
        plt.savefig(
            log_plot_file,
            bbox_inches="tight",
            facecolor="white",
        )
        logging.info(f"Saved intersection plot copy to: {log_plot_file}")

    # Close the figure to free memory
    plt.close(fig)


def save_intersections(
    paths: Dict[str, Path],
    tissue: str,
    gene_selection: str,
    output_file: Optional[Path] = None,
) -> None:
    """Save intersection data to pickle file.

    Args:
        paths: Dictionary of paths
        tissue: GTEx tissue to analyze
        gene_selection: Gene selection strategy
        output_file: Path to save the data. If None, uses default location.
    """
    if df_plot is None:
        raise ValueError("Plot data not prepared. Call prepare_plot_data() first.")

    if output_file is None:
        output_file = (
            RESULTS_DIR
            / f"gene_pair_intersections-gtex_v8-{tissue}-{gene_selection}.pkl"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_plot.to_pickle(output_file)
    logging.info(f"Saved intersection data to: {output_file}")


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
        log_file = log_dir / f"intersections_{timestamp}.log"

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
        description="Analyze intersections between correlation coefficients"
    )
    parser.add_argument(
        "--tissue",
        type=str,
        default=DEFAULT_TISSUE,
        help=f"GTEx tissue to analyze (default: {DEFAULT_TISSUE})",
    )
    parser.add_argument(
        "--gene-selection",
        type=str,
        default=DEFAULT_GENE_SELECTION,
        help=f"Gene selection strategy (default: {DEFAULT_GENE_SELECTION})",
    )
    parser.add_argument(
        "--q-diff",
        type=float,
        default=DEFAULT_Q_DIFF,
        help=f"Quantile difference threshold (default: {DEFAULT_Q_DIFF})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Base data directory",
    )
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        help="Manuscript directory for saving figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for saving results",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup logging
        log_dir = Path("logs")
        setup_logging(log_dir)
        logging.info(f"Starting intersection analysis for tissue '{args.tissue}'")

        # Setup paths
        paths = setup_paths(
            tissue=args.tissue,
            gene_selection=args.gene_selection,
            top_n_genes=DEFAULT_TOP_N_GENES,
            data_dir=args.data_dir,
            manuscript_dir=args.manuscript_dir,
        )

        # Run analysis
        load_data(paths, args.tissue, args.gene_selection)
        prepare_plot_data(args.q_diff)

        # Save results
        if args.output_dir:
            output_file = args.output_dir / f"intersections_{args.tissue}.pkl"
            save_intersections(paths, args.tissue, args.gene_selection, output_file)
        else:
            save_intersections(paths, args.tissue, args.gene_selection)

        # Create plot
        plot_intersections(paths, log_dir=log_dir)

        logging.info("✅ Done!")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
