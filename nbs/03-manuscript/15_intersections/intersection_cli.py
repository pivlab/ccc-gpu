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
from pathlib import Path
import argparse
import logging
from datetime import datetime
import sys
import re
import gc
import psutil
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

from ccc.plots import MyUpSet


# Global configuration
DEFAULT_DATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
DEFAULT_MANUSCRIPT_DIR = Path("/mnt/data/proj_data/ccc-gpu/")
DEFAULT_Q_DIFF = 0.30
DEFAULT_TOP_N_GENES = "all"
DEFAULT_GENE_SELECTION = "var_pc_log2"
DEFAULT_CHUNK_SIZE = 100_000_000  # Process 100M rows at a time

RESULTS_DIR = DEFAULT_MANUSCRIPT_DIR / "results"

# Global state - removed to reduce memory usage
# Will use local variables and chunked processing instead


def discover_tissues(
    data_dir: Path,
    gene_selection: str,
    top_n_genes: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Discover available tissues and filter them based on include/exclude patterns.

    Args:
        data_dir: Base data directory
        gene_selection: Gene selection strategy
        top_n_genes: Number of top genes
        include_patterns: List of patterns to include (fuzzy match)
        exclude_patterns: List of patterns to exclude (fuzzy match)

    Returns:
        List of tissue names
    """
    similarity_matrices_dir = data_dir / "similarity_matrices" / top_n_genes

    # Find all "all.pkl" files (combined correlation files)
    pattern = f"gtex_v8_data_*-{gene_selection}-all.pkl"
    all_files = list(similarity_matrices_dir.glob(pattern))

    # Extract tissue names from filenames
    tissues = []
    for file_path in all_files:
        # Extract tissue from filename: gtex_v8_data_{tissue}-{gene_selection}-all.pkl
        match = re.match(
            r"gtex_v8_data_(.+?)-" + re.escape(gene_selection) + r"-all\.pkl",
            file_path.name,
        )
        if match:
            tissues.append(match.group(1))

    tissues.sort()
    logging.info(f"Found {len(tissues)} available tissues: {tissues}")

    # Apply include filters
    if include_patterns:
        filtered_tissues = []
        for tissue in tissues:
            for pattern in include_patterns:
                if re.search(pattern.lower(), tissue.lower()):
                    filtered_tissues.append(tissue)
                    break
        tissues = filtered_tissues
        logging.info(f"After include filtering: {len(tissues)} tissues: {tissues}")

    # Apply exclude filters
    if exclude_patterns:
        filtered_tissues = []
        for tissue in tissues:
            excluded = False
            for pattern in exclude_patterns:
                if re.search(pattern.lower(), tissue.lower()):
                    excluded = True
                    break
            if not excluded:
                filtered_tissues.append(tissue)
        tissues = filtered_tissues
        logging.info(f"After exclude filtering: {len(tissues)} tissues: {tissues}")

    return tissues


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


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def calculate_quantiles(input_file: Path, q_diff: float) -> Dict[str, Tuple[float, float]]:
    """Calculate quantiles for correlation methods without loading full dataset.
    
    Args:
        input_file: Path to correlation data file
        q_diff: Quantile difference threshold
        
    Returns:
        Dictionary with quantile bounds for each method
    """
    logging.info(f"Calculating quantiles from {input_file.name}...")
    logging.info(f"Memory usage before quantile calculation: {get_memory_usage():.2f} GB")
    
    # Load data in chunks to calculate quantiles
    chunk_size = DEFAULT_CHUNK_SIZE
    quantiles = {}
    
    # First pass: collect samples for quantile estimation
    sample_size = min(1_000_000, chunk_size)  # Sample up to 1M rows for quantile estimation
    
    df_iter = pd.read_pickle(input_file)
    total_rows = len(df_iter)
    
    # Sample every nth row for quantile estimation
    step = max(1, total_rows // sample_size)
    sample_df = df_iter.iloc[::step].copy()
    
    logging.info(f"Using {len(sample_df)} samples (every {step}th row) for quantile estimation")
    
    for method in ['ccc', 'pearson', 'spearman']:
        quantiles[method] = tuple(sample_df[method].quantile([q_diff, 1 - q_diff]))
        logging.info(f"{method}: q={q_diff:.3f} -> {quantiles[method][0]:.6f}, q={1-q_diff:.3f} -> {quantiles[method][1]:.6f}")
    
    del df_iter, sample_df
    gc.collect()
    
    logging.info(f"Memory usage after quantile calculation: {get_memory_usage():.2f} GB")
    return quantiles


def count_intersections_chunked(input_file: Path, quantiles: Dict[str, Tuple[float, float]], 
                               chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[tuple, int]:
    """Count intersections using chunked processing to minimize memory usage.
    
    Args:
        input_file: Path to correlation data file
        quantiles: Pre-calculated quantile bounds for each method
        chunk_size: Size of chunks to process
        
    Returns:
        Dictionary mapping intersection patterns to counts
    """
    logging.info(f"Starting chunked intersection counting with chunk size: {chunk_size:,}")
    
    intersection_counts = defaultdict(int)
    
    # Load data in chunks
    df_full = pd.read_pickle(input_file)
    total_rows = len(df_full)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    logging.info(f"Processing {total_rows:,} rows in {num_chunks} chunks")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        
        logging.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}: rows {start_idx:,} to {end_idx:,}")
        logging.info(f"Memory usage: {get_memory_usage():.2f} GB")
        
        # Get chunk data
        chunk_df = df_full.iloc[start_idx:end_idx].copy()
        
        # Create boolean masks for this chunk
        masks = {}
        for method, (low_q, high_q) in quantiles.items():
            masks[f"{method}_higher"] = chunk_df[method] >= high_q
            masks[f"{method}_lower"] = chunk_df[method] <= low_q
        
        # Convert to DataFrame for easier processing
        masks_df = pd.DataFrame(masks)
        
        # Count intersections for this chunk
        # Create all possible intersection patterns
        for _, row in masks_df.iterrows():
            pattern = tuple(row.values)
            intersection_counts[pattern] += 1
        
        # Clean up chunk data
        del chunk_df, masks, masks_df
        gc.collect()
        
        if (chunk_idx + 1) % 10 == 0:  # Log every 10 chunks
            logging.info(f"Processed {chunk_idx + 1}/{num_chunks} chunks, found {len(intersection_counts)} intersection patterns")
    
    del df_full
    gc.collect()
    
    logging.info(f"Completed intersection counting. Found {len(intersection_counts)} unique intersection patterns")
    logging.info(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    return dict(intersection_counts)


def create_upset_data_from_counts(intersection_counts: Dict[tuple, int]) -> pd.Series:
    """Create UpSet plot data from intersection counts.
    
    Args:
        intersection_counts: Dictionary mapping intersection patterns to counts
        
    Returns:
        pandas Series suitable for UpSet plotting
    """
    # Define the column order (must match the boolean mask order)
    column_names = [
        "Pearson (low)",
        "Pearson (high)", 
        "Spearman (low)",
        "Spearman (high)",
        "CCC (low)",
        "CCC (high)"
    ]
    
    # Create MultiIndex from intersection patterns
    index_tuples = []
    values = []
    
    for pattern, count in intersection_counts.items():
        if len(pattern) == 6:  # Ensure we have the right number of boolean values
            index_tuples.append(pattern)
            values.append(count)
        else:
            logging.warning(f"Skipping pattern with wrong length: {pattern}")
    
    if not index_tuples:
        raise ValueError("No valid intersection patterns found")
    
    # Create MultiIndex
    multi_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=column_names
    )
    
    # Create Series
    upset_data = pd.Series(values, index=multi_index)
    upset_data = upset_data.sort_index()
    
    logging.info(f"Created UpSet data with {len(upset_data)} intersection patterns")
    return upset_data


def plot_intersections(
    upset_data: pd.Series,
    paths: Dict[str, Path],
    tissue: str,
    output_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    plot_types: str = "both",
) -> None:
    """Plot intersections between correlation methods using pre-computed intersection data.

    Generates UpSet plots based on the specified plot_types parameter.

    Args:
        upset_data: Pre-computed intersection counts as pandas Series
        paths: Dictionary of paths
        tissue: Tissue name
        output_file: Path to save the plot. If None, uses default location.
        log_dir: Directory to save a copy of the plot in logs. If None, no copy is saved.
        plot_types: Which plots to generate - "full", "trimmed", or "both" (default: "both")
    """
    logging.info(f"Memory usage before plotting: {get_memory_usage():.2f} GB")
    
    # Generate plots based on plot_types parameter
    if plot_types in ["full", "both"]:
        logging.info(f"Generating full UpSet plot for {tissue}...")
        # Generate FULL plot (all intersections)
        _plot_upset_figure(
            upset_data,
            paths=paths,
            tissue=tissue,
            plot_type="full",
            output_file=output_file,
            log_dir=log_dir,
        )

    if plot_types in ["trimmed", "both"]:
        logging.info(f"Generating trimmed UpSet plot for {tissue}...")
        # Define the order of subsets for trimmed plot
        ordered_subsets = [
            # full agreements on high:
            (False, False, False, False, False, True),
            # agreements on top
            (False, False, False, False, True, True),
            (False, False, False, False, False, True),
            (False, False, False, False, True, False),
            # agreements on bottom
            (False, True, True, False, False, False),
            (True, False, True, False, False, False),
            (True, True, False, False, False, False),
            # full agreements on low:
            (True, True, True, False, False, False),
            # disagreements
            #   ccc
            (False, True, False, False, False, True),
            (False, True, False, False, False, True),
            (True, False, False, False, False, True),
            (True, True, False, False, False, True),
            #   pearson
            (False, False, True, False, True, False),
            (True, False, False, False, True, False),
            (True, False, True, False, True, False),
            #   spearman
            (False, True, False, False, True, False),
        ]

        # Filter for trimmed plot - only include subsets that exist in the data
        available_subsets = set(upset_data.index.tolist())
        trimmed_subsets = [
            subset for subset in ordered_subsets if subset in available_subsets
        ]

        if trimmed_subsets:
            # Create trimmed dataset
            upset_data_trimmed = upset_data.loc[trimmed_subsets]

            # Generate TRIMMED plot (selected intersections only)
            _plot_upset_figure(
                upset_data_trimmed,
                paths=paths,
                tissue=tissue,
                plot_type="trimmed",
                output_file=output_file,
                log_dir=log_dir,
            )
        else:
            logging.warning("No trimmed subsets found in the data. Skipping trimmed plot.")


def _plot_upset_figure(
    data: pd.Series,
    paths: Dict[str, Path],
    tissue: str,
    plot_type: str,
    output_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """Helper function to create and save an UpSet plot.

    Args:
        data: Series with intersection data (MultiIndex with boolean values)
        paths: Dictionary of paths
        tissue: Tissue name for file naming
        plot_type: Either "full" or "trimmed" for naming files
        output_file: Base output file path
        log_dir: Directory to save logs
    """
    logging.info(f"Creating {plot_type} UpSet plot with {len(data)} intersections")
    
    # Create figure
    fig = plt.figure(figsize=(14, 5))

    # Plot using custom UpSet class
    g = MyUpSet(
        data,
        show_counts=True,
        sort_categories_by=None,
        sort_by=None,
        show_percentages=True,
        element_size=None,
    ).plot(fig)

    # Remove totals
    if "totals" in g:
        g["totals"].remove()

    # Determine output file path
    if output_file is None:
        output_file = paths["figure_dir"] / f"upsetplot-{tissue}-{plot_type}.svg"
    else:
        # Insert tissue and plot type into the filename
        output_file = (
            output_file.parent
            / f"{output_file.stem}-{tissue}-{plot_type}{output_file.suffix}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_file,
        bbox_inches="tight",
        facecolor="white",
    )
    logging.info(f"Saved {plot_type} intersection plot to: {output_file}")

    # Save a copy in logs directory if provided
    if log_dir:
        log_plot_file = log_dir / f"intersection_plot-{tissue}-{plot_type}.svg"
        plt.savefig(
            log_plot_file,
            bbox_inches="tight",
            facecolor="white",
        )
        logging.info(f"Saved {plot_type} intersection plot copy to: {log_plot_file}")

    # Close the figure to free memory
    plt.close(fig)
    gc.collect()  # Force garbage collection after plotting


def save_intersections(
    upset_data: pd.Series,
    tissue: str,
    gene_selection: str,
    output_file: Optional[Path] = None,
) -> None:
    """Save intersection data to pickle file.

    Args:
        upset_data: Pre-computed intersection counts
        tissue: GTEx tissue to analyze
        gene_selection: Gene selection strategy
        output_file: Path to save the data. If None, uses default location.
    """
    if output_file is None:
        output_file = (
            RESULTS_DIR
            / f"gene_pair_intersections-gtex_v8-{tissue}-{gene_selection}.pkl"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    upset_data.to_pickle(output_file)
    logging.info(f"Saved intersection data to: {output_file}")


def setup_logging(log_dir: Path = None) -> Path:
    """Configure logging to write to both file and stdout.

    Args:
        log_dir: Directory to store log files. If None, logs will only go to stdout.

    Returns:
        Path to the timestamped log directory, or None if log_dir is None.
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
    timestamped_log_dir = None
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_log_dir = log_dir / timestamp
        timestamped_log_dir.mkdir(parents=True, exist_ok=True)

        log_file = timestamped_log_dir / "intersections.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Log file created at: {log_file}")

    return timestamped_log_dir


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze intersections between correlation coefficients"
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        help="Patterns to include tissues (fuzzy match). If not specified, all tissues are included.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        help="Patterns to exclude tissues (fuzzy match).",
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
    parser.add_argument(
        "--upset-plot",
        type=str,
        choices=["full", "trimmed", "both"],
        default="both",
        help="Which UpSet plot(s) to generate: 'full' (all intersections), 'trimmed' (selected intersections), or 'both' (default: both)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Number of rows to process at once (default: {DEFAULT_CHUNK_SIZE:,})",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup logging
        log_dir = Path("logs")
        timestamped_log_dir = setup_logging(log_dir)

        # Determine data directory
        data_dir = args.data_dir or DEFAULT_DATA_DIR

        # Discover tissues to process
        tissues = discover_tissues(
            data_dir=data_dir,
            gene_selection=args.gene_selection,
            top_n_genes=DEFAULT_TOP_N_GENES,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )

        if not tissues:
            logging.error("No tissues found matching the specified criteria.")
            return

        logging.info(
            f"Starting intersection analysis for {len(tissues)} tissues: {tissues}"
        )

        # Process each tissue
        for i, tissue in enumerate(tissues, 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing tissue {i}/{len(tissues)}: {tissue}")
            logging.info(f"{'='*60}")

            try:
                # Setup paths for this tissue
                paths = setup_paths(
                    tissue=tissue,
                    gene_selection=args.gene_selection,
                    top_n_genes=DEFAULT_TOP_N_GENES,
                    data_dir=data_dir,
                    manuscript_dir=args.manuscript_dir,
                )

                # Get input file path
                input_corr_file = paths["similarity_matrices_dir"] / str(
                    paths["input_corr_file_template"]
                ).format(
                    tissue=tissue,
                    gene_sel_strategy=args.gene_selection,
                    corr_method="all",
                )
                
                if not input_corr_file.exists():
                    raise FileNotFoundError(f"Input file not found: {input_corr_file}")
                
                logging.info(f"Processing file: {input_corr_file.name}")
                logging.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
                
                # Calculate quantiles efficiently
                quantiles = calculate_quantiles(input_corr_file, args.q_diff)
                
                # Count intersections using chunked processing
                intersection_counts = count_intersections_chunked(input_corr_file, quantiles)
                
                # Create UpSet data from counts
                upset_data = create_upset_data_from_counts(intersection_counts)
                
                # Save results for this tissue
                if args.output_dir:
                    output_file = args.output_dir / f"intersections_{tissue}.pkl"
                    save_intersections(upset_data, tissue, args.gene_selection, output_file)
                else:
                    save_intersections(upset_data, tissue, args.gene_selection)

                # Create UpSet plots for this tissue based on user selection
                logging.info(f"Generating UpSet plots ({args.upset_plot}) for {tissue}...")
                plot_intersections(upset_data, paths, tissue, log_dir=timestamped_log_dir, plot_types=args.upset_plot)
                
                # Clean up
                del intersection_counts, upset_data
                gc.collect()

                logging.info(f"✅ Completed analysis for {tissue}")

            except Exception as e:
                logging.error(f"❌ Error processing tissue '{tissue}': {str(e)}")
                # Continue with other tissues instead of failing completely
                continue

        logging.info(
            f"\n🎉 Done! Processed {len(tissues)} tissues with {args.upset_plot} intersection plots."
        )

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
