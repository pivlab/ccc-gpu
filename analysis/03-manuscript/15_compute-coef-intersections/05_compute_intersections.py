#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from upsetplot import plot, from_indicators
from pathlib import Path
from ccc.plots import MyUpSet
from ccc import conf

import logging
from datetime import datetime
import shutil
import sys
import argparse
from typing import Dict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute gene pair intersections for correlation coefficients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--gtex-tissue",
        default="whole_blood",
        help="GTEx tissue to analyze"
    )
    
    parser.add_argument(
        "--gene-sel-strategy",
        default="",
        help="Gene selection strategy"
    )
    
    parser.add_argument(
        "--top-n-genes",
        default="all",
        help="Number of top genes to consider"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/pividori_lab/haoyu_projects/ccc-gpu/data/gtex/"),
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--q-diff",
        type=float,
        default=0.30,
        help="Threshold to compare coefficients (quantile difference from extremes)"
    )
    
    parser.add_argument(
        "--tissue-threshold-dir",
        type=Path,
        help="Directory containing tissue-specific threshold files (format: {tissue}-null_coefs_percentiles.pkl)"
    )
    
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=0.30,
        help="Percentile to use for low threshold from tissue threshold file (e.g., 0.90 for 90th percentile)"
    )
    
    parser.add_argument(
        "--high-percentile", 
        type=float,
        default=0.70,
        help="Percentile to use for high threshold from tissue threshold file (e.g., 0.95 for 95th percentile)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/pividori_lab/haoyu_projects/ccc-gpu/results/gene_pair_intersections"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="If set, load existing intersection.pkl file (if it exists) to generate plots instead of recomputing"
    )
    
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Log directory to use. If not provided, creates a new timestamped directory"
    )
    
    return parser.parse_args()


def generate_upset_plots(df_plot, categories, args, OUTPUT_FIGURE_NAME, LOG_DIR, logger):
    """Generate both full and trimmed upset plots from the intersection data."""
    
    # Generate full upset plot
    logger.info("Creating full upset plot...")
    df_r_data = df_plot
    gene_pairs_by_cats = from_indicators(categories, data=df_r_data)
    
    fig = plt.figure(figsize=(18, 5))
    g = plot(
        gene_pairs_by_cats,
        show_counts=True,
        sort_categories_by=None,
        element_size=None,
        fig=fig,
    )

    full_svg_path = args.output_dir / f"{OUTPUT_FIGURE_NAME}_full.svg"
    plt.savefig(
        full_svg_path,
        bbox_inches="tight",
        facecolor="white",
    )
    logger.info(f"Saved full upset plot to: {full_svg_path}")

    # Also save to log directory
    log_svg_path = LOG_DIR / f"{OUTPUT_FIGURE_NAME}_full.svg"
    shutil.copy2(full_svg_path, log_svg_path)
    logger.info(f"Copied full upset plot to log directory: {log_svg_path}")

    # Generate trimmed upset plot
    logger.info("Creating trimmed upset plot...")
    df_r_data = df_plot
    gene_pairs_by_cats = from_indicators(categories, data=df_r_data)
    gene_pairs_by_cats = gene_pairs_by_cats.sort_index()

    _tmp_index = gene_pairs_by_cats.index.unique().to_frame(False)
    logger.info("Unique index combinations:")
    logger.info(f"\n{_tmp_index}")

    combinations_with_3 = _tmp_index[_tmp_index.sum(axis=1) == 3]
    logger.info(f"Combinations with exactly 3 True values:")
    logger.info(f"\n{combinations_with_3}")

    first_3_zero = _tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)
    logger.info(f"Number of combinations where first 3 are all False: {first_3_zero.sum()}")

    # agreements on top
    agreements_top = _tmp_index.loc[
        _tmp_index[
            _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
            & _tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)
            & _tmp_index.apply(lambda x: 3 > x[3:].sum() > 1, axis=1)
        ].index
    ].apply(tuple, axis=1).to_numpy()
    logger.info(f"Agreements on top: {agreements_top}")

    # agreements on bottom
    agreements_bottom = _tmp_index.loc[
        _tmp_index[
            _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
            & _tmp_index.apply(lambda x: 3 > x[0:3].sum() > 1, axis=1)
            & _tmp_index.apply(lambda x: x[3:].sum() == 0, axis=1)
        ].index
    ].apply(tuple, axis=1).to_numpy()
    logger.info(f"Agreements on bottom: {agreements_bottom}")

    # diagreements
    disagreements = _tmp_index.loc[
        _tmp_index[
            _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
            & _tmp_index.apply(lambda x: x[0:3].sum() > 0, axis=1)
            & _tmp_index.apply(lambda x: x[3:].sum() > 0, axis=1)
        ].index
    ].apply(tuple, axis=1).to_numpy()
    logger.info(f"Disagreements: {disagreements}")

    # order subsets - only select indices that actually exist to avoid KeyError
    desired_order = [
        # pairs not included in categories:
        # (False, False, False, False, False, False),
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
        # diagreements
        #   ccc
        (False, True, False, False, False, True),
        (True, False, False, False, False, True),
        (True, True, False, False, False, True),
        (True, False, False, False, True, True),
        (False, True, False, True, False, True),
        #   pearson
        (False, False, True, False, True, False),
        (True, False, True, False, True, False),
        (True, False, False, False, True, False),
        (False, False, True, True, True, False),
        #   spearman
        (False, False, True, True, False, False),
        (False, True, True, True, False, False),
        (False, True, False, True, False, False),
    ]
    
    # Filter desired order to only include indices that exist in gene_pairs_by_cats
    existing_indices = [idx for idx in desired_order if idx in gene_pairs_by_cats.index]
    logger.info(f"Desired order indices: {len(desired_order)}, existing indices: {len(existing_indices)}")
    
    # Log which indices are missing
    missing_indices = [idx for idx in desired_order if idx not in gene_pairs_by_cats.index]
    if missing_indices:
        logger.info(f"Missing indices (will be skipped): {missing_indices}")
    
    # Only select existing indices
    gene_pairs_by_cats = gene_pairs_by_cats.loc[existing_indices]

    logger.info("Gene pairs by categories after reordering:")
    logger.info(f"\n{gene_pairs_by_cats.head()}")

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

    fig = plt.figure(figsize=(14, 5))

    g = MyUpSet(
        gene_pairs_by_cats,
        show_counts=True,
        sort_categories_by=None,
        sort_by=None,
        show_percentages=True,
        element_size=None,
    ).plot(fig)

    g["totals"].remove()

    trimmed_svg_path = args.output_dir / f"{OUTPUT_FIGURE_NAME}_trimmed.svg"
    plt.savefig(
        trimmed_svg_path,
        bbox_inches="tight",
        facecolor="white",
    )
    logger.info(f"Saved trimmed upset plot to: {trimmed_svg_path}")

    # Also save to log directory
    log_svg_path = LOG_DIR / f"{OUTPUT_FIGURE_NAME}_trimmed.svg"
    shutil.copy2(trimmed_svg_path, log_svg_path)
    logger.info(f"Copied trimmed upset plot to log directory: {log_svg_path}")


def load_tissue_thresholds(tissue_threshold_dir: Path, tissue_name: str, 
                          low_percentile: float = 0.30, high_percentile: float = 0.70) -> Dict:
    """
    Load tissue-specific thresholds from null_coefs_percentiles file.
    
    Args:
        tissue_threshold_dir: Directory containing threshold files
        tissue_name: Name of the tissue
        low_percentile: Percentile to use for low threshold (e.g., 0.90)
        high_percentile: Percentile to use for high threshold (e.g., 0.95)
        
    Returns:
        Dictionary with threshold values for each coefficient
    """
    # Get logger
    logger = logging.getLogger(__name__)
    
    # Clean tissue name (remove any extra whitespace or commas)
    tissue_name = tissue_name.strip().rstrip(',')
    logger.info(f"Loading thresholds for tissue: '{tissue_name}'")
    
    # Expected filename pattern (based on actual file structure)
    threshold_file = tissue_threshold_dir / f"{tissue_name}-null_coefs_percentiles.pkl"
    
    # Try alternative naming patterns if the first doesn't exist
    alternative_patterns = [
        f"null_coefs_percentiles_{tissue_name}.pkl",
        f"{tissue_name}_null_coefs_percentiles.pkl",
        f"null_coefs_percentiles-{tissue_name}.pkl",
        f"percentiles_{tissue_name}.pkl",
        f"{tissue_name}-null_coefs_percentiles.csv",
        f"null_coefs_percentiles_{tissue_name}.csv",
        f"{tissue_name}_null_coefs_percentiles.csv"
    ]
    
    if not threshold_file.exists():
        for pattern in alternative_patterns:
            alt_file = tissue_threshold_dir / pattern
            if alt_file.exists():
                threshold_file = alt_file
                break
        else:
            raise FileNotFoundError(f"No threshold file found for tissue '{tissue_name}' in {tissue_threshold_dir}")
    
    # Load the threshold file
    try:
        # Detect file type and load accordingly
        if threshold_file.suffix.lower() == '.pkl':
            thresholds_df = pd.read_pickle(threshold_file)
            logger.info(f"Loaded pickle threshold file: {threshold_file}")
        elif threshold_file.suffix.lower() == '.csv':
            thresholds_df = pd.read_csv(threshold_file, index_col=0)
            logger.info(f"Loaded CSV threshold file: {threshold_file}")
        else:
            raise ValueError(f"Unsupported file format: {threshold_file.suffix}")
            
        logger.info(f"Threshold file shape: {thresholds_df.shape}")
        logger.info(f"Available columns: {list(thresholds_df.columns)}")
        logger.info(f"Index (percentiles): {sorted(thresholds_df.index.tolist())}")
        
        # Extract thresholds for each coefficient
        # Assuming the file has percentiles as index and coefficients as columns
        thresholds = {}
        
        # Map coefficient names (adjust these based on actual column names in the file)
        coef_mapping = {
            'ccc': ['ccc', 'clustermatch', 'CCC', 'Clustermatch'],
            'pearson': ['pearson', 'Pearson', 'PEARSON'],
            'spearman': ['spearman', 'Spearman', 'SPEARMAN']
        }
        
        for coef_name, possible_names in coef_mapping.items():
            found = False
            for possible_name in possible_names:
                if possible_name in thresholds_df.columns:
                    # Use specified low/high percentiles
                    if low_percentile in thresholds_df.index and high_percentile in thresholds_df.index:
                        thresholds[f'{coef_name}_low'] = thresholds_df.loc[low_percentile, possible_name]
                        thresholds[f'{coef_name}_high'] = thresholds_df.loc[high_percentile, possible_name]
                        logger.info(f"Using exact percentiles for {coef_name}: {low_percentile:.2f} (low), {high_percentile:.2f} (high)")
                    else:
                        # Find closest percentiles if exact ones don't exist
                        available_percentiles = sorted(thresholds_df.index)
                        low_idx = min(available_percentiles, key=lambda x: abs(x - low_percentile))
                        high_idx = min(available_percentiles, key=lambda x: abs(x - high_percentile))
                        thresholds[f'{coef_name}_low'] = thresholds_df.loc[low_idx, possible_name]
                        thresholds[f'{coef_name}_high'] = thresholds_df.loc[high_idx, possible_name]
                        logger.info(f"Using closest percentiles for {coef_name}: {low_idx:.2f} (low, requested {low_percentile:.2f}), {high_idx:.2f} (high, requested {high_percentile:.2f})")
                    found = True
                    logger.info(f"Found {coef_name} thresholds using column '{possible_name}': low={thresholds[f'{coef_name}_low']:.6f}, high={thresholds[f'{coef_name}_high']:.6f}")
                    break
            
            if not found:
                raise ValueError(f"Could not find {coef_name} coefficient in threshold file columns: {list(thresholds_df.columns)}")
        
        logger.info(f"Loaded thresholds: {thresholds}")
        return thresholds
        
    except Exception as e:
        logger.error(f"Error loading threshold file {threshold_file}: {str(e)}")
        raise


def main():
    """Main function to run the gene pair intersection analysis."""
    args = parse_arguments()
    
    # Get logger
    logger = logging.getLogger(__name__)
    
    # Validate percentile arguments if tissue threshold directory is provided
    if args.tissue_threshold_dir:
        if not (0 < args.low_percentile < 1):
            raise ValueError(f"low-percentile must be between 0 and 1, got: {args.low_percentile}")
        if not (0 < args.high_percentile < 1):
            raise ValueError(f"high-percentile must be between 0 and 1, got: {args.high_percentile}")
        if args.low_percentile >= args.high_percentile:
            raise ValueError(f"low-percentile ({args.low_percentile}) must be less than "
                           f"high-percentile ({args.high_percentile})")
    
    # Load tissue-specific thresholds if threshold directory is provided
    tissue_thresholds = None
    if args.tissue_threshold_dir:
        if not args.tissue_threshold_dir.exists():
            raise FileNotFoundError(f"Tissue threshold directory does not exist: {args.tissue_threshold_dir}")
        
        tissue_thresholds = load_tissue_thresholds(
            args.tissue_threshold_dir, args.gtex_tissue, 
            args.low_percentile, args.high_percentile
        )
        logger.info(f"Using tissue-specific thresholds from: {args.tissue_threshold_dir}")
        logger.info(f"  Low percentile: {args.low_percentile} ({args.low_percentile*100:.1f}th percentile)")
        logger.info(f"  High percentile: {args.high_percentile} ({args.high_percentile*100:.1f}th percentile)")
    else:
        logger.info(f"No tissue threshold directory provided, using default q_diff={args.q_diff}")
    
    # Set up derived paths and names
    SIMILARITY_MATRICES_DIR = args.data_dir / "similarity_matrices" / args.top_n_genes
    OUTPUT_FIGURE_NAME = f"upsetplot_gtex_{args.gtex_tissue}"
    OUTPUT_GENE_PAIR_INTERSECTIONS_NAME = f"gene_pair_intersections-gtex_v8-{args.gtex_tissue}-{args.gene_sel_strategy}.pkl"
    
    # Check if output file already exists
    output_file = args.output_dir / OUTPUT_GENE_PAIR_INTERSECTIONS_NAME
    
    # If --use-existing is enabled and file exists, load it for plotting
    if args.use_existing and output_file.exists():
        print(f"Loading existing intersection file: {output_file}")
        
        # Setup log directory
        if args.log_dir:
            LOG_DIR = args.log_dir
            LOG_DIR.mkdir(parents=True, exist_ok=True)
        else:
            # Create timestamp-based log folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            LOG_DIR = Path("logs") / timestamp
            LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Setup logging with tissue-specific log file
        log_file = LOG_DIR / f"compute_intersections_{args.gtex_tissue}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        logger.info(f"Starting compute_intersections.py with --use-existing flag for tissue: {args.gtex_tissue}")
        logger.info(f"Loading existing intersection data from: {output_file}")
        
        # Load existing data
        df_plot = pd.read_pickle(output_file)
        logger.info(f"Loaded dataframe shape: {df_plot.shape}")
        logger.info(f"Loaded dataframe columns: {df_plot.columns.tolist()}")
        
        # Create categories list (same as in computation)
        categories = sorted(
            [x for x in df_plot.columns if " (" in x],
            reverse=True,
            key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],
        )
        logger.info(f"Categories for upset plot: {categories}")
        
        # Generate upset plots
        generate_upset_plots(df_plot, categories, args, OUTPUT_FIGURE_NAME, LOG_DIR, logger)
        
        logger.info("Plots generated successfully from existing data!")
        logger.info(f"All outputs saved to log directory: {LOG_DIR}")
        return

    # If not using existing or file doesn't exist, check for early exit
    if not args.use_existing and output_file.exists():
        print(f"Output file already exists: {output_file}")
        print("Script stopping - output file already exists")
        print("Use --use-existing flag to load existing data and generate plots")
        sys.exit(0)

    # Setup log directory
    if args.log_dir:
        LOG_DIR = args.log_dir
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = "user_provided"
    else:
        # Create timestamp-based log folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_DIR = Path("logs") / timestamp
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging with tissue-specific log file
    log_file = LOG_DIR / f"compute_intersections_{args.gtex_tissue}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting compute_intersections.py for tissue: {args.gtex_tissue}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Configuration:")
    logger.info(f"  GTEX_TISSUE: {args.gtex_tissue}")
    logger.info(f"  GENE_SEL_STRATEGY: {args.gene_sel_strategy}")
    logger.info(f"  TOP_N_GENES: {args.top_n_genes}")
    logger.info(f"  Q_DIFF: {args.q_diff}")
    logger.info(f"  OUTPUT_DIR: {args.output_dir}")
    logger.info(f"  USE_EXISTING: {args.use_existing}")
    logger.info(f"  LOG_DIR: {args.log_dir}")
    if args.tissue_threshold_dir is not None:
        logger.info(f"  TISSUE_THRESHOLD_DIR: {args.tissue_threshold_dir}")
        logger.info(f"  LOW_PERCENTILE: {args.low_percentile}")
        logger.info(f"  HIGH_PERCENTILE: {args.high_percentile}")
    else:
        logger.info(f"  TISSUE_THRESHOLD_DIR: Not specified (using default q_diff)")

    assert args.output_dir.exists()
    assert SIMILARITY_MATRICES_DIR.exists()

    SIMILARITY_MATRIX_FILENAME_TEMPLATE = "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"
    INPUT_CORR_FILE_TEMPLATE = SIMILARITY_MATRICES_DIR / SIMILARITY_MATRIX_FILENAME_TEMPLATE
    logger.info(f"Input correlation file template: {INPUT_CORR_FILE_TEMPLATE}")

    INPUT_CORR_FILE = SIMILARITY_MATRICES_DIR / str(
        INPUT_CORR_FILE_TEMPLATE
    ).format(
        tissue=args.gtex_tissue,
        gene_sel_strategy=args.gene_sel_strategy,
        corr_method="all",
    )
    logger.info(f"Input correlation file: {INPUT_CORR_FILE}")

    assert INPUT_CORR_FILE.exists()

    logger.info("Loading correlation data...")
    df = pd.read_pickle(INPUT_CORR_FILE)

    logger.info(f"Dataframe shape: {df.shape}")

    logger.info("Dataframe head (first 15 rows):")
    logger.info(f"\n{df.head(15)}")

    logger.info("Dataframe statistics:")
    logger.info(f"\n{df.describe()}")

    # Calculate quantiles from 70% to 100% in 0.01 steps for each correlation method
    # This helps understand the distribution of correlation values and identify
    # appropriate thresholds for high/low correlation gene pairs
    quantiles_result = df.apply(lambda x: x.quantile(np.linspace(0.70, 1.0, 31)))
    logger.info("Quantiles from 70% to 100% in 0.01 steps (31 points):")
    logger.info(f"\n{quantiles_result}")

    def get_threshold_values(method_name, q, tissue_thresholds=None):
        """Get the lower and upper threshold values for a correlation method.
        
        This function either calculates quantile thresholds from the data or uses
        pre-computed threshold values from tissue-specific files. Gene pairs
        with correlation values <= lower threshold are considered "low correlation",
        while those with values >= upper threshold are considered "high correlation".
        
        Args:
            method_name (str): Name of the correlation method column ('ccc', 'pearson', 'spearman')
            q (float): Quantile difference from extremes (e.g., 0.30 means use 30% and 70% quantiles)
            tissue_thresholds (dict): Optional dict with pre-computed threshold values
        
        Returns:
            tuple: (lower_threshold, upper_threshold) values
        """
        if tissue_thresholds:
            # Use pre-computed thresholds from file
            lower_key = f'{method_name}_low'
            upper_key = f'{method_name}_high'
            if lower_key in tissue_thresholds and upper_key in tissue_thresholds:
                return tissue_thresholds[lower_key], tissue_thresholds[upper_key]
            else:
                raise ValueError(f"Threshold values for {method_name} not found in tissue thresholds")
        else:
            # Calculate quantiles from data
            quantiles = df[method_name].quantile([q, 1 - q])
            return quantiles.iloc[0], quantiles.iloc[1]

    # Test the get_threshold_values function with CCC method
    # Using 0.20 quantile difference (20% and 80% quantiles)
    ccc_lower_test, ccc_upper_test = get_threshold_values("ccc", 0.20, tissue_thresholds)
    logger.info("CCC threshold test:")
    logger.info(f"Lower: {ccc_lower_test}, Upper: {ccc_upper_test}")

    # Clean up test variables
    del ccc_lower_test, ccc_upper_test

    # Calculate thresholds using either tissue-specific values or default quantiles
    if tissue_thresholds:
        logger.info(f"Using tissue-specific thresholds for tissue '{args.gtex_tissue}':")
        
        clustermatch_lq, clustermatch_hq = get_threshold_values("ccc", args.q_diff, tissue_thresholds)
        pearson_lq, pearson_hq = get_threshold_values("pearson", args.q_diff, tissue_thresholds)
        spearman_lq, spearman_hq = get_threshold_values("spearman", args.q_diff, tissue_thresholds)
    else:
        logger.info(f"Using default quantiles with q_diff={args.q_diff}:")
        logger.info(f"  Lower quantile: {args.q_diff} ({args.q_diff*100:.1f}th percentile)")
        logger.info(f"  Upper quantile: {1-args.q_diff} ({(1-args.q_diff)*100:.1f}th percentile)")
        
        clustermatch_lq, clustermatch_hq = get_threshold_values("ccc", args.q_diff, tissue_thresholds)
        pearson_lq, pearson_hq = get_threshold_values("pearson", args.q_diff, tissue_thresholds)
        spearman_lq, spearman_hq = get_threshold_values("spearman", args.q_diff, tissue_thresholds)

    logger.info(f"Clustermatch quantiles: lower={clustermatch_lq:.6f}, upper={clustermatch_hq:.6f}")
    logger.info(f"Pearson quantiles: lower={pearson_lq:.6f}, upper={pearson_hq:.6f}")
    logger.info(f"Spearman quantiles: lower={spearman_lq:.6f}, upper={spearman_hq:.6f}")
    
    # Log detailed threshold information with percentile values
    logger.info("\n" + "="*60)
    logger.info("DETAILED THRESHOLD INFORMATION")
    logger.info("="*60)
    
    if tissue_thresholds:
        logger.info(f"Using TISSUE-SPECIFIC thresholds from file:")
        logger.info(f"  Threshold source: {args.tissue_threshold_dir}")
        logger.info(f"  Tissue: {args.gtex_tissue}")
        logger.info(f"  Low threshold percentile:  {args.low_percentile*100:.1f}th percentile")
        logger.info(f"  High threshold percentile: {args.high_percentile*100:.1f}th percentile")
    else:
        logger.info(f"Using DEFAULT quantiles with q_diff={args.q_diff}:")
        logger.info(f"  Low threshold percentile:  {args.q_diff*100:.1f}th percentile")
        logger.info(f"  High threshold percentile: {(1-args.q_diff)*100:.1f}th percentile")
    
    logger.info(f"\nActual threshold values:")
    logger.info(f"  CCC (Clustermatch):")
    logger.info(f"    Low threshold:  {clustermatch_lq:.6f}")
    logger.info(f"    High threshold: {clustermatch_hq:.6f}")
    logger.info(f"  Pearson:")
    logger.info(f"    Low threshold:  {pearson_lq:.6f}")
    logger.info(f"    High threshold: {pearson_hq:.6f}")
    logger.info(f"  Spearman:")
    logger.info(f"    Low threshold:  {spearman_lq:.6f}")
    logger.info(f"    High threshold: {spearman_hq:.6f}")
    logger.info("="*60)

    pearson_higher = df["pearson"] >= pearson_hq
    logger.info(f"Pearson higher count: {pearson_higher.sum()}")

    pearson_lower = df["pearson"] <= pearson_lq
    logger.info(f"Pearson lower count: {pearson_lower.sum()}")

    spearman_higher = df["spearman"] >= spearman_hq
    logger.info(f"Spearman higher count: {spearman_higher.sum()}")

    spearman_lower = df["spearman"] <= spearman_lq
    logger.info(f"Spearman lower count: {spearman_lower.sum()}")

    clustermatch_higher = df["ccc"] >= clustermatch_hq
    logger.info(f"Clustermatch higher count: {clustermatch_higher.sum()}")

    clustermatch_lower = df["ccc"] <= clustermatch_lq
    logger.info(f"Clustermatch lower count: {clustermatch_lower.sum()}")

    # Create a dataframe for plotting with boolean columns indicating whether each gene pair
    # falls into the high or low quantile ranges for each correlation method
    df_plot = pd.DataFrame(
        {
            "pearson_higher": pearson_higher,  # Gene pairs with Pearson correlation >= high quantile
            "pearson_lower": pearson_lower,    # Gene pairs with Pearson correlation <= low quantile
            "spearman_higher": spearman_higher,  # Gene pairs with Spearman correlation >= high quantile
            "spearman_lower": spearman_lower,    # Gene pairs with Spearman correlation <= low quantile
            "clustermatch_higher": clustermatch_higher,  # Gene pairs with CCC >= high quantile
            "clustermatch_lower": clustermatch_lower,    # Gene pairs with CCC <= low quantile
        }
    )

    # Add the original correlation values (ccc, pearson, spearman) to the plot dataframe
    df_plot = pd.concat([df_plot, df], axis=1)

    # Clean up
    del df
    del pearson_higher, pearson_lower, spearman_higher, spearman_lower, clustermatch_higher, clustermatch_lower

    logger.info(f"Plot dataframe shape: {df_plot.shape}")
    logger.info(f"Plot dataframe columns: {df_plot.columns.tolist()}")

    assert not df_plot.isna().any().any()

    # Rename columns to more descriptive names for plotting
    # This creates cleaner labels for the UpSet plot visualization
    df_plot = df_plot.rename(
        columns={
            "pearson_higher": "Pearson (high)",
            "pearson_lower": "Pearson (low)",
            "spearman_higher": "Spearman (high)",
            "spearman_lower": "Spearman (low)",
            "clustermatch_higher": "Clustermatch (high)",
            "clustermatch_lower": "Clustermatch (low)",
        }
    )

    # Create sorted list of category names for the UpSet plot
    # Filter columns that contain " (" (our boolean indicator columns)
    # Sort by: first by threshold level (high/low), then by method name
    # This ensures consistent ordering: high thresholds before low thresholds,
    # and within each threshold level, methods are alphabetically ordered
    categories = sorted(
        [x for x in df_plot.columns if " (" in x],  # Get boolean indicator columns
        reverse=True,  # Reverse to get "high" before "low"
        key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],  # Sort by threshold then method
    )

    logger.info(f"Categories for upset plot: {categories}")

    # Generate upset plots
    generate_upset_plots(df_plot, categories, args, OUTPUT_FIGURE_NAME, LOG_DIR, logger)

    logger.info(f"Final dataframe shape: {df_plot.shape}")
    logger.info("Final dataframe head:")
    logger.info(f"\n{df_plot.head()}")

    logger.info(f"Output file for gene pair intersections: {output_file}")

    logger.info("Saving gene pair intersections data...")
    df_plot.to_pickle(output_file)
    logger.info(f"Saved gene pair intersections to: {output_file}")

    logger.info("Script completed successfully!")
    logger.info(f"All outputs saved to log directory: {LOG_DIR}")


if __name__ == "__main__":
    main()