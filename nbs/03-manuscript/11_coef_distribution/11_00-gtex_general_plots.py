#!/usr/bin/env python
# coding: utf-8
"""
GTEx Coefficient Distribution Analysis Script

Generates comprehensive plots to compare coefficient values from Pearson, Spearman and CCC,
including their distributions, cumulative histograms, and joint plots.

Features:
- Comprehensive logging with timestamped log directories
- Dual output: saves figures to both original destinations and log folder
- Command-line interface with configurable parameters
- Robust error handling and progress tracking
- Clean, modular code structure
"""

import sys
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# Import heavy dependencies only when needed
# This allows --help to work even without all dependencies installed

# Global logger
logger = None


def setup_logging(log_dir: str) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    global logger
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging configuration
    log_file = log_path / 'coefficient_analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Log file: {log_file}")
    logger.info("="*80)
    logger.info("GTEx COEFFICIENT DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    return logger


def validate_input_files(gene_expr_file: Path, correlation_file: Path) -> bool:
    """
    Validate that required input files exist.
    
    Args:
        gene_expr_file: Path to gene expression data file
        correlation_file: Path to correlation data file
        
    Returns:
        True if all files exist, False otherwise
    """
    logger.info("Validating input files...")
    
    files_to_check = [
        ("Gene expression file", gene_expr_file),
        ("Correlation file", correlation_file)
    ]
    
    all_exist = True
    for file_desc, file_path in files_to_check:
        if file_path.exists():
            logger.info(f"✅ {file_desc}: {file_path}")
        else:
            logger.error(f"❌ {file_desc} not found: {file_path}")
            all_exist = False
    
    return all_exist


def load_and_prepare_data(input_file: Path, clustermatch_label: str, 
                         pearson_label: str, spearman_label: str) -> 'pd.DataFrame':
    """
    Load correlation data and prepare it for analysis.
    
    This function loads the correlation data, renames columns to use provided labels,
    removes any rows with NaN values in the coefficient columns, and logs detailed
    statistics about the data cleaning process.
    
    Args:
        input_file: Path to input correlation data file
        clustermatch_label: Label for CCC/Clustermatch column
        pearson_label: Label for Pearson column
        spearman_label: Label for Spearman column
        
    Returns:
        Clean DataFrame with renamed columns and no NaN values in coefficient columns
    """
    # Import dependencies when needed
    import pandas as pd
    from scipy import stats
    
    logger.info(f"Loading correlation data from: {input_file}")
    
    try:
        df = pd.read_pickle(input_file).rename(
            columns={
                "ccc": clustermatch_label,
                "pearson": pearson_label,
                "spearman": spearman_label,
            }
        )
        
        logger.info(f"✅ Data loaded successfully - Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Remove rows with NaN values in any of the three coefficient columns
        original_shape = df.shape
        coefficient_columns = [clustermatch_label, pearson_label, spearman_label]
        
        logger.info("🧹 Cleaning data: removing rows with NaN values...")
        df_clean = df.dropna(subset=coefficient_columns)
        
        rows_removed = original_shape[0] - df_clean.shape[0]
        if rows_removed > 0:
            logger.info(f"  ❌ Removed {rows_removed:,} rows with NaN values ({rows_removed/original_shape[0]*100:.2f}%)")
            logger.info(f"  ✅ Clean data shape: {df_clean.shape}")
        else:
            logger.info(f"  ✅ No NaN values found - all {original_shape[0]:,} rows retained")
        
        # Check for any remaining NaN values (safety check)
        nan_counts = df_clean[coefficient_columns].isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(f"⚠️  Warning: {total_nans} NaN values still present after cleaning:")
            for col, count in nan_counts.items():
                if count > 0:
                    logger.warning(f"    {col}: {count} NaN values")
        else:
            logger.info("✅ Data cleaning verified: no NaN values in coefficient columns")
        
        # Log basic statistics on clean data
        logger.info("Data Statistics (after cleaning):")
        logger.info(f"  Mean values: {df_clean.mean().to_dict()}")
        logger.info(f"  Std values: {df_clean.std().to_dict()}")
        logger.info(f"  Skewness: {df_clean.apply(lambda x: stats.skew(x)).to_dict()}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def save_figure_dual_location(figure_or_path, original_dir: Path, log_dir: Path, 
                             filename: str) -> None:
    """
    Save figure to both original directory and log directory.
    
    Args:
        figure_or_path: matplotlib figure object or path to existing file
        original_dir: Original output directory
        log_dir: Log directory
        filename: Figure filename
    """
    original_path = original_dir / filename
    log_path = log_dir / filename
    
    try:
        if isinstance(figure_or_path, (str, Path)):
            # Copy existing file
            if Path(figure_or_path).exists():
                shutil.copy2(figure_or_path, log_path)
                logger.info(f"📊 Figure copied to log: {filename}")
            else:
                logger.warning(f"Source figure not found: {figure_or_path}")
        else:
            # Save matplotlib figure
            figure_or_path.savefig(log_path, bbox_inches="tight", dpi=300, facecolor="white")
            logger.info(f"📊 Figure saved to log: {filename}")
            
    except Exception as e:
        logger.warning(f"Failed to copy figure to log directory: {str(e)}")


def generate_histograms(df: 'pd.DataFrame', original_output_dir: Path, log_dir: Path) -> None:
    """
    Generate histogram plots for coefficient distributions.
    
    Args:
        df: DataFrame with coefficient data
        original_output_dir: Original output directory
        log_dir: Log directory for additional output
    """
    # Import dependencies when needed
    import seaborn as sns
    from ccc.plots import plot_histogram
    
    logger.info("Generating histogram plots...")
    
    try:
        with sns.plotting_context("talk", font_scale=1.0):
            plot_histogram(df, output_dir=original_output_dir, fill=False)
        
        # Copy to log directory
        hist_file = original_output_dir / "dist-histograms.svg"
        if hist_file.exists():
            save_figure_dual_location(hist_file, original_output_dir, log_dir, "dist-histograms.svg")
        
        logger.info("✅ Histogram plots generated successfully")
        
    except Exception as e:
        logger.error(f"Failed to generate histogram plots: {str(e)}")
        raise


def generate_cumulative_histograms(df: 'pd.DataFrame', gene_pairs_percent: float,
                                 original_output_dir: Path, log_dir: Path) -> None:
    """
    Generate cumulative histogram plots.
    
    Args:
        df: DataFrame with coefficient data
        gene_pairs_percent: Percentage for cumulative histogram
        original_output_dir: Original output directory
        log_dir: Log directory for additional output
    """
    # Import dependencies when needed
    import seaborn as sns
    from ccc.plots import plot_cumulative_histogram
    
    logger.info(f"Generating cumulative histogram plots (gene pairs percent: {gene_pairs_percent})...")
    
    try:
        with sns.plotting_context("talk", font_scale=1.0):
            plot_cumulative_histogram(df, gene_pairs_percent, output_dir=original_output_dir)
        
        # Copy to log directory
        cumhist_file = original_output_dir / "dist-cum_histograms.svg"
        if cumhist_file.exists():
            save_figure_dual_location(cumhist_file, original_output_dir, log_dir, "dist-cum_histograms.svg")
        
        logger.info("✅ Cumulative histogram plots generated successfully")
        
    except Exception as e:
        logger.error(f"Failed to generate cumulative histogram plots: {str(e)}")
        raise


def generate_joint_plots(df: 'pd.DataFrame', pearson_label: str, spearman_label: str,
                        clustermatch_label: str, original_output_dir: Path, log_dir: Path) -> None:
    """
    Generate joint plots comparing each coefficient.
    
    Args:
        df: DataFrame with coefficient data
        pearson_label: Label for Pearson coefficient
        spearman_label: Label for Spearman coefficient
        clustermatch_label: Label for CCC coefficient
        original_output_dir: Original output directory
        log_dir: Log directory for additional output
    """
    # Import dependencies when needed
    import seaborn as sns
    from ccc.plots import jointplot
    
    logger.info("Generating joint plots...")
    
    try:
        # Plot 1: Pearson vs CCC
        logger.info("  Creating Pearson vs CCC joint plot...")
        with sns.plotting_context("talk", font_scale=1.0):
            jointplot(
                data=df,
                x=pearson_label,
                y=clustermatch_label,
                add_corr_coefs=False,
                output_dir=original_output_dir,
            )
        
        # Copy to log directory
        joint1_file = original_output_dir / f"dist-{pearson_label.lower()}_vs_{clustermatch_label.lower()}.svg"
        if joint1_file.exists():
            save_figure_dual_location(joint1_file, original_output_dir, log_dir, 
                                    f"dist-{pearson_label.lower()}_vs_{clustermatch_label.lower()}.svg")
        
        # Plot 2: Spearman vs CCC (custom styling)
        logger.info("  Creating Spearman vs CCC joint plot...")
        with sns.plotting_context("talk", font_scale=1.0):
            x, y = spearman_label, clustermatch_label
            
            g = jointplot(
                data=df,
                x=x,
                y=y,
                add_corr_coefs=False,
            )
            
            sns.despine(ax=g.ax_joint, left=True)
            g.ax_joint.set_yticks([])
            g.ax_joint.set_ylabel(None)
            
            output_file = original_output_dir / f"dist-{x.lower()}_vs_{y.lower()}.svg"
            g.savefig(output_file, bbox_inches="tight", dpi=300, facecolor="white")
        
        # Copy to log directory
        if output_file.exists():
            save_figure_dual_location(output_file, original_output_dir, log_dir, 
                                    f"dist-{x.lower()}_vs_{y.lower()}.svg")
        
        # Plot 3: Spearman vs Pearson
        logger.info("  Creating Spearman vs Pearson joint plot...")
        with sns.plotting_context("talk", font_scale=1.0):
            jointplot(
                data=df,
                x=spearman_label,
                y=pearson_label,
                add_corr_coefs=False,
                output_dir=original_output_dir,
            )
        
        # Copy to log directory
        joint3_file = original_output_dir / f"dist-{spearman_label.lower()}_vs_{pearson_label.lower()}.svg"
        if joint3_file.exists():
            save_figure_dual_location(joint3_file, original_output_dir, log_dir, 
                                    f"dist-{spearman_label.lower()}_vs_{pearson_label.lower()}.svg")
        
        logger.info("✅ Joint plots generated successfully")
        
    except Exception as e:
        logger.error(f"Failed to generate joint plots: {str(e)}")
        raise


def create_composite_figure(original_output_dir: Path, log_dir: Path) -> None:
    """
    Create the final composite figure combining all plots.
    
    Args:
        original_output_dir: Original output directory
        log_dir: Log directory for additional output
    """
    # Import dependencies when needed
    from svgutils.compose import Figure, SVG, Panel, Text
    
    logger.info("Creating composite figure...")
    
    try:
        # Create composite figure
        composite_fig = Figure(
            "643.71cm",
            "427.66cm",
            Panel(
                SVG(original_output_dir / "dist-histograms.svg").scale(0.5),
                Text("a)", 2, 10, size=9, weight="bold"),
            ),
            Panel(
                SVG(original_output_dir / "dist-cum_histograms.svg").scale(0.5),
                Text("b)", 2, 10, size=9, weight="bold"),
            ).move(320, 0),
            Panel(
                SVG(original_output_dir / "dist-pearson_vs_ccc.svg").scale(0.595),
                Panel(
                    SVG(original_output_dir / "dist-spearman_vs_ccc.svg")
                    .scale(0.595)
                    .move(215, 0)
                ),
                Panel(
                    SVG(original_output_dir / "dist-spearman_vs_pearson.svg")
                    .scale(0.595)
                    .move(430, 0)
                ),
                Text("c)", 2, 10, size=9, weight="bold"),
            ).move(0, 220),
        )
        
        # Save to both directories
        original_composite = original_output_dir / "dist-main.svg"
        log_composite = log_dir / "dist-main.svg"
        
        composite_fig.save(original_composite)
        composite_fig.save(log_composite)
        
        logger.info("✅ Composite figure created successfully")
        logger.info(f"   Original: {original_composite}")
        logger.info(f"   Log copy: {log_composite}")
        
    except Exception as e:
        logger.error(f"Failed to create composite figure: {str(e)}")
        raise


def main() -> int:
    """
    Main execution function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Generate GTEx coefficient distribution plots with comprehensive analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument("--tissue", default="whole_blood", 
                       help="GTEx tissue type to analyze")
    parser.add_argument("--top-n-genes", default="all",
                       help="Number of top genes to use ('all' for all genes)")
    parser.add_argument("--gene-selection-strategy", default="var_pc_log2",
                       help="Gene selection strategy")
    parser.add_argument("--gene-pairs-percent", type=float, default=0.70,
                       help="Percentage for cumulative histogram")
    
    # Directory arguments
    parser.add_argument("--data-dir", 
                       default="/pividori_lab/haoyu_projects/ccc-gpu/data/gtex",
                       help="Base data directory")
    parser.add_argument("--output-dir",
                       default="/pividori_lab/haoyu_projects/ccc-gpu/figures",
                       help="Base output directory")
    parser.add_argument("--log-dir", 
                       help="Custom log directory (default: ./logs/YYYYMMDD_HHMMSS)")
    
    # Labels
    parser.add_argument("--ccc-label", default="CCC", help="Label for CCC coefficient")
    parser.add_argument("--pearson-label", default="Pearson", help="Label for Pearson coefficient")
    parser.add_argument("--spearman-label", default="Spearman", help="Label for Spearman coefficient")
    
    args = parser.parse_args()
    
    # Setup timestamped log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path("logs") / timestamp
    
    # Initialize logging
    global logger
    logger = setup_logging(str(log_dir))
    
    try:
        # Log configuration
        logger.info("Configuration:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")
        
        # Setup paths
        data_dir = Path(args.data_dir)
        gene_selection_dir = data_dir / "gene_selection" / args.top_n_genes
        similarity_matrices_dir = data_dir / "similarity_matrices" / args.top_n_genes
        
        # Original output directory
        original_output_dir = (
            Path(args.output_dir) / "coefs_comp" / f"gtex_{args.tissue}"
        )
        original_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Original output directory: {original_output_dir}")
        logger.info(f"Log directory: {log_dir}")
        
        # Input files
        gene_expr_file = (
            gene_selection_dir / f"gtex_v8_data_{args.tissue}-{args.gene_selection_strategy}.pkl"
        )
        correlation_file = similarity_matrices_dir / (
            f"gtex_v8_data_{args.tissue}-{args.gene_selection_strategy}-all.pkl"
        )
        
        # Validate input files
        if not validate_input_files(gene_expr_file, correlation_file):
            logger.error("Input file validation failed")
            return 1
        
        # Load and prepare data
        df = load_and_prepare_data(
            correlation_file, 
            args.ccc_label, 
            args.pearson_label, 
            args.spearman_label
        )
        
        # Generate all plots
        logger.info("Starting plot generation...")
        
        # 1. Histograms
        generate_histograms(df, original_output_dir, log_dir)
        
        # 2. Cumulative histograms
        generate_cumulative_histograms(df, args.gene_pairs_percent, original_output_dir, log_dir)
        
        # 3. Joint plots
        generate_joint_plots(df, args.pearson_label, args.spearman_label, 
                           args.ccc_label, original_output_dir, log_dir)
        
        # 4. Composite figure
        create_composite_figure(original_output_dir, log_dir)
        
        # Final summary
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("Generated files:")
        
        # List generated files
        for output_dir_name, output_path in [("Original", original_output_dir), ("Log", log_dir)]:
            logger.info(f"\n{output_dir_name} directory ({output_path}):")
            if output_path.exists():
                for file_path in sorted(output_path.glob("*.svg")):
                    logger.info(f"  📊 {file_path.name}")
        
        logger.info("\n✅ All plots generated successfully!")
        logger.info(f"📁 Log directory: {log_dir}")
        logger.info(f"📁 Original directory: {original_output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

