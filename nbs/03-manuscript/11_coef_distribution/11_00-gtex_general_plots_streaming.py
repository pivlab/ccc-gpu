#!/usr/bin/env python
# coding: utf-8
"""
GTEx Coefficient Distribution Analysis Script - Streaming Version

Memory-efficient version that processes large datasets in chunks to avoid OOM kills.
Generates cumulative histogram plots using streaming approach.

Features:
- Memory-efficient chunk-based processing  
- Incremental histogram computation
- Comprehensive logging with progress tracking
- Custom matplotlib plotting functions
- Robust error handling
"""

import sys
import logging
import argparse
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    log_file = log_path / 'coefficient_analysis_streaming.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Streaming Analysis Logging initialized - Log file: {log_file}")
    logger.info("="*80)
    logger.info("GTEx COEFFICIENT DISTRIBUTION ANALYSIS (STREAMING)")
    logger.info("="*80)
    
    return logger


def get_data_info(input_file: Path) -> Dict:
    """
    Get basic information about the pickle file without loading all data.
    
    Args:
        input_file: Path to the pickle file
        
    Returns:
        Dict with basic info about the data
    """
    logger.info(f"Analyzing data file: {input_file}")
    
    try:
        # Try to load just a small sample to understand structure
        with open(input_file, 'rb') as f:
            # Load the data but we'll only peek at structure
            data = pickle.load(f)
            
        info = {
            'type': type(data).__name__,
            'shape': getattr(data, 'shape', 'unknown'),
            'columns': getattr(data, 'columns', 'unknown'),
            'dtypes': getattr(data, 'dtypes', 'unknown'),
            'memory_usage_mb': sys.getsizeof(data) / (1024**2) if hasattr(data, '__sizeof__') else 'unknown'
        }
        
        logger.info(f"📊 Data info: {info}")
        return info, data
        
    except Exception as e:
        logger.error(f"Failed to analyze data file: {str(e)}")
        raise


def process_data_in_chunks(data, coefficient_columns: List[str], chunk_size: int = 100000) -> Dict:
    """
    Process data in chunks to compute histograms incrementally.
    
    Args:
        data: Full dataset (we'll chunk this)
        coefficient_columns: List of coefficient column names
        chunk_size: Number of rows per chunk
        
    Returns:
        Dict containing accumulated histogram data
    """
    # Import pandas here to avoid issues if it's not available
    import pandas as pd
    
    total_rows = len(data)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    logger.info(f"🔄 Processing {total_rows:,} rows in {num_chunks} chunks of {chunk_size:,} rows each...")
    
    # Define consistent bins for all coefficients (-1 to 1 range)
    bins = np.linspace(-1, 1, 201)  # 200 bins from -1 to 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Initialize histogram accumulator
    histogram_data = {col: np.zeros(len(bins)-1) for col in coefficient_columns}
    total_processed = 0
    total_nan_removed = 0
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        
        logger.info(f"  📋 Processing chunk {chunk_idx+1}/{num_chunks} (rows {start_idx:,} to {end_idx-1:,})...")
        
        # Get chunk
        chunk_data = data.iloc[start_idx:end_idx]
        
        # Clean chunk - remove NaN values
        original_chunk_size = len(chunk_data)
        chunk_clean = chunk_data.dropna(subset=coefficient_columns)
        chunk_nan_removed = original_chunk_size - len(chunk_clean)
        total_nan_removed += chunk_nan_removed
        
        if chunk_nan_removed > 0:
            logger.info(f"    🧹 Removed {chunk_nan_removed} rows with NaN values from chunk")
        
        if len(chunk_clean) == 0:
            logger.warning(f"    ⚠️  Chunk {chunk_idx+1} is empty after cleaning - skipping")
            continue
            
        # Compute histograms for this chunk
        for col in coefficient_columns:
            values = chunk_clean[col].values
            # Clip values to be within bin range to avoid out-of-bounds issues
            values = np.clip(values, bins[0], bins[-1])
            hist, _ = np.histogram(values, bins=bins)
            histogram_data[col] += hist
            
        total_processed += len(chunk_clean)
        
        # Log progress every 10 chunks or at the end
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1:
            logger.info(f"    ✅ Processed {chunk_idx+1}/{num_chunks} chunks ({total_processed:,} clean rows)")
    
    logger.info(f"🎯 Chunk processing complete:")
    logger.info(f"  Total rows processed: {total_processed:,}")
    logger.info(f"  Total NaN rows removed: {total_nan_removed:,}")
    logger.info(f"  Clean data percentage: {total_processed/(total_processed+total_nan_removed)*100:.2f}%")
    
    return {
        'histograms': histogram_data,
        'bins': bins,
        'bin_centers': bin_centers,
        'total_processed': total_processed,
        'total_nan_removed': total_nan_removed,
        'coefficient_columns': coefficient_columns
    }


def generate_streaming_cumulative_histogram(histogram_result: Dict, gene_pairs_percent: float, 
                                          output_dir: Path, log_dir: Path) -> None:
    """
    Generate cumulative histogram using pre-computed histogram data.
    
    Args:
        histogram_result: Result from process_data_in_chunks
        gene_pairs_percent: Percentage for cumulative histogram
        output_dir: Output directory for figures
        log_dir: Log directory
    """
    import matplotlib.pyplot as plt
    import matplotlib
    # Use non-interactive backend to avoid memory issues
    matplotlib.use('Agg')
    
    logger.info(f"📊 Generating streaming cumulative histogram (target: {gene_pairs_percent*100}% of gene pairs)...")
    
    try:
        histograms = histogram_result['histograms']
        bins = histogram_result['bins']
        coefficient_columns = histogram_result['coefficient_columns']
        total_processed = histogram_result['total_processed']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors for each coefficient
        colors = {'CCC': '#1f77b4', 'Pearson': '#ff7f0e', 'Spearman': '#2ca02c'}
        
        # Filter to only positive range (0 to 1.0) - match the desired plot
        positive_mask = bins >= 0.0
        positive_bins = bins[positive_mask]
        
        # Generate cumulative histograms for each coefficient
        for col in coefficient_columns:
            # Get histogram counts for full range
            full_counts = histograms[col]
            
            # Filter counts to positive range only
            # We need to be careful about indexing since bins has one more element than counts
            positive_counts = full_counts[positive_mask[:-1]]  # bins has n+1 elements, counts has n
            positive_bin_centers = (positive_bins[:-1] + positive_bins[1:]) / 2
            
            total_count = np.sum(positive_counts)
            if total_count == 0:
                logger.warning(f"⚠️  No positive data found for {col}")
                continue
            
            # Compute NORMAL cumulative distribution (from low to high values)
            # This gives us "percentage of gene pairs with coefficient <= x"
            cumulative_counts = np.cumsum(positive_counts)
            cumulative_percentages = cumulative_counts / total_count * 100
            
            # Plot cumulative histogram
            color = colors.get(col, '#333333')
            ax.plot(positive_bin_centers, cumulative_percentages, label=col, linewidth=2, color=color)
        
        # Customize plot to match the desired format
        ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percent of gene pairs', fontsize=12, fontweight='bold')
        
        # Set title with subtitle for processed gene pairs count
        main_title = 'Cumulative Distribution of Correlation Coefficients'
        subtitle = f'Processed: {total_processed:,} gene pairs'
        ax.set_title(f'{main_title}\n{subtitle}', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set axis limits to match desired plot (0 to 1.0 on x, 0 to 100 on y)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save to both directories
        output_file = output_dir / 'dist-cum_histograms_streaming.svg'
        log_file = log_dir / 'dist-cum_histograms_streaming.svg'
        
        fig.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
        fig.savefig(log_file, bbox_inches='tight', dpi=300, facecolor='white')
        
        plt.close(fig)  # Free memory
        
        logger.info(f"✅ Streaming cumulative histogram generated successfully")
        logger.info(f"   Original: {output_file}")
        logger.info(f"   Log copy: {log_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate streaming cumulative histogram: {str(e)}")
        raise


def generate_streaming_regular_histogram(histogram_result: Dict, output_dir: Path, log_dir: Path) -> None:
    """
    Generate regular histogram using pre-computed histogram data.
    
    Args:
        histogram_result: Result from process_data_in_chunks
        output_dir: Output directory for figures
        log_dir: Log directory
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    logger.info("📊 Generating streaming regular histograms...")
    
    try:
        histograms = histogram_result['histograms']
        bins = histogram_result['bins']
        coefficient_columns = histogram_result['coefficient_columns']
        total_processed = histogram_result['total_processed']
        
        # Filter to only positive range (0 to 1.0) - same as cumulative plot
        positive_mask = bins >= 0.0
        positive_bins = bins[positive_mask]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = {'CCC': '#1f77b4', 'Pearson': '#ff7f0e', 'Spearman': '#2ca02c'}
        
        for idx, col in enumerate(coefficient_columns):
            ax = axes[idx]
            
            # Filter counts to positive range only
            full_counts = histograms[col]
            positive_counts = full_counts[positive_mask[:-1]]  # bins has n+1 elements, counts has n
            positive_bin_centers = (positive_bins[:-1] + positive_bins[1:]) / 2
            
            # Normalize to get density for positive range only
            total_count = np.sum(positive_counts)
            if total_count == 0:
                logger.warning(f"⚠️  No positive data found for {col}")
                continue
                
            bin_width = positive_bin_centers[1] - positive_bin_centers[0]
            density = positive_counts / total_count / bin_width
            
            # Plot histogram
            color = colors.get(col, '#333333')
            ax.bar(positive_bin_centers, density, width=bin_width * 0.8, 
                  color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Customize subplot
            ax.set_xlabel(f'{col} Value', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.0, 1.0)  # Set range to 0-1.0 only
            
            # Calculate quantiles from histogram data
            def calculate_quantiles_from_histogram(bin_centers, counts, quantiles):
                """Calculate quantiles from histogram bins and counts."""
                total = np.sum(counts)
                if total == 0:
                    return {q: 0.0 for q in quantiles}
                
                # Create cumulative distribution
                cumulative = np.cumsum(counts)
                cumulative_normalized = cumulative / total
                
                # Find quantile values
                quantile_values = {}
                for q in quantiles:
                    # Find the bin where cumulative probability crosses the quantile
                    idx = np.searchsorted(cumulative_normalized, q)
                    if idx >= len(bin_centers):
                        idx = len(bin_centers) - 1
                    quantile_values[q] = bin_centers[idx]
                
                return quantile_values
            
            # Calculate requested quantiles
            target_quantiles = [0.10, 0.25, 0.30, 0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            quantile_values = calculate_quantiles_from_histogram(positive_bin_centers, positive_counts, target_quantiles)
            
            # Add statistics (calculated from positive range only) 
            mean_approx = np.sum(positive_bin_centers * positive_counts) / total_count
            ax.axvline(mean_approx, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_approx:.3f}')
            ax.legend(fontsize=9)
            
            # Add quantile information as text below the plot with better formatting
            # Format quantiles with proper line breaks for better readability
            quantile_items = [f"{q:.0%}={v:.3f}" for q, v in quantile_values.items()]
            
            # Option 1: Group quantiles: 5 per line for better readability (comment out to use option 2)
            # quantile_lines = []
            # for i in range(0, len(quantile_items), 5):
            #     line_items = quantile_items[i:i+5]
            #     quantile_lines.append(", ".join(line_items))
            # quantile_text = "Quantiles:\n" + "\n".join(quantile_lines)
            
            # Option 2: One quantile per line (as requested by user)
            quantile_text = "Quantiles:\n" + "\n".join(quantile_items)
            
            # Position text below the subplot with more space for one-quantile-per-line
            ax.text(0.5, -0.30, quantile_text, transform=ax.transAxes, 
                   fontsize=7, ha='center', va='top', 
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.45)  # Make room for one-quantile-per-line text
        
        # Save to both directories
        output_file = output_dir / 'dist-histograms_streaming.svg'
        log_file = log_dir / 'dist-histograms_streaming.svg'
        
        fig.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
        fig.savefig(log_file, bbox_inches='tight', dpi=300, facecolor='white')
        
        plt.close(fig)  # Free memory
        
        logger.info(f"✅ Streaming regular histograms generated successfully")
        logger.info(f"   Original: {output_file}")
        logger.info(f"   Log copy: {log_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate streaming regular histograms: {str(e)}")
        raise


def generate_streaming_density_plot(histogram_result: Dict, output_dir: Path, log_dir: Path) -> None:
    """
    Generate overlaid density plot using pre-computed histogram data.
    This matches the style of the original ccc.plots.plot_histogram function.
    
    Args:
        histogram_result: Result from process_data_in_chunks
        output_dir: Output directory for figures
        log_dir: Log directory
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy.interpolate import interp1d
    matplotlib.use('Agg')
    
    logger.info("📊 Generating streaming density plot (overlaid coefficients)...")
    
    try:
        histograms = histogram_result['histograms']
        bins = histogram_result['bins']
        coefficient_columns = histogram_result['coefficient_columns']
        total_processed = histogram_result['total_processed']
        
        # Filter to only positive range (0 to 1.0) - same as other plots
        positive_mask = bins >= 0.0
        positive_bins = bins[positive_mask]
        
        # Create figure with seaborn-style formatting
        plt.style.use('default')  # Reset to clean style
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors for each coefficient (same as other plots)
        colors = {'CCC': '#1f77b4', 'Pearson': '#ff7f0e', 'Spearman': '#2ca02c'}
        
        # Generate density curves for each coefficient
        for col in coefficient_columns:
            # Filter counts to positive range only
            full_counts = histograms[col]
            positive_counts = full_counts[positive_mask[:-1]]  # bins has n+1 elements, counts has n
            positive_bin_centers = (positive_bins[:-1] + positive_bins[1:]) / 2
            
            total_count = np.sum(positive_counts)
            if total_count == 0:
                logger.warning(f"⚠️  No positive data found for {col}")
                continue
            
            # Convert counts to density (normalize by total count and bin width)
            bin_width = positive_bin_centers[1] - positive_bin_centers[0]
            density = positive_counts / total_count / bin_width
            
            # Create smooth density curve using interpolation
            # Filter out zero densities for cleaner curves
            nonzero_mask = density > 0
            if np.sum(nonzero_mask) > 3:  # Need at least 3 points for interpolation
                x_smooth = positive_bin_centers[nonzero_mask]
                y_smooth = density[nonzero_mask]
                
                # Create interpolation function
                try:
                    # Use cubic interpolation for smooth curves
                    interp_func = interp1d(x_smooth, y_smooth, kind='cubic', 
                                         bounds_error=False, fill_value=0)
                    
                    # Generate smooth x values
                    x_fine = np.linspace(x_smooth.min(), x_smooth.max(), 300)
                    y_fine = interp_func(x_fine)
                    y_fine = np.maximum(y_fine, 0)  # Ensure no negative densities
                    
                    # Plot smooth density curve
                    color = colors.get(col, '#333333')
                    ax.plot(x_fine, y_fine, label=col, linewidth=2.5, color=color)
                    
                except Exception as interp_e:
                    logger.warning(f"Interpolation failed for {col}, using bar plot: {str(interp_e)}")
                    # Fallback to simple line plot
                    color = colors.get(col, '#333333')
                    ax.plot(positive_bin_centers, density, label=col, linewidth=2, color=color)
            else:
                # Fallback for insufficient data points
                color = colors.get(col, '#333333')
                ax.plot(positive_bin_centers, density, label=col, linewidth=2, color=color)
        
        # Customize plot to match original histogram style
        ax.set_xlabel('Coefficient Value', fontsize=14, fontweight='bold')
        ax.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Correlation Coefficients', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper right')
        
        # Set axis limits to match other plots (0 to 1.0 on x)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0)  # Start y-axis at 0
        
        # Style improvements to match seaborn aesthetic
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        plt.tight_layout()
        
        # Save to both directories
        output_file = output_dir / 'dist-histograms.svg'  # Match original filename
        log_file = log_dir / 'dist-histograms.svg'
        
        fig.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
        fig.savefig(log_file, bbox_inches='tight', dpi=300, facecolor='white')
        
        plt.close(fig)  # Free memory
        
        logger.info(f"✅ Streaming density plot generated successfully")
        logger.info(f"   Original: {output_file}")
        logger.info(f"   Log copy: {log_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate streaming density plot: {str(e)}")
        raise


def main() -> int:
    """
    Main execution function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Generate GTEx coefficient distribution plots using memory-efficient streaming",
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
    parser.add_argument("--chunk-size", type=int, default=100000,
                       help="Number of rows to process per chunk")
    
    # Directory arguments
    parser.add_argument("--data-dir", 
                       default="/pividori_lab/haoyu_projects/ccc-gpu/data/gtex",
                       help="Base data directory")
    parser.add_argument("--output-dir",
                       default="/pividori_lab/haoyu_projects/ccc-gpu/figures",
                       help="Base output directory")
    parser.add_argument("--log-dir", 
                       help="Custom log directory (default: ./logs/streaming_YYYYMMDD_HHMMSS)")
    
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
        log_dir = Path("logs") / f"streaming_{timestamp}"
    
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
        similarity_matrices_dir = data_dir / "similarity_matrices" / args.top_n_genes
        
        # Original output directory
        original_output_dir = (
            Path(args.output_dir) / "coefs_comp" / f"gtex_{args.tissue}"
        )
        original_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Original output directory: {original_output_dir}")
        logger.info(f"Log directory: {log_dir}")
        
        # Input file
        correlation_file = similarity_matrices_dir / (
            f"gtex_v8_data_{args.tissue}-{args.gene_selection_strategy}-all.pkl"
        )
        
        if not correlation_file.exists():
            logger.error(f"❌ Correlation file not found: {correlation_file}")
            return 1
        
        logger.info(f"✅ Correlation file: {correlation_file}")
        
        # Get data info and load
        logger.info("📊 Loading and analyzing data...")
        data_info, data = get_data_info(correlation_file)
        
        # Rename columns
        coefficient_columns = [args.ccc_label, args.pearson_label, args.spearman_label]
        original_columns = ["ccc", "pearson", "spearman"]
        
        logger.info("🔄 Renaming columns...")
        for original, new in zip(original_columns, coefficient_columns):
            if original in data.columns:
                data = data.rename(columns={original: new})
                logger.info(f"  {original} → {new}")
        
        # Verify required columns exist
        missing_columns = [col for col in coefficient_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(data.columns)}")
            return 1
        
        # Process data in chunks
        logger.info("🔄 Starting chunk-based processing...")
        histogram_result = process_data_in_chunks(data, coefficient_columns, args.chunk_size)
        
        # Clear original data from memory
        del data
        logger.info("💾 Original data cleared from memory")
        
        # Generate plots using streaming approach
        logger.info("📊 Starting plot generation...")
        
        # 1. Overlaid density plot (matches original ccc.plots.plot_histogram)
        generate_streaming_density_plot(histogram_result, original_output_dir, log_dir)
        
        # 2. Cumulative histogram
        generate_streaming_cumulative_histogram(
            histogram_result, args.gene_pairs_percent, original_output_dir, log_dir
        )
        
        # 3. Regular histograms (individual subplots)
        generate_streaming_regular_histogram(histogram_result, original_output_dir, log_dir)
        
        # Final summary
        logger.info("="*80)
        logger.info("STREAMING ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("Generated files:")
        
        # List generated files
        for output_dir_name, output_path in [("Original", original_output_dir), ("Log", log_dir)]:
            logger.info(f"\n{output_dir_name} directory ({output_path}):")
            if output_path.exists():
                for file_path in sorted(output_path.glob("*.svg")):
                    logger.info(f"  📊 {file_path.name}")
        
        logger.info(f"\n✅ All streaming plots generated successfully!")
        logger.info(f"📁 Log directory: {log_dir}")
        logger.info(f"📁 Original directory: {original_output_dir}")
        logger.info(f"💾 Memory-efficient processing completed with {args.chunk_size:,} rows per chunk")
        
        return 0
        
    except Exception as e:
        logger.error(f"Streaming analysis failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 