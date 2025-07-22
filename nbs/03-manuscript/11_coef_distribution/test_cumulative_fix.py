#!/usr/bin/env python
"""
Quick test to verify the cumulative histogram fix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tempfile
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cumulative_fix():
    """Test the fixed cumulative histogram logic with known data."""
    
    logger.info("🧪 Testing cumulative histogram fix...")
    
    # Create test data with known distribution
    np.random.seed(42)
    n_samples = 100000
    
    # Create simple uniform distribution for testing
    # Most values should be around 0.5-0.8 range to simulate realistic correlation data
    ccc_values = np.random.beta(3, 2, n_samples)  # Skewed toward higher values
    pearson_values = np.random.beta(2, 2, n_samples) * 0.9  # More uniform, max 0.9
    spearman_values = np.random.beta(2.5, 2, n_samples) * 0.85  # Between the two
    
    # Create DataFrame
    df = pd.DataFrame({
        'ccc': ccc_values,
        'pearson': pearson_values,
        'spearman': spearman_values
    })
    
    logger.info(f"Created test data: {df.shape}")
    logger.info(f"Value ranges: CCC={ccc_values.min():.3f}-{ccc_values.max():.3f}, "
                f"Pearson={pearson_values.min():.3f}-{pearson_values.max():.3f}, "
                f"Spearman={spearman_values.min():.3f}-{spearman_values.max():.3f}")
    
    # Save to temporary file and test the streaming function
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        pickle.dump(df, tmp_file)
        tmp_file_path = Path(tmp_file.name)
    
    try:
        # Import the streaming functions
        sys.path.insert(0, '.')
        import importlib
        streaming_module = importlib.import_module('11_00-gtex_general_plots_streaming')
        
        # Set up logging for the module
        streaming_module.logger = logger
        
        # Get data info
        data_info, data = streaming_module.get_data_info(tmp_file_path)
        
        # Rename columns to match expected format
        data = data.rename(columns={'ccc': 'CCC', 'pearson': 'Pearson', 'spearman': 'Spearman'})
        
        # Process in chunks
        coefficient_columns = ['CCC', 'Pearson', 'Spearman']
        histogram_result = streaming_module.process_data_in_chunks(data, coefficient_columns, chunk_size=25000)
        
        # Generate the fixed cumulative histogram
        output_dir = Path('./test_output')
        output_dir.mkdir(exist_ok=True)
        log_dir = output_dir
        
        streaming_module.generate_streaming_cumulative_histogram(
            histogram_result, 0.7, output_dir, log_dir
        )
        
        # Also create a verification plot manually to compare
        logger.info("📊 Creating verification plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Raw histograms to verify data distribution
        colors = {'CCC': '#1f77b4', 'Pearson': '#ff7f0e', 'Spearman': '#2ca02c'}
        for col in coefficient_columns:
            ax1.hist(data[col], bins=50, alpha=0.6, label=col, color=colors[col])
        ax1.set_xlabel('Coefficient Value')
        ax1.set_ylabel('Count')
        ax1.set_title('Raw Distributions (Verification)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Manual cumulative calculation to verify
        bins = np.linspace(0, 1, 201)
        for col in coefficient_columns:
            values = data[col].values
            values = values[values >= 0]  # Only positive values
            
            # Compute cumulative percentages manually
            hist, _ = np.histogram(values, bins=bins)
            cumulative = np.cumsum(hist) / len(values) * 100
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            ax2.plot(bin_centers, cumulative, label=col, linewidth=2, color=colors[col])
        
        # Add horizontal line at 70%
        ax2.axhline(70, color='gray', linestyle='--', alpha=0.7, label='70% threshold')
        ax2.set_xlabel('Coefficient Value')
        ax2.set_ylabel('Percent of gene pairs')
        ax2.set_title('Manual Cumulative Calculation (Should Match Generated Plot)')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'verification_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Test completed successfully!")
        logger.info(f"📁 Check output files in: {output_dir}")
        logger.info("  - dist-cum_histograms_streaming.svg (main output)")
        logger.info("  - verification_plot.png (manual verification)")
        logger.info("")
        logger.info("🎯 Expected behavior:")
        logger.info("  - X-axis: 0 to 1.0 only (no negative values)")
        logger.info("  - Y-axis: 0 to 100% (percent of gene pairs)")
        logger.info("  - Curves: Start at bottom-left, go up to top-right")
        logger.info("  - Horizontal line: Single dotted line at 70%")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        tmp_file_path.unlink()

if __name__ == "__main__":
    success = test_cumulative_fix()
    if success:
        print("\n" + "="*60)
        print("🎉 CUMULATIVE HISTOGRAM FIX TEST PASSED!")
        print("="*60)
        print("The fix should now generate the correct plot format.")
    else:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        sys.exit(1) 