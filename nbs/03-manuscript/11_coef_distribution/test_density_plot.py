#!/usr/bin/env python
"""
Test script to verify the new density plot function works correctly
"""

import numpy as np
import pandas as pd
import pickle
import tempfile
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_density_plot():
    """Test the new density plot functionality."""
    
    logger.info("🧪 Testing density plot function...")
    
    # Create realistic test data 
    np.random.seed(42)
    n_samples = 100000
    
    # Create data that mimics the distribution shown in the image
    # CCC: high peak near 0, long tail
    ccc_values = np.random.exponential(0.15, n_samples)
    ccc_values = np.clip(ccc_values, 0.01, 1.0)
    
    # Pearson: broader distribution, peak around 0.2-0.3
    pearson_values = np.random.beta(1.5, 4, n_samples) * 0.95 + 0.05
    pearson_values = np.clip(pearson_values, 0.0, 1.0)
    
    # Spearman: similar to Pearson but slightly different shape
    spearman_values = np.random.beta(1.8, 3.5, n_samples) * 0.9 + 0.05
    spearman_values = np.clip(spearman_values, 0.0, 1.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ccc': ccc_values,
        'pearson': pearson_values,
        'spearman': spearman_values
    })
    
    logger.info(f"Created test data: {df.shape}")
    logger.info(f"Value ranges:")
    for col in ['ccc', 'pearson', 'spearman']:
        vals = df[col]
        logger.info(f"  {col}: {vals.min():.3f} to {vals.max():.3f} (mean: {vals.mean():.3f})")
    
    # Save to temporary file
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
        
        # Load and process data
        logger.info("📊 Loading and processing data...")
        data_info, data = streaming_module.get_data_info(tmp_file_path)
        
        # Rename columns to match expected format
        data = data.rename(columns={'ccc': 'CCC', 'pearson': 'Pearson', 'spearman': 'Spearman'})
        
        # Process in chunks
        coefficient_columns = ['CCC', 'Pearson', 'Spearman']
        histogram_result = streaming_module.process_data_in_chunks(data, coefficient_columns, chunk_size=25000)
        
        # Create output directory
        output_dir = Path('./test_density_output')
        output_dir.mkdir(exist_ok=True)
        log_dir = output_dir
        
        # Test the new density plot function
        logger.info("📊 Testing new density plot function...")
        streaming_module.generate_streaming_density_plot(histogram_result, output_dir, log_dir)
        
        # Also test all other functions for comparison
        logger.info("📊 Testing cumulative histogram...")
        streaming_module.generate_streaming_cumulative_histogram(
            histogram_result, 0.7, output_dir, log_dir
        )
        
        logger.info("📊 Testing regular histograms...")
        streaming_module.generate_streaming_regular_histogram(histogram_result, output_dir, log_dir)
        
        # Verify output files exist
        expected_files = [
            'dist-histograms.svg',  # New overlaid density plot
            'dist-cum_histograms_streaming.svg',  # Cumulative plot
            'dist-histograms_streaming.svg'  # Regular bar plots
        ]
        
        logger.info("\n📊 Generated files:")
        all_generated = True
        for filename in expected_files:
            file_path = output_dir / filename
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                logger.info(f"  ✅ {filename} ({size_kb:.1f} KB)")
            else:
                logger.error(f"  ❌ {filename} - NOT FOUND")
                all_generated = False
        
        logger.info("")
        logger.info("🎯 Expected plot characteristics:")
        logger.info("📈 Overlaid density plot (dist-histograms.svg):")
        logger.info("  - Three smooth overlaid curves (CCC, Pearson, Spearman)")
        logger.info("  - CCC should have highest peak near 0")
        logger.info("  - X-axis: 0.0 to 1.0 only")
        logger.info("  - Y-axis: Density values")
        logger.info("  - Clean legend in upper right")
        
        return all_generated
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        tmp_file_path.unlink()

if __name__ == "__main__":
    success = test_density_plot()
    if success:
        print("\n" + "="*60)
        print("🎉 DENSITY PLOT TEST COMPLETED!")
        print("="*60)
        print("Check the generated plots in ./test_density_output/ directory")
        print("The overlaid density plot should match the style from the original image")
    else:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        sys.exit(1) 