import pandas as pd
import numpy as np
import pytest
from scipy.spatial.distance import squareform
from ccc.coef.impl import ccc
from ccc.coef.impl_gpu import ccc as ccc_gpu
from utils import clean_gpu_memory


@pytest.fixture
def titanic_data():
    """Load and preprocess Titanic dataset."""
    try:
        titanic_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/titanic.csv"
        df = pd.read_csv(titanic_url)
        # Drop rows with missing 'embarked' and columns with any missing values
        df = df.dropna(subset=["embarked"]).dropna(axis=1)
        return df
    except Exception as e:
        pytest.fail(f"Failed to load Titanic dataset: {str(e)}")


def print_correlation_matrix(correlations, title):
    """Print correlation matrix in a readable format with 9 elements per row."""
    print(f"\n{title}:")
    values = correlations.flatten()
    for i in range(0, len(values), 9):
        row = values[i : i + 9]
        print(" ".join(f"{x:8.4f}" for x in row))


@clean_gpu_memory
def test_ccc_gpu_with_titanic_dataset(titanic_data):
    """
    Test the CCC (Categorical Correlation Coefficient) computation on the Titanic dataset.

    This test verifies that the GPU implementation produces results that match the CPU implementation.
    The test uses the Titanic dataset as a real-world example with mixed categorical and numerical features.

    Args:
        titanic_data (pd.DataFrame): Preprocessed Titanic dataset fixture
    """
    # Compute correlations using both implementations
    try:
        # CPU implementation
        cpu_corrs = squareform(ccc(titanic_data))
        np.fill_diagonal(cpu_corrs, 1.0)

        # GPU implementation
        gpu_corrs = squareform(ccc_gpu(titanic_data))
        np.fill_diagonal(gpu_corrs, 1.0)
    except Exception as e:
        pytest.fail(f"Failed to compute correlations: {str(e)}")

    # Print results for debugging
    print_correlation_matrix(cpu_corrs, "CPU correlations")
    print_correlation_matrix(gpu_corrs, "GPU correlations")

    # Verify results
    try:
        # Use numpy testing instead of pandas since these are numpy arrays
        np.testing.assert_allclose(cpu_corrs, gpu_corrs, rtol=1e-5, atol=1e-6,
                                  equal_nan=True, 
                                  err_msg="CPU and GPU correlation matrices should be nearly identical")
    except AssertionError as e:
        # Print more detailed error information
        print("\nDetailed comparison:")
        print("Maximum absolute difference:", np.nanmax(np.abs(cpu_corrs - gpu_corrs)))
        print("Mean absolute difference:", np.nanmean(np.abs(cpu_corrs - gpu_corrs)))
        
        # Check NaN patterns
        cpu_nan_mask = np.isnan(cpu_corrs)
        gpu_nan_mask = np.isnan(gpu_corrs)
        print(f"CPU NaN count: {np.sum(cpu_nan_mask)}")
        print(f"GPU NaN count: {np.sum(gpu_nan_mask)}")
        
        if not np.array_equal(cpu_nan_mask, gpu_nan_mask):
            print("ERROR: NaN patterns differ between CPU and GPU!")
            print("This indicates categorical data processing issues in GPU implementation")
        
        raise e
