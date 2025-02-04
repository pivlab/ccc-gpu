import time
import pytest
import numpy as np
from typing import Tuple

from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc
from utils import clean_gpu_memory


@pytest.mark.parametrize(
    "seed", [42, 0, 57]
)  # More seeds for simple cases, only 42 for large cases
@pytest.mark.parametrize(
    "shape,max_not_close_percentage",
    [
        # Simple cases
        ((10, 100), 0.6),
        ((20, 200), 0.6),
        ((30, 300), 0.6),
        # Large cases
        ((5, 10000), 0.008),
        # ((100, 1000), 0.008), # Skipped, too slow for a unit test
        # ((5000, 1000), 0.008), # Skipped, too slow for a unit test
    ],
)
@pytest.mark.parametrize("n_cpu_cores", [24, 48])
@clean_gpu_memory
def test_ccc_gpu_2d(
    seed: int, shape: Tuple[int, int], n_cpu_cores: int, max_not_close_percentage: float
):
    """
    Test 2D CCC implementation with various data shapes and random seeds.
    Combines both simple and large test cases.

    Args:
        seed: Random seed for reproducibility
        shape: Tuple of (n_features, n_samples)
        n_cpu_cores: Number of CPU cores to use
        max_not_close_percentage: Maximum allowed percentage of coefficients that can differ
    """
    np.random.seed(seed)
    print(f"\nTesting with {shape[0]} features, {shape[1]} samples, seed {seed}")
    df = np.random.rand(*shape)

    # Time GPU version
    start_gpu = time.time()
    c1 = ccc_gpu(df)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # Time CPU version
    start_cpu = time.time()
    c2 = ccc(df, n_jobs=n_cpu_cores)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # Calculate speedup
    speedup = cpu_time / gpu_time

    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    num_coefs = len(c1)
    print(f"Number of coefficients: {num_coefs}")

    # Report how many coefficients are not close
    not_close = np.sum(~np.isclose(c1, c2, rtol=1e-3, atol=1e-3))
    print(f"Number of coefficients not close: {not_close}")
    # Report the max difference
    max_diff = np.max(np.abs(c1 - c2))
    print(f"Max difference: {max_diff:.6f}")
    # Report the number of results that max_diff is 1.0
    max_diff_count = np.sum(np.abs(c1 - c2) == max_diff)
    print(f"Number of coefficients with max difference: {max_diff_count}")
    # Report the percentage of coefficients that are not close
    not_close_percentage = not_close / num_coefs * 100
    print(f"Percentage of coefficients not close: {not_close_percentage}%")

    assert (
        not_close_percentage <= max_not_close_percentage
    ), f"Results differ for shape={shape}, seed={seed}"
