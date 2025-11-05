import time
import pytest
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os
import pandas as pd
from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc
from utils import clean_gpu_memory, generate_categorical_data


def setup_logging(
    seed: int,
    shape: Tuple[int, int],
    n_cpu_cores: int,
    generate_logs: bool,
) -> Optional[Dict[str, Any]]:
    """Setup logging infrastructure if logging is enabled.

    Args:
        seed: Random seed for reproducibility
        shape: Tuple of (n_features, n_samples)
        n_cpu_cores: Number of CPU cores to use
        generate_logs: Whether to generate log files

    Returns:
        Dictionary containing logging information if logging is enabled, None otherwise
    """
    if not generate_logs:
        return None

    logs_dir = os.path.join("tests", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    base_filename = f"test_ccc_gpu_{seed}_f{shape[0]}_n{shape[1]}_c{n_cpu_cores}"
    log_files = {
        "log": os.path.join(logs_dir, f"{base_filename}.log"),
        "gpu_results": os.path.join(logs_dir, f"{base_filename}_gpu_results.log"),
        "cpu_results": os.path.join(logs_dir, f"{base_filename}_cpu_results.log"),
        "input_data": os.path.join(logs_dir, f"{base_filename}_input_data.log"),
    }

    print("Writing test output to:")
    for name, path in log_files.items():
        print(f"  - {name}: {path}")

    return {"files": log_files, "log_file": open(log_files["log"], "w")}


def log_test_info(log_file, shape: Tuple[int, int], seed: int) -> None:
    """Log basic test information."""
    print(
        f"\nTesting with {shape[0]} features, {shape[1]} samples, seed {seed}",
        file=log_file,
    )


def log_performance_metrics(
    log_file, gpu_time: float, cpu_time: float, num_coefs: int
) -> None:
    """Log performance metrics."""
    speedup = cpu_time / gpu_time
    print(f"GPU time: {gpu_time:.4f} seconds", file=log_file)
    print(f"CPU time: {cpu_time:.4f} seconds", file=log_file)
    print(f"Speedup: {speedup:.2f}x", file=log_file)
    print(f"Number of coefficients: {num_coefs}", file=log_file)


def analyze_differences(
    c1: np.ndarray,
    c2: np.ndarray,
    shape: Tuple[int, int],
    log_file: Optional[Any] = None,
) -> Tuple[int, float, float, int]:
    """Analyze differences between GPU and CPU results.

    Returns:
        Tuple of (number of differences, max difference, percentage of differences, number of max differences)
    """
    not_close_mask = ~np.isclose(c1, c2, rtol=1e-3, atol=1e-3)
    not_close_indices = np.where(not_close_mask)[0]
    not_close = len(not_close_indices)

    if log_file and not_close > 0:
        log_differences(c1, c2, not_close_indices, shape, log_file)

    num_coefs = len(c1)
    not_close_percentage = not_close / num_coefs * 100
    max_diff = np.max(np.abs(c1 - c2))
    max_diff_count = np.sum(np.abs(c1 - c2) == max_diff) if not_close > 0 else 0

    if log_file:
        log_statistics(
            not_close, max_diff, max_diff_count, not_close_percentage, log_file
        )

    return not_close, max_diff, not_close_percentage, max_diff_count


def log_differences(
    c1: np.ndarray,
    c2: np.ndarray,
    not_close_indices: np.ndarray,
    shape: Tuple[int, int],
    log_file: Any,
) -> None:
    """Log detailed information about differences between results."""
    print("\nDifferences found:", file=log_file)
    print(
        "idx | GPU Result | CPU Result | Absolute Diff | Row i | Row j",
        file=log_file,
    )
    print("-" * 65, file=log_file)

    n_features = shape[0]
    for diff_count, idx in enumerate(not_close_indices, 1):
        i = int(
            np.floor(
                (2 * n_features - 1 - np.sqrt((2 * n_features - 1) ** 2 - 8 * idx)) / 2
            )
        )
        j = idx - i * n_features + (i * (i + 1)) // 2

        print(f"\n[Diff #{diff_count}]", file=log_file)
        print(
            f"{idx:3d} | {c1[idx]:10.6f} | {c2[idx]:10.6f} | {abs(c1[idx] - c2[idx]):12.6f} | {i:5d} | {j:5d}",
            file=log_file,
        )
        print("-" * 65, file=log_file)


def log_statistics(
    not_close: int,
    max_diff: float,
    max_diff_count: int,
    not_close_percentage: float,
    log_file: Any,
) -> None:
    """Log statistical information about the differences."""
    print(f"\nNumber of coefficients not close: {not_close}", file=log_file)
    print(f"Max difference: {max_diff:.6f}", file=log_file)
    print(
        f"Number of coefficients with max difference: {max_diff_count}",
        file=log_file,
    )
    print(
        f"Percentage of coefficients not close: {not_close_percentage}%",
        file=log_file,
    )


@pytest.mark.parametrize(
    "seed", [42]
)  # More seeds for simple cases, only 42 for large cases
@pytest.mark.parametrize(
    "shape, contain_singletons, max_not_close_percentage, generate_logs",
    [
        # Simple cases
        ((10, 100), False, 0.0, False),
        ((20, 200), False, 0.0, False),
        ((30, 300), False, 0.0, False),
        ((10, 100), True, 0.0, False),
        ((20, 200), True, 0.0, False),
        ((30, 300), True, 0.0, False),
        # ((100, 100), 0.0, True),
        # ((100, 1000), 0.0, True),
        # Large cases
        # ((1000, 100), 0.0, True),
        # ((100, 1000), 0.008, False), # Skipped, too slow for a unit test
        # ((1000, 1000), 0.0, False),
        # ((2000, 1000), 0.0, False),
        # ((3000, 1000), 0.0, True),
        # ((4000, 1000), 0.0, True),
        # ((5000, 100), 0.0, True),
        # ((20000, 100), 0.0, False),
        # Benchmark cases
        # ((5000, 1000), 0.0, True),
        # ((10000, 1000), 0.0, True),
        # ((500, 1000), 0.0, True),
        # ((1000, 1000), 0.0, True),
        # ((2000, 1000), 0.0, True),
        # ((4000, 1000), 0.0, True),
        # ((6000, 1000), False, 0.0, True),
        # ((12000, 1000), 0.0, True),
        # ((16000, 1000), False, 0.0, True),
        # ((20000, 1000), False, 0.0, True),
        # ((8000, 1000), 0.0, True),
        # ((12000, 1000), 0.0, True),
        # ((56200, 755), False, 0.0, True),
    ],
)
@pytest.mark.parametrize("n_cpu_cores", [24])
@clean_gpu_memory
def test_ccc_gpu_with_numerical_input(
    seed: int,
    shape: Tuple[int, int],
    contain_singletons: bool,
    n_cpu_cores: int,
    max_not_close_percentage: float,
    generate_logs: bool,
):
    """
    Test 2D CCC implementation with various data shapes and random seeds.
    Combines both simple and large test cases.

    Args:
        seed: Random seed for reproducibility
        shape: Tuple of (n_features, n_samples)
        n_cpu_cores: Number of CPU cores to use
        max_not_close_percentage: Maximum allowed percentage of coefficients that can differ
        generate_logs: Whether to generate detailed log files
    """
    # Setup logging if enabled
    logging_info = setup_logging(seed, shape, n_cpu_cores, generate_logs)
    log_file = logging_info["log_file"] if logging_info else None

    # Generate test data
    np.random.seed(seed)
    if log_file:
        log_test_info(log_file, shape, seed)
    df = np.random.rand(*shape)
    # If contain_singletons is True, set the first row to be a singleton, using value 0.0
    if contain_singletons:
        df[0, :] = 0.0

    # Time GPU version
    start_gpu = time.time()
    # c1 = ccc_gpu(df)
    # Catch exceptions
    try:
        c1 = ccc_gpu(df)
    except Exception as e:
        print(f"Error: {e}")
        raise e
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # Time CPU version
    start_cpu = time.time()
    c2 = ccc(df, n_jobs=n_cpu_cores)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # Log performance metrics
    if log_file:
        log_performance_metrics(log_file, gpu_time, cpu_time, len(c1))

    # Analyze differences
    not_close, max_diff, not_close_percentage, _ = analyze_differences(
        c1, c2, shape, log_file
    )

    # Cleanup logging
    if logging_info:
        logging_info["log_file"].close()

    # Convert to DataFrame for better assertion hints
    gpu_df = pd.DataFrame(c1)
    gpu_df = gpu_df.astype(np.float64)
    cpu_df = pd.DataFrame(c2)
    pd.testing.assert_frame_equal(gpu_df, cpu_df, atol=1e-6, rtol=1e-6)

    # Assert results using percentages. Useful if get_parts is implemented in GPU version in the future, in which
    # case the parts generated will be slightly different due to floating point precision
    # assert (
    #     not_close_percentage <= max_not_close_percentage
    # ), f"Results differ for shape={shape}, seed={seed}"


@pytest.mark.parametrize(
    "seed", [42]
)  # More seeds for simple cases, only 42 for large cases
@pytest.mark.parametrize(
    "shape, n_categories, str_length",
    [
        # Simple cases
        ((10, 20), 10, 2),
        ((20, 200), 50, 3),
        ((30, 300), 200, 4),
        ((9, 10000), 500, 5)
    ],
)
@pytest.mark.parametrize("n_cpu_cores", [48])
@clean_gpu_memory
def test_ccc_gpu_with_categorical_input(
    seed: int,
    shape: Tuple[int, int],
    n_categories: int,
    str_length: int,
    n_cpu_cores: int,
):
    n_features, n_samples = shape
    df = generate_categorical_data(
        n_features, n_samples, n_categories, str_length=str_length, random_state=seed
    )
    res_cpu = ccc(df, n_jobs=n_cpu_cores)
    res_gpu = ccc_gpu(df)

    cpu_df = pd.DataFrame(res_cpu)
    gpu_df = pd.DataFrame(res_gpu.astype(np.float64))
    pd.testing.assert_frame_equal(gpu_df, cpu_df, atol=1e-6, rtol=1e-6)

# @clean_gpu_memory
# def test_ccc_gpu_with_mixed_input():
#     return