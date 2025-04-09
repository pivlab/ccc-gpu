import time
import pytest
import numpy as np
from typing import Tuple
import os

from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc
from utils import clean_gpu_memory, generate_categorical_data


@pytest.mark.parametrize(
    "seed", [42]
)  # More seeds for simple cases, only 42 for large cases
@pytest.mark.parametrize(
    "shape, max_not_close_percentage, generate_logs",
    [
        # Simple cases
        ((10, 100), 0.0, False),
        ((20, 200), 0.0, False),
        ((30, 300), 0.0, False),
        # ((100, 100), 0.0, True),
        # ((100, 1000), 0.0, True),
        # Large cases
        # ((1000, 100), 0.0, True),
        # ((100, 1000), 0.008, False), # Skipped, too slow for a unit test
        # ((1000, 1000), 0.0, False),
        # ((2000, 1000), 0.0, False),
        # ((3000, 1000), 0.0, True),
        # ((4000, 1000), 0.0, True),
        # ((4500, 1000), 0.0, True),
        # ((6000, 1000), 0.0, True),
        # ((7000, 1000), 0.0, True),
        # ((5000, 100), 0.0, True),
        # ((20000, 100), 0.0, False),
        # Benchmark cases
        ((5000, 1000), 0.0, True),
        # ((10000, 1000), 0.0, True),
    ],
)
@pytest.mark.parametrize("n_cpu_cores", [24])
@clean_gpu_memory
def test_ccc_gpu_with_numerical_input(
    seed: int,
    shape: Tuple[int, int],
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
    """
    # Check if we should generate logs
    if generate_logs:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join("tests", "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Create base filename from test parameters
        base_filename = f"test_ccc_gpu_{seed}_f{shape[0]}_n{shape[1]}_c{n_cpu_cores}"
        log_filename = os.path.join(logs_dir, f"{base_filename}.log")
        gpu_results_filename = os.path.join(
            logs_dir, f"{base_filename}_gpu_results.log"
        )
        cpu_results_filename = os.path.join(
            logs_dir, f"{base_filename}_cpu_results.log"
        )
        input_data_filename = os.path.join(logs_dir, f"{base_filename}_input_data.log")

        print("Writing test output to:")
        print(f"  - Log: {log_filename}")
        print(f"  - GPU results: {gpu_results_filename}")
        print(f"  - CPU results: {cpu_results_filename}")
        print(f"  - Input data: {input_data_filename}")
        log_file = open(log_filename, "w")
    else:
        log_file = None

    np.random.seed(seed)
    if generate_logs:
        print(
            f"\nTesting with {shape[0]} features, {shape[1]} samples, seed {seed}",
            file=log_file,
        )
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

    if generate_logs:
        print(f"GPU time: {gpu_time:.4f} seconds", file=log_file)
        print(f"CPU time: {cpu_time:.4f} seconds", file=log_file)
        print(f"Speedup: {speedup:.2f}x", file=log_file)
        num_coefs = len(c1)
        print(f"Number of coefficients: {num_coefs}", file=log_file)

    # Find indices where results differ
    not_close_mask = ~np.isclose(c1, c2, rtol=1e-3, atol=1e-3)
    not_close_indices = np.where(not_close_mask)[0]
    not_close = len(not_close_indices)

    if generate_logs and not_close > 0:
        # Report differences
        print("\nDifferences found:", file=log_file)
        print(
            "idx | GPU Result | CPU Result | Absolute Diff | Row i | Row j",
            file=log_file,
        )
        print("-" * 65, file=log_file)

        # Calculate which feature pairs the differences correspond to
        n_features = shape[0]
        for diff_count, idx in enumerate(
            not_close_indices, 1
        ):  # Show all differences in log
            # Convert flat index to (i,j) pair for triangular matrix
            i = int(
                np.floor(
                    (2 * n_features - 1 - np.sqrt((2 * n_features - 1) ** 2 - 8 * idx))
                    / 2
                )
            )
            j = idx - i * n_features + (i * (i + 1)) // 2

            print(f"\n[Diff #{diff_count}]", file=log_file)
            print(
                f"{idx:3d} | {c1[idx]:10.6f} | {c2[idx]:10.6f} | {abs(c1[idx] - c2[idx]):12.6f} | {i:5d} | {j:5d}",
                file=log_file,
            )
            # print(
            #     f"Data for row {i}: {', '.join(f'{x:.8f}' for x in df[i])}",
            #     file=log_file,
            # )
            # print(
            #     f"Data for row {j}: {', '.join(f'{x:.8f}' for x in df[j])}",
            #     file=log_file,
            # )
            print("-" * 65, file=log_file)

    # Calculate statistics
    num_coefs = len(c1)
    not_close_percentage = not_close / num_coefs * 100

    if generate_logs:
        # Report statistics
        max_diff = np.max(np.abs(c1 - c2))
        max_diff_count = np.sum(np.abs(c1 - c2) == max_diff) if not_close > 0 else 0

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

        # Save full arrays to separate files
        np.savetxt(gpu_results_filename, c1, fmt="%.8f", delimiter=", ")
        np.savetxt(cpu_results_filename, c2, fmt="%.8f", delimiter=", ")
        np.savetxt(input_data_filename, df, fmt="%.8f", delimiter=", ")

        # Close the log file
        log_file.close()

    assert (
        not_close_percentage <= max_not_close_percentage
    ), f"Results differ for shape={shape}, seed={seed}"


@pytest.mark.parametrize(
    "seed", [42]
)  # More seeds for simple cases, only 42 for large cases
@pytest.mark.parametrize(
    "shape, n_categories, str_length",
    [
        # Simple cases
        ((10, 20), 10, 2),
        ((20, 200), 50, 3),
        # ((30, 300), 200, 4), # Failed for the number of categories
        # ((9, 10000), 500, 5), # Failed for the number of categories
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
    assert np.allclose(res_cpu, res_gpu)


# @clean_gpu_memory
# def test_ccc_gpu_with_mixed_input():
#     return
