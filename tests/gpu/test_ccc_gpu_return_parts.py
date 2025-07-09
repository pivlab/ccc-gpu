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


# Original CCC test in tests/test_coef.py
def test_cm_return_parts_quadratic():
    # Prepare
    np.random.seed(0)

    # two features with a quadratic relationship
    feature0 = np.array([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4])
    feature1 = np.array([10, 9, 8, 7, 6, 6, 7, 8, 9, 10])

    # Run
    cm_value, max_parts, parts = ccc_gpu(
        feature0, feature1, internal_n_clusters=[2, 3], return_parts=True
    )

    # Validate
    assert np.isclose(round(cm_value, 2), 0.31)

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (2, 10)
    assert len(np.unique(parts[0][0])) == 2
    assert len(np.unique(parts[0][1])) == 3
    assert parts[1].shape == (2, 10)
    assert len(np.unique(parts[1][0])) == 2
    assert len(np.unique(parts[1][1])) == 3

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # the set of partitions that maximize ari is:
    #   - k == 3 for feature0
    #   - k == 2 for feature1
    np.testing.assert_array_equal(max_parts, np.array([1, 0]))
    

def test_cm_return_parts_linear():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    # Run
    cm_value, max_parts, parts = ccc_gpu(feature0, feature1, return_parts=True)

    # Validate
    assert cm_value == 1.0

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (9, 100)
    assert parts[1].shape == (9, 100)

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # even in this test we do not specify internal_n_clusters (so it goes from
    # k=2 to k=10, nine partitions), k=2 for both features should already have
    # the maximum value
    np.testing.assert_array_equal(max_parts, np.array([0, 0]))


@pytest.mark.parametrize(
    "seed", [42]
)  # More seeds for simple cases, only 42 for large cases
@pytest.mark.parametrize(
    "shape, contain_singletons, generate_logs",
    [
        # Simple cases
        ((10, 100), False, False),
        ((20, 200), False, False),
        ((30, 300), False, False),
        ((10, 100), True, False),
        ((20, 200), True, False),
        ((30, 300), True, False),
        ((100, 100), False, False),
        ((100, 1000), False, False),
        # # Large cases
        # ((100, 2000), False, False),
        # ((2000, 1000), False, False),
    ],
)
@pytest.mark.parametrize("n_cpu_cores", [24])
@clean_gpu_memory
def test_ccc_gpu_with_numerical_input(
    seed: int,
    shape: Tuple[int, int],
    contain_singletons: bool,
    n_cpu_cores: int,
    generate_logs: bool,
):
    """
    Test 2D CCC implementation with various data shapes and random seeds.
    Combines both simple and large test cases.

    Args:
        seed: Random seed for reproducibility
        shape: Tuple of (n_features, n_samples)
        n_cpu_cores: Number of CPU cores to use
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
        c1, g_max_parts, g_parts = ccc_gpu(df, return_parts=True)
    except Exception as e:
        print(f"Error: {e}")
        raise e
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # Time CPU version
    start_cpu = time.time()
    c2, c_max_parts, c_parts = ccc(df, n_jobs=n_cpu_cores, return_parts=True)
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

    # Check if return_parts is correctly implemented
    assert g_max_parts.shape == c_max_parts.shape
    assert g_parts.shape == c_parts.shape
    
    for i in range(len(g_parts)):
        pd.testing.assert_frame_equal(pd.DataFrame(g_parts[i].astype(np.int16)), pd.DataFrame(c_parts[i]), check_exact=True)
    
    # Validate max_parts: Both GPU and CPU choices should represent valid maxima
    # This handles tie-breaking differences between implementations
    for i in range(len(g_max_parts)):
        gpu_parts = g_max_parts[i]
        cpu_parts = c_max_parts[i]
        
        # If they match exactly, no need for further validation
        if np.array_equal(gpu_parts, cpu_parts):
            continue
            
        # For mismatches, verify both choices are valid maxima
        # We need to compute the ARI matrix to check this
        from ccc.coef.impl import cdist_parts_basic, get_coords_from_index
        
        # Get feature indices for this comparison
        feat_i, feat_j = get_coords_from_index(shape[0], i)
        
        # Compute ARI matrix for this feature pair
        ari_matrix = cdist_parts_basic(c_parts[feat_i], c_parts[feat_j])
        max_ari = np.max(ari_matrix)
        
        # Check GPU choice
        gpu_ari = ari_matrix[gpu_parts[0], gpu_parts[1]]
        # Check CPU choice  
        cpu_ari = ari_matrix[cpu_parts[0], cpu_parts[1]]
        
        # Both should be maximum values (within floating point tolerance)
        assert np.abs(gpu_ari - max_ari) < 1e-8, f"GPU choice at comparison {i} is not maximum: {gpu_ari} vs {max_ari}"
        assert np.abs(cpu_ari - max_ari) < 1e-8, f"CPU choice at comparison {i} is not maximum: {cpu_ari} vs {max_ari}"
        
        # Print info about the tie for debugging (optional)
        ties = np.where(np.abs(ari_matrix - max_ari) < 1e-8)
        num_ties = len(ties[0])
        if num_ties > 1:
            print(f"Tie detected at comparison {i} (features {feat_i} vs {feat_j}): {num_ties} positions with ARI={max_ari:.10f}")
            print(f"  GPU chose: {tuple(gpu_parts)}, CPU chose: {tuple(cpu_parts)}")
    
    

# @pytest.mark.parametrize(
#     "seed", [42]
# )  # More seeds for simple cases, only 42 for large cases
# @pytest.mark.parametrize(
#     "shape, n_categories, str_length",
#     [
#         # Simple cases
#         ((10, 20), 10, 2),
#         ((20, 200), 50, 3),
#         # ((30, 300), 200, 4), # Failed for the number of categories
#         # ((9, 10000), 500, 5), # Failed for the number of categories
#     ],
# )
# @pytest.mark.parametrize("n_cpu_cores", [48])
# @clean_gpu_memory
# def test_ccc_gpu_with_categorical_input(
#     seed: int,
#     shape: Tuple[int, int],
#     n_categories: int,
#     str_length: int,
#     n_cpu_cores: int,
# ):
#     n_features, n_samples = shape
#     df = generate_categorical_data(
#         n_features, n_samples, n_categories, str_length=str_length, random_state=seed
#     )
#     res_cpu = ccc(df, n_jobs=n_cpu_cores)
#     res_gpu = ccc_gpu(df)
#     assert np.allclose(res_cpu, res_gpu)

