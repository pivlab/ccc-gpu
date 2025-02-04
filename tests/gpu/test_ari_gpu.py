import pytest
import numpy as np
import ccc_cuda_ext

from ccc.sklearn.metrics import (
    adjusted_rand_index,
)


# Test cases taken from sklearn.metrics.adjusted_rand_score
@pytest.mark.parametrize(
    "parts, expected_ari",
    [
        (np.array([[[0, 0, 1, 2]], [[0, 0, 1, 1]]], dtype=np.int32), 0.57),
        (np.array([[[0, 0, 1, 1]], [[0, 1, 0, 1]]], dtype=np.int32), -0.5),
        (np.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=np.int32), 1.0),
        (np.array([[[0, 0, 1, 1]], [[1, 1, 0, 0]]], dtype=np.int32), 1.0),
        (np.array([[[0, 0, 0, 0]], [[0, 1, 2, 3]]], dtype=np.int32), 0.0),
    ],
)
def test_simple_ari_results(parts, expected_ari):
    n_features, n_parts, n_objs = parts.shape
    res = ccc_cuda_ext.ari_int32(parts, n_features, n_parts, n_objs)
    assert np.isclose(res[0], expected_ari, atol=1e-2)


def generate_pairwise_combinations(arr):
    pairs = []
    num_slices = arr.shape[0]  # Number of 2D arrays in the 3D array

    for i in range(num_slices):
        for j in range(i + 1, num_slices):  # Only consider pairs in different slices
            for row_i in arr[i]:  # Each row in slice i
                for row_j in arr[j]:  # Pairs with each row in slice j
                    pairs.append([row_i, row_j])

    # Convert list of pairs to a NumPy array
    return np.array(pairs)


# Test ari generation given a full 3D array of partitions
@pytest.mark.parametrize(
    "n_features, n_parts, n_objs, k, seed",
    [
        (2, 2, 100, 2, 42),
        (5, 10, 200, 2, 42),
        (2, 2, 1024, 2, 42),
        (200, 20, 300, 10, 42),
    ],
)
def test_pairwise_ari(n_features, n_parts, n_objs, k, seed):
    # Set random seed for reproducibility
    np.random.seed(seed)

    parts = np.random.randint(0, k, size=(n_features, n_parts, n_objs), dtype=np.int32)
    # Create test inputs
    n_feature_comp = n_features * (n_features - 1) // 2
    n_aris = n_feature_comp * n_parts * n_parts
    ref_aris = np.zeros(n_aris, dtype=np.float32)
    # Get partition pairs
    pairs = generate_pairwise_combinations(parts)
    # Use map-reduce to compute ARIs for all pairs of partitions
    for i, (part0, part1) in enumerate(pairs):
        ari = adjusted_rand_index(part0, part1)
        ref_aris[i] = ari
    # Compute ARIs using CUDA
    res_aris = ccc_cuda_ext.ari_int32(parts, n_features, n_parts, n_objs)

    # Report number of mismatches
    mismatches = np.sum(res_aris != ref_aris)
    print(f"Number of mismatches: {mismatches}")

    print(f"\nres_aris: {res_aris}, ref_aris: {ref_aris}")
    assert np.allclose(res_aris, ref_aris)
