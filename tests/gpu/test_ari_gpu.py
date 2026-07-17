"""Kernel-level ARI parity: ``ccc_cuda_ext.ari_int32`` vs the CPU reference.

The throughput/benchmark variant of the pairwise test was removed; its
measurement intent now lives in ``ccc-gpu-bench ari``.
"""

import ccc_cuda_ext
import numpy as np
import pytest
from ccc.sklearn.metrics import adjusted_rand_index


# Expected values are 2-decimal literals from the sklearn.metrics.adjusted_rand_score
# documentation, so a 1e-2 tolerance is used here (the exact kernel-vs-reference
# parity at 1e-5 is covered by test_pairwise_ari below).
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
    num_slices = arr.shape[0]
    for i in range(num_slices):
        for j in range(i + 1, num_slices):
            for row_i in arr[i]:
                for row_j in arr[j]:
                    pairs.append([row_i, row_j])
    return np.array(pairs)


# Test ARI generation given a full 3D array of partitions against the CPU reference.
@pytest.mark.parametrize(
    "n_features, n_parts, n_objs, k, seed",
    [
        (2, 2, 100, 2, 42),
        (5, 10, 200, 2, 42),
        (2, 2, 1024, 2, 42),
        (50, 10, 2048, 10, 42),
    ],
)
def test_pairwise_ari(n_features, n_parts, n_objs, k, seed):
    np.random.seed(seed)

    parts = np.random.randint(0, k, size=(n_features, n_parts, n_objs), dtype=np.int32)
    n_feature_comp = n_features * (n_features - 1) // 2
    n_aris = n_feature_comp * n_parts * n_parts
    ref_aris = np.zeros(n_aris, dtype=np.float32)
    pairs = generate_pairwise_combinations(parts)

    for i, (part0, part1) in enumerate(pairs):
        ref_aris[i] = adjusted_rand_index(part0, part1)

    res_aris = ccc_cuda_ext.ari_int32(parts, n_features, n_parts, n_objs)
    assert np.allclose(res_aris, ref_aris)
