"""GPU-vs-CPU parity for ``return_parts=True`` (partitions + max_parts).

Pure correctness tests; timing/logging machinery removed (measurement lives in
``ccc-gpu-bench``). GPU memory is cleaned by the autouse fixture in conftest.
"""

import numpy as np
import pandas as pd
import pytest
from ccc.coef.impl import ccc
from ccc.coef.impl_gpu import ccc as ccc_gpu

PARITY_ATOL = 1e-6
PARITY_RTOL = 1e-6


def test_cm_return_parts_quadratic():
    # two features with a quadratic relationship
    np.random.seed(0)
    feature0 = np.array([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4])
    feature1 = np.array([10, 9, 8, 7, 6, 6, 7, 8, 9, 10])

    cm_value, max_parts, parts = ccc_gpu(
        feature0, feature1, internal_n_clusters=[2, 3], return_parts=True
    )

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
    # the set of partitions that maximize ari is k==3 for feature0, k==2 for feature1
    np.testing.assert_array_equal(max_parts, np.array([1, 0]))


def test_cm_return_parts_linear():
    # two features on 100 objects with a linear relationship
    np.random.seed(0)
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    cm_value, max_parts, parts = ccc_gpu(feature0, feature1, return_parts=True)

    assert cm_value == 1.0

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (9, 100)
    assert parts[1].shape == (9, 100)

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # k=2 for both features already yields the maximum
    np.testing.assert_array_equal(max_parts, np.array([0, 0]))


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "shape, contain_singletons",
    [
        ((10, 100), False),
        ((20, 200), False),
        ((30, 300), False),
        ((10, 100), True),
        ((20, 200), True),
        ((30, 300), True),
        ((100, 100), False),
        ((100, 1000), False),
    ],
)
def test_ccc_gpu_with_numerical_input(
    seed: int,
    shape: tuple[int, int],
    contain_singletons: bool,
):
    """GPU return_parts output matches the CPU reference (coefs + parts + max_parts)."""
    np.random.seed(seed)
    df = np.random.rand(*shape)
    if contain_singletons:
        df[0, :] = 0.0

    c_gpu, g_max_parts, g_parts = ccc_gpu(df, return_parts=True)
    c_cpu, c_max_parts, c_parts = ccc(df, n_jobs=2, return_parts=True)

    gpu_df = pd.DataFrame(c_gpu).astype(np.float64)
    cpu_df = pd.DataFrame(c_cpu)
    pd.testing.assert_frame_equal(gpu_df, cpu_df, atol=PARITY_ATOL, rtol=PARITY_RTOL)

    # Partitions must be identical (they are integer cluster labels).
    assert g_max_parts.shape == c_max_parts.shape
    assert g_parts.shape == c_parts.shape
    for i in range(len(g_parts)):
        pd.testing.assert_frame_equal(
            pd.DataFrame(g_parts[i].astype(np.int16)),
            pd.DataFrame(c_parts[i]),
            check_exact=True,
        )

    # max_parts: where GPU and CPU disagree, both selections must be valid maxima
    # of the ARI matrix (ties break differently between implementations).
    from ccc.coef.impl import cdist_parts_basic, get_coords_from_index

    for i in range(len(g_max_parts)):
        gpu_choice = g_max_parts[i]
        cpu_choice = c_max_parts[i]
        if np.array_equal(gpu_choice, cpu_choice):
            continue

        feat_i, feat_j = get_coords_from_index(shape[0], i)
        ari_matrix = cdist_parts_basic(c_parts[feat_i], c_parts[feat_j])
        max_ari = np.max(ari_matrix)

        gpu_ari = ari_matrix[gpu_choice[0], gpu_choice[1]]
        cpu_ari = ari_matrix[cpu_choice[0], cpu_choice[1]]
        assert np.abs(gpu_ari - max_ari) < 1e-8, (
            f"GPU choice at comparison {i} is not maximum: {gpu_ari} vs {max_ari}"
        )
        assert np.abs(cpu_ari - max_ari) < 1e-8, (
            f"CPU choice at comparison {i} is not maximum: {cpu_ari} vs {max_ari}"
        )


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "shape, n_categories, str_length",
    [
        ((10, 20), 10, 2),
        ((20, 200), 50, 3),
    ],
)
def test_ccc_gpu_with_categorical_input_return_parts(
    seed: int,
    shape: tuple[int, int],
    n_categories: int,
    str_length: int,
    categorical_data_generator,
):
    """Categorical ``return_parts`` on GPU matches the CPU reference.

    This path was previously commented out as known-broken; the fix-cuda-correctness
    work made it pass, so it is a real parity test now (openspec restructure-tests
    coverage gap 5.1).
    """
    n_features, n_samples = shape
    df = categorical_data_generator(
        n_features, n_samples, n_categories, str_length=str_length, random_state=seed
    )

    c_gpu, g_max_parts, g_parts = ccc_gpu(df, return_parts=True)
    c_cpu, c_max_parts, c_parts = ccc(df, n_jobs=2, return_parts=True)

    gpu_df = pd.DataFrame(np.asarray(c_gpu, dtype=np.float64))
    cpu_df = pd.DataFrame(np.asarray(c_cpu, dtype=np.float64))
    pd.testing.assert_frame_equal(gpu_df, cpu_df, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    assert g_parts.shape == c_parts.shape
    assert g_max_parts.shape == c_max_parts.shape
