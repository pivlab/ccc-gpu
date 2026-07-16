"""
Regression tests for the CUDA correctness fixes (openspec change
`fix-cuda-correctness`).

Covered behaviors:
  * Input-shape / partition-count validation raises a Python ``ValueError``
    (not a crash) at the pybind boundary, and the interpreter keeps running.
  * The permutation (p-value) path is correct for partitions with more than 16
    clusters -- the old ``MAX_CLUSTERS = 16`` cliff silently produced ARI 0.0 for
    every permutation, biasing p-values. Values are compared against the CPU
    reference implementation within tolerance.
  * Categorical-feature p-values are consistent with the CPU permutation scheme
    (categorical marker -> ARI 0.0, same clamp/reduction).
  * Singleton (constant) features yield NaN p-values on both GPU and CPU.

Note: several deeper correctness invariants (64-bit indexing when
n_feature_comp * n_perms > 2**32, and true CUDA-runtime error -> RuntimeError)
are covered by construction/inspection rather than at runtime because forcing
them needs either >2**32-element allocations or an unsafe GPU fault; see the
change's HANDOFF notes.
"""

import numpy as np
import pandas as pd
import pytest

import ccc_cuda_ext
from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc as ccc_cpu
from utils import clean_gpu_memory


# ---------------------------------------------------------------------------
# 1. Input validation -> Python exception, no interpreter crash
# ---------------------------------------------------------------------------


@clean_gpu_memory
def test_compute_coef_shape_mismatch_raises_value_error():
    # A valid int16 partitions array of shape (2, 1, 4)...
    parts = np.array([[[0, 0, 1, 1]], [[0, 0, 1, 2]]], dtype=np.int16)

    # ...but scalar dims that disagree with the array shape must be rejected.
    with pytest.raises(ValueError):
        ccc_cuda_ext.compute_coef(parts, 5, 1, 4)  # n_features=5 != shape[0]=2
    with pytest.raises(ValueError):
        ccc_cuda_ext.compute_coef(parts, 2, 3, 4)  # n_partitions=3 != shape[1]=1
    with pytest.raises(ValueError):
        ccc_cuda_ext.compute_coef(parts, 2, 1, 8)  # n_objs=8 != shape[2]=4


@clean_gpu_memory
def test_compute_coef_zero_partitions_raises_value_error():
    parts = np.zeros((2, 0, 4), dtype=np.int16)
    with pytest.raises(ValueError):
        ccc_cuda_ext.compute_coef(parts, 2, 0, 4)


@clean_gpu_memory
def test_compute_coef_too_many_partitions_raises_value_error():
    # max_parts stores partition indices as uint8, so n_partitions must be <= 255.
    parts = np.zeros((2, 256, 4), dtype=np.int16)
    with pytest.raises(ValueError):
        ccc_cuda_ext.compute_coef(parts, 2, 256, 4)


@clean_gpu_memory
def test_invalid_input_does_not_crash_interpreter():
    """A rejected call raises a Python exception and the process keeps working:
    a subsequent valid computation still succeeds."""
    parts = np.array([[[0, 0, 1, 1]], [[0, 0, 1, 2]]], dtype=np.int16)
    with pytest.raises(ValueError):
        ccc_cuda_ext.compute_coef(parts, 99, 1, 4)

    # Interpreter is alive -> a valid call returns a sensible result.
    cm_values, _, _ = ccc_cuda_ext.compute_coef(parts, 2, 1, 4)
    assert np.isfinite(cm_values[0])


# ---------------------------------------------------------------------------
# 2. p-values for > 16 clusters vs CPU reference (silent-zero cliff removed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_list", [[20], [10, 20], [17, 24]])
@clean_gpu_memory
def test_pvalue_more_than_16_clusters_matches_cpu(k_list):
    """With > 16 clusters the GPU permutation path used to silently return
    ARI 0.0 for every permutation (MAX_CLUSTERS = 16), producing biased
    p-values. It must now match the CPU reference within tolerance."""
    rs = np.random.RandomState(7)
    n = 200
    x = rs.rand(n)
    # Weak/moderate relationship: observed CCC is positive but the null exceeds
    # it often enough that the correct p-value is clearly non-tiny. The old bug
    # would drive the GPU p-value to ~1/(n_perms+1).
    y = 0.35 * x + 0.65 * rs.rand(n)
    n_perms = 300

    gpu_ccc, gpu_p = ccc_gpu(x, y, internal_n_clusters=k_list, pvalue_n_perms=n_perms)
    cpu_ccc, cpu_p = ccc_cpu(x, y, internal_n_clusters=k_list, pvalue_n_perms=n_perms)

    # Observed coefficients are computed correctly regardless of the bug.
    assert gpu_ccc == pytest.approx(cpu_ccc, abs=0.03)

    # Ground truth from the CPU permutation scheme.
    assert not np.isnan(cpu_p)
    assert not np.isnan(gpu_p)

    # Regression guard: the fixed GPU p-value tracks the CPU one. The silent
    # >16-cluster bug would make gpu_p ~ 1/(n_perms+1) while cpu_p is moderate.
    if cpu_p > 0.1:
        assert gpu_p > 0.05, (
            f"GPU p-value {gpu_p} is spuriously tiny for k={k_list} "
            f"(CPU={cpu_p}); the >16-cluster silent-zero bug is back"
        )
    assert gpu_p == pytest.approx(cpu_p, abs=0.2)


# ---------------------------------------------------------------------------
# 3. Categorical feature p-value consistency vs CPU permutation scheme
# ---------------------------------------------------------------------------


@clean_gpu_memory
def test_pvalue_categorical_feature_consistency_vs_cpu():
    """A DataFrame mixing a numerical and categorical features exercises the
    categorical-marker (-1 -> ARI 0.0) semantics in the permutation null."""
    rs = np.random.RandomState(3)
    n = 120
    num = rs.rand(n)
    # Categorical feature correlated with `num` (binned into 4 categories).
    cat_related = pd.cut(num, bins=4, labels=["a", "b", "c", "d"]).astype(str)
    # Independent categorical feature.
    cat_random = rs.randint(0, 4, n).astype(str)

    df = pd.DataFrame({"num": num, "cat_rel": cat_related, "cat_rnd": cat_random})
    n_perms = 300

    gpu_ccc, gpu_p = ccc_gpu(df, pvalue_n_perms=n_perms)
    cpu_ccc, cpu_p = ccc_cpu(df, pvalue_n_perms=n_perms)

    gpu_ccc = np.atleast_1d(gpu_ccc)
    cpu_ccc = np.atleast_1d(cpu_ccc)
    gpu_p = np.atleast_1d(gpu_p)
    cpu_p = np.atleast_1d(cpu_p)

    # Coefficients should be essentially identical.
    np.testing.assert_allclose(gpu_ccc, cpu_ccc, rtol=1e-4, atol=1e-3)

    # NaN patterns must agree.
    np.testing.assert_array_equal(np.isnan(gpu_p), np.isnan(cpu_p))

    valid = ~np.isnan(gpu_p)
    # p-values differ only by RNG/permutation ordering: they must stay close.
    assert np.all(np.abs(gpu_p[valid] - cpu_p[valid]) < 0.25), (
        f"GPU p-values {gpu_p} inconsistent with CPU {cpu_p} for categorical data"
    )


# ---------------------------------------------------------------------------
# 4. Singleton (constant) feature -> NaN p-value on both GPU and CPU
# ---------------------------------------------------------------------------


@clean_gpu_memory
def test_pvalue_singleton_feature_nan_consistency_vs_cpu():
    rs = np.random.RandomState(11)
    n = 100
    f0 = rs.rand(n)
    f1 = rs.rand(n)
    f2 = np.full(n, 5.0)  # constant -> singleton partitions (-2 marker)

    data = np.array([f0, f1, f2])
    n_perms = 100

    gpu_ccc, gpu_p = ccc_gpu(data, pvalue_n_perms=n_perms)
    cpu_ccc, cpu_p = ccc_cpu(data, pvalue_n_perms=n_perms)

    # NaN patterns for both coefficients and p-values must match the CPU exactly.
    np.testing.assert_array_equal(np.isnan(gpu_ccc), np.isnan(cpu_ccc))
    np.testing.assert_array_equal(np.isnan(gpu_p), np.isnan(cpu_p))

    # Comparisons that involve the constant feature (indices 1 and 2 in the
    # condensed 3-pair array) must be NaN.
    assert np.isnan(gpu_p[1]) and np.isnan(gpu_p[2])
    # The f0-f1 comparison is well defined and non-NaN.
    assert not np.isnan(gpu_p[0])
