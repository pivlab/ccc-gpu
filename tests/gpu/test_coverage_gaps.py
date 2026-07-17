"""GPU coverage for known GPU/CPU divergence points (openspec restructure-tests §5.2).

Closes the coverage gaps for:
  * constant-feature matrices (NaN parity in the max reduction),
  * too-few-objects error-path parity (both raise the same ValueError),
  * mixed numerical + categorical DataFrames.

Categorical ``return_parts`` parity (§5.1) lives in
``test_ccc_gpu_return_parts.py``.
"""

import numpy as np
import pandas as pd
import pytest
from ccc.coef.impl import ccc as ccc_cpu
from ccc.coef.impl_gpu import ccc as ccc_gpu

PARITY_ATOL = 1e-6
PARITY_RTOL = 1e-6


def test_constant_feature_matrix_nan_parity():
    """A constant (singleton) feature yields NaN for its comparisons on both
    GPU and CPU, with identical NaN patterns and matching finite values."""
    np.random.seed(0)
    data = np.random.rand(4, 100)
    data[1, :] = 3.0  # constant feature -> -2 singleton marker

    g = np.asarray(ccc_gpu(data), dtype=np.float64)
    c = np.asarray(ccc_cpu(data, n_jobs=2), dtype=np.float64)

    np.testing.assert_array_equal(np.isnan(g), np.isnan(c))
    finite = ~np.isnan(g)
    np.testing.assert_allclose(g[finite], c[finite], atol=PARITY_ATOL, rtol=PARITY_RTOL)


def test_too_few_objects_error_parity():
    """Too-few-objects raises the same ValueError on GPU and CPU."""
    np.random.seed(123)
    data = np.random.rand(10, 2)

    with pytest.raises(ValueError, match="too few objects"):
        ccc_cpu(data, internal_n_clusters=3)
    with pytest.raises(ValueError, match="too few objects"):
        ccc_gpu(data, internal_n_clusters=3)


def test_mixed_numerical_and_categorical_dataframe_parity():
    """A DataFrame mixing numerical and categorical columns matches CPU."""
    np.random.seed(123)
    numerical = np.random.rand(100)
    median = np.percentile(numerical, 50)
    categorical = np.full(numerical.shape[0], "", dtype=object)
    categorical[numerical < median] = "l"
    categorical[numerical >= median] = "u"
    other = np.random.rand(100)

    df = pd.DataFrame({"num": numerical, "cat": categorical, "num2": other})

    g = np.asarray(ccc_gpu(df), dtype=np.float64)
    c = np.asarray(ccc_cpu(df, n_jobs=2), dtype=np.float64)

    np.testing.assert_array_equal(np.isnan(g), np.isnan(c))
    finite = ~np.isnan(g)
    np.testing.assert_allclose(g[finite], c[finite], atol=PARITY_ATOL, rtol=PARITY_RTOL)
