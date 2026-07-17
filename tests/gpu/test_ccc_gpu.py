"""GPU-vs-CPU parity for the end-to-end coefficient (numerical + categorical).

These are pure correctness tests: no timing, no speedup assertions, no log
files. Performance measurement lives in the ``ccc-gpu-bench`` CLI. GPU memory is
cleaned between tests by the autouse ``clean_gpu_memory`` fixture in conftest.
"""

import numpy as np
import pandas as pd
import pytest
from ccc.coef.impl import ccc
from ccc.coef.impl_gpu import ccc as ccc_gpu

# Parity tolerance contract (see openspec restructure-tests): GPU-vs-CPU
# end-to-end results must match within atol/rtol = 1e-6.
PARITY_ATOL = 1e-6
PARITY_RTOL = 1e-6


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
    ],
)
def test_ccc_gpu_with_numerical_input(
    seed: int,
    shape: tuple[int, int],
    contain_singletons: bool,
):
    """GPU coefficients match the CPU reference for numerical input."""
    np.random.seed(seed)
    df = np.random.rand(*shape)
    if contain_singletons:
        # Force a constant (singleton) feature to exercise the -2 marker path.
        df[0, :] = 0.0

    c_gpu = ccc_gpu(df)
    c_cpu = ccc(df, n_jobs=2)

    gpu_df = pd.DataFrame(c_gpu).astype(np.float64)
    cpu_df = pd.DataFrame(c_cpu)
    pd.testing.assert_frame_equal(gpu_df, cpu_df, atol=PARITY_ATOL, rtol=PARITY_RTOL)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize(
    "shape, n_categories, str_length",
    [
        ((10, 20), 10, 2),
        ((20, 200), 50, 3),
        ((30, 300), 200, 4),
        ((9, 10000), 500, 5),
    ],
)
def test_ccc_gpu_with_categorical_input(
    seed: int,
    shape: tuple[int, int],
    n_categories: int,
    str_length: int,
    categorical_data_generator,
):
    """GPU coefficients match the CPU reference for categorical input."""
    n_features, n_samples = shape
    df = categorical_data_generator(
        n_features, n_samples, n_categories, str_length=str_length, random_state=seed
    )
    res_cpu = ccc(df, n_jobs=2)
    res_gpu = ccc_gpu(df)

    cpu_df = pd.DataFrame(res_cpu)
    gpu_df = pd.DataFrame(res_gpu.astype(np.float64))
    pd.testing.assert_frame_equal(gpu_df, cpu_df, atol=PARITY_ATOL, rtol=PARITY_RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_ccc_gpu_large_input_parity(shape: tuple[int, int]):
    """Safety net: parity on a large grid (moved off the default fast path).

    The commented-out mega grids in the old test lived here as benchmark cases;
    the full sweeps now belong to ``ccc-gpu-bench``. This single ``slow`` case
    guards against large-input regressions in CI-excluded local runs.
    """
    np.random.seed(42)
    df = np.random.rand(*shape)

    c_gpu = ccc_gpu(df)
    c_cpu = ccc(df, n_jobs=-1)

    gpu_df = pd.DataFrame(c_gpu).astype(np.float64)
    cpu_df = pd.DataFrame(c_cpu)
    pd.testing.assert_frame_equal(gpu_df, cpu_df, atol=PARITY_ATOL, rtol=PARITY_RTOL)
