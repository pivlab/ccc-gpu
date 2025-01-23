import time
import pytest
import numpy as np
from typing import Tuple, Dict, Any, Literal
from numpy.typing import NDArray

from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc
from utils import clean_gpu_memory

# Constants
ABSOLUTE_TOLERANCE = 1e-3
ALLOWED_FAILURE_RATE = 0.10
TEST_SIZES = np.linspace(100, 100000, num=5, dtype=int)
TEST_SEEDS = np.linspace(0, 1000, num=5, dtype=int)

DistributionType = Literal["rand", "randn", "randint", "exponential"]

# Test configurations
DISTRIBUTION_CONFIGS = [
    ("rand", {}),  # Uniform distribution
    ("randn", {}),  # Normal distribution
    ("randint", {"low": 0, "high": 100}),  # Integer distribution
    ("exponential", {"scale": 2.0}),  # Exponential distribution
]


@pytest.fixture
def random_features() -> Tuple[NDArray, NDArray]:
    """Fixture for generating simple random features with fixed seed."""
    np.random.seed(0)
    return np.random.rand(10), np.random.rand(10)


@clean_gpu_memory
def test_ccc_gpu_1d_simple(random_features: Tuple[NDArray, NDArray]) -> None:
    """
    Basic test to verify GPU implementation matches CPU implementation
    for a simple case with fixed random seed.
    """
    feature1, feature2 = random_features
    gpu_result = ccc_gpu(feature1, feature2)
    cpu_result = ccc(feature1, feature2)

    assert np.isclose(
        gpu_result, cpu_result, atol=ABSOLUTE_TOLERANCE
    ), f"GPU ({gpu_result:.6f}) and CPU ({cpu_result:.6f}) results differ by {abs(gpu_result - cpu_result):.6f}"


@clean_gpu_memory
def generate_random_features(
    size: int, distribution: DistributionType, params: Dict[str, Any], seed: int
) -> Tuple[NDArray, NDArray]:
    """
    Generate random features based on specified distribution and parameters.

    Args:
        size: Length of the feature arrays
        distribution: Type of random distribution to use
        params: Distribution-specific parameters
        seed: Random seed for reproducibility

    Returns:
        Tuple of two random feature arrays
    """
    np.random.seed(seed)

    if distribution == "rand":
        return np.random.rand(size), np.random.rand(size)
    elif distribution == "randn":
        return np.random.randn(size), np.random.randn(size)
    elif distribution == "randint":
        return (
            np.random.randint(params["low"], params["high"], size),
            np.random.randint(params["low"], params["high"], size),
        )
    elif distribution == "exponential":
        return (
            np.random.exponential(params["scale"], size),
            np.random.exponential(params["scale"], size),
        )
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


@clean_gpu_memory
def compare_gpu_cpu_results(
    feature1: NDArray, feature2: NDArray
) -> Tuple[bool, float, float]:
    """
    Compare CCC results between GPU and CPU implementations.

    Returns:
        Tuple of (is_close, gpu_result, cpu_result)
    """
    gpu_result = ccc_gpu(feature1, feature2)
    cpu_result = ccc(feature1, feature2)
    return (
        np.isclose(gpu_result, cpu_result, atol=ABSOLUTE_TOLERANCE),
        gpu_result,
        cpu_result,
    )


@pytest.mark.parametrize("distribution, params", DISTRIBUTION_CONFIGS)
def test_ccc_gpu_1d(distribution: DistributionType, params: Dict[str, Any]) -> None:
    """
    Comprehensive test comparing GPU and CPU implementations across different:
    - Data distributions
    - Array sizes
    - Random seeds

    Allows for a small percentage (ALLOWED_FAILURE_RATE) of individual tests to fail
    for each distribution to account for floating-point differences.
    """
    total_tests = len(TEST_SIZES) * len(TEST_SEEDS)
    max_allowed_failures = int(total_tests * ALLOWED_FAILURE_RATE)
    failures = 0

    for size in TEST_SIZES:
        for seed in TEST_SEEDS:
            feature1, feature2 = generate_random_features(
                size, distribution, params, seed
            )
            is_close, gpu_result, cpu_result = compare_gpu_cpu_results(
                feature1, feature2
            )

            if not is_close:
                failures += 1
                print(
                    f"\nTest failed for size={size}, seed={seed}, distribution={distribution}"
                )
                print(f"GPU result: {gpu_result:.6f}")
                print(f"CPU result: {cpu_result:.6f}")
                print(f"Absolute difference: {abs(gpu_result - cpu_result):.6f}")

    assert (
        failures <= max_allowed_failures
    ), f"Too many failures for {distribution} distribution: {failures} > {max_allowed_failures}"

    if failures > 0:
        print(
            f"Warning: {failures}/{total_tests} tests failed, but within "
            f"the allowed failure rate of {ALLOWED_FAILURE_RATE * 100}%"
        )


@pytest.mark.parametrize("seed", [42, 0, 57])
@pytest.mark.parametrize(
    "shape",
    [
        (10, 100),
        (20, 200),
        (30, 300),
    ],
)
@clean_gpu_memory
def test_ccc_gpu_2d_simple(seed: int, shape: Tuple[int, int]):
    """
    Test 2D CCC implementation with various data shapes and random seeds.

    Args:
        seed: Random seed for reproducibility
        shape: Tuple of (n_features, n_samples)
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
    c2 = ccc(df)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # Calculate speedup
    speedup = cpu_time / gpu_time

    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Number of coefficients: {len(c1)}")

    assert np.allclose(
        c1, c2, rtol=1e-3, atol=1e-3
    ), f"Results differ for shape={shape}, seed={seed}"
