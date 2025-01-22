import numpy as np
import ccc_cuda_ext
import pytest


def test_example_return_optional_vectors():
    # Test all vectors included
    result = ccc_cuda_ext.example_return_optional_vectors(True, True, True)
    assert len(result) == 3
    assert isinstance(result, tuple)

    # Check first vector
    assert result[0] is not None
    assert isinstance(result[0], list)  # pybind11 converts std::vector to list
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    # Check second vector
    assert result[1] is not None
    assert isinstance(result[1], list)
    np.testing.assert_array_equal(result[1], [4, 5, 6])

    # Check third vector
    assert result[2] is not None
    assert isinstance(result[2], list)
    np.testing.assert_array_equal(result[2], [7.0, 8.0, 9.0])


def test_example_return_optional_vectors_partial():
    # Test with only first vector
    result = ccc_cuda_ext.example_return_optional_vectors(True, False, False)
    assert len(result) == 3

    # First vector should be present
    assert result[0] is not None
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    # Other vectors should be None
    assert result[1] is None
    assert result[2] is None


def test_example_return_optional_vectors_none():
    # Test with no vectors
    result = ccc_cuda_ext.example_return_optional_vectors(False, False, False)
    assert len(result) == 3
    assert result[0] is None
    assert result[1] is None
    assert result[2] is None


def test_example_return_optional_vectors_types():
    result = ccc_cuda_ext.example_return_optional_vectors()

    # Check types of returned vectors
    assert all(isinstance(x, float) for x in result[0])  # first vector should be floats
    assert all(isinstance(x, int) for x in result[1])  # second vector should be ints
    assert all(
        isinstance(x, float) for x in result[2]
    )  # third vector should be doubles/floats


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
    res = ccc_cuda_ext.compute_coef(parts, n_features, n_parts, n_objs)
    print(res)
    assert np.isclose(res[0][0], expected_ari, atol=1e-4)
