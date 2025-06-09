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


@pytest.mark.parametrize(
    "parts, expected_ari",
    [
        (np.array([[[0, 0, 1, 1]], [[0, 0, 1, 2]]], dtype=np.int8), 0.57),
        (np.array([[[0, 0, 1, 1]], [[0, 1, 0, 1]]], dtype=np.int8), 0.0),  # -0.5
        (np.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=np.int8), 1.0),
        (np.array([[[0, 0, 1, 1]], [[1, 1, 0, 0]]], dtype=np.int8), 1.0),
        (np.array([[[0, 0, 1, 1]], [[2, 1, 2, 0]]], dtype=np.int8), 0.0),  # -0.287
        (np.array([[[0, 0, 0, 0]], [[0, 1, 2, 3]]], dtype=np.int8), 0.0),
        (np.array([[[0, 1, 0, 1]], [[1, 1, 0, 0]]], dtype=np.int8), 0.0),  # -0.5
        (np.array([[[1, 1, 0, 0]], [[0, 0, 1, 2]]], dtype=np.int8), 0.57),
    ],
)
def test_compute_coef_simple_2_1_4(parts, expected_ari):
    """
    Test case with parts of shape (2, 1, 4), 2 features, 1 part, 4 objects
    """
    n_features, n_parts, n_objs = parts.shape
    res = ccc_cuda_ext.compute_coef(parts, n_features, n_parts, n_objs)
    cm_values, cm_pvalues, max_parts = res
    assert np.isclose(cm_values[0], expected_ari, atol=1e-2)


@pytest.mark.parametrize(
    "parts, expected_ari",
    [
        (
            np.array(
                [
                    [[0, 0, 1, 1]],
                    [[0, 0, 1, 1]],
                    [[0, 1, 0, 1]],
                    [[1, 1, 0, 0]],
                ],
                dtype=np.int8,
            ),
            # np.array([1.0, -0.5, 1.0, -0.5, 1.0, -0.5]),
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ),
    ],
)
def test_compute_coef_simple_4_1_4(parts, expected_ari):
    """
    Test case with parts of shape (4, 1, 4), 4 features, 1 part, 4 objects
    """
    n_features, n_parts, n_objs = parts.shape
    res = ccc_cuda_ext.compute_coef(parts, n_features, n_parts, n_objs)
    print(res)
    cm_values, cm_pvalues, max_parts = res
    assert np.allclose(cm_values, expected_ari, atol=1e-2)


@pytest.mark.parametrize(
    "parts, expected_ari, expected_max_parts",
    [
        (
            np.array(
                [
                    [[0, 0, 1, 1], [1, 1, 0, 0]],  # Feature 0 with 2 partitions
                    [[0, 1, 0, 1], [2, 1, 2, 0]],  # Feature 1 with 2 partitions
                    [[0, 0, 1, 2], [0, 1, 0, 1]],  # Feature 2 with 2 partitions
                ],
                dtype=np.int8,
            ),
            np.array(
                [
                    0.0,  # Feature 0 vs 1 (partition 0,0) -0.287
                    0.57,  # Feature 0 vs 2 (partition 0,0)
                    1.0,  # Feature 1 vs 2 (partition 0,1)
                ]
            ),
            np.array(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],  # TODO: double check this case
                ],
                dtype=np.int8,
            ),
        ),
    ],
)
def test_compute_coef_simple_3_2_4(parts, expected_ari, expected_max_parts):
    """
    Test case with parts of shape (3, 2, 4), 3 features, 2 partitions, 4 objects
    """
    n_features, n_parts, n_objs = parts.shape
    res = ccc_cuda_ext.compute_coef(parts, n_features, n_parts, n_objs)

    cm_values, cm_pvalues, max_parts = res
    assert np.allclose(cm_values, expected_ari, atol=1e-2)

    # Check max_parts shape
    n_feature_comp = int(n_features * (n_features - 1) / 2)
    # Convert max_parts from list to numpy array
    max_parts = np.array(max_parts)
    assert max_parts.shape == (n_feature_comp, 2)

    # Check that max_parts contains valid partition indices
    assert np.all(max_parts >= 0)
    assert np.all(max_parts < n_parts)
    assert np.all(max_parts == expected_max_parts)
