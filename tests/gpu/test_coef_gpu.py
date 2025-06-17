import numpy as np
import pandas as pd
import ccc_cuda_ext
import pytest
from ccc.coef.impl_gpu import ccc as ccc_gpu


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
    assert np.isclose(np.round(cm_value, 2), 0.31)

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
    assert np.isclose(cm_value, 1.0)

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


def test_cm_return_parts_categorical_variable():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)
    numerical_feature0_median = np.percentile(numerical_feature0, 50)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.str_)
    categorical_feature1[numerical_feature0 < numerical_feature0_median] = "l"
    categorical_feature1[numerical_feature0 >= numerical_feature0_median] = "u"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    # Run
    cm_value, max_parts, parts = ccc_gpu(
        numerical_feature0, categorical_feature1, return_parts=True
    )

    # Validate
    assert cm_value == 1.0

    assert parts is not None
    assert len(parts) == 2

    # for numerical_feature0 all partititions should be there
    assert parts[0].shape == (9, 100)
    assert set(range(2, 10 + 1)) == set(map(lambda x: np.unique(x).shape[0], parts[0]))

    # for categorical_feature1 only the first partition is meaningful
    assert parts[1].shape == (9, 100)
    assert np.unique(parts[1][0, :]).shape[0] == 2
    _unique_in_rest = np.unique(parts[1][1:, :])
    assert _unique_in_rest.shape[0] == 1
    assert _unique_in_rest[0] == -1

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # even in this test we do not specify internal_n_clusters (so it goes from
    # k=2 to k=10, nine partitions), k=2 for both features should already have
    # the maximum value
    np.testing.assert_array_equal(max_parts, np.array([0, 0]))


def test_cm_return_parts_with_matrix_as_input():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0
    X = pd.DataFrame(
        {
            "feature0": feature0,
            "feature1": feature1,
        }
    )

    # Run
    cm_value, max_parts, parts = ccc_gpu(X, return_parts=True)

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
    # the maximum value (because the relationship is linear)
    np.testing.assert_array_equal(max_parts, np.array([0, 0]))
