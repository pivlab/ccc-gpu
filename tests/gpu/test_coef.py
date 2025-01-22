import numpy as np
from numpy.typing import NDArray
from numba import njit
from typing import Iterable
from ccc_cuda_ext import example_return_optional_vectors, compute_coef
from ccc.scipy.stats import rank


def test_example_return_optional_vectors():
    # Test all vectors included
    result = example_return_optional_vectors(True, True, True)
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
    result = example_return_optional_vectors(True, False, False)
    assert len(result) == 3

    # First vector should be present
    assert result[0] is not None
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    # Other vectors should be None
    assert result[1] is None
    assert result[2] is None


def test_example_return_optional_vectors_none():
    # Test with no vectors
    result = example_return_optional_vectors(False, False, False)
    assert len(result) == 3
    assert result[0] is None
    assert result[1] is None
    assert result[2] is None


def test_example_return_optional_vectors_types():
    result = example_return_optional_vectors()

    # Check types of returned vectors
    assert all(isinstance(x, float) for x in result[0])  # first vector should be floats
    assert all(isinstance(x, int) for x in result[1])  # second vector should be ints
    assert all(
        isinstance(x, float) for x in result[2]
    )  # third vector should be doubles/floats


@njit(cache=True, nogil=True)
def get_perc_from_k(k: int) -> list[float]:
    """
    It returns the percentiles (from 0.0 to 1.0) that separate the data into k
    clusters. For example, if k=2, it returns [0.5]; if k=4, it returns [0.25,
    0.50, 0.75].

    Args:
        k: number of clusters. If less than 2, the function returns an empty
            list.

    Returns:
        A list of percentiles (from 0.0 to 1.0).
    """
    return [(1.0 / k) * i for i in range(1, k)]


@njit(cache=True, nogil=True)
def run_quantile_clustering(data: NDArray, k: int) -> NDArray[np.int16]:
    """
    Performs a simple quantile clustering on one dimensional data (1d). Quantile
    clustering is defined as the procedure that forms clusters in 1d data by
    separating objects using quantiles (for instance, if the median is used, two
    clusters are generated with objects separated by the median). In the case
    data contains all the same values (zero variance), this implementation can
    return less clusters than specified with k.

    Args:
        data: a 1d numpy array with numerical values.
        k: the number of clusters to split the data into.

    Returns:
        A 1d array with the data partition.
    """
    data_sorted = np.argsort(data, kind="quicksort")
    data_rank = rank(data, data_sorted)
    data_perc = data_rank / len(data)

    percentiles = [0.0] + get_perc_from_k(k) + [1.0]

    cut_points = np.searchsorted(data_perc[data_sorted], percentiles, side="right")

    current_cluster = 0
    part = np.zeros(data.shape, dtype=np.int16) - 1

    for i in range(len(cut_points) - 1):
        lim1 = cut_points[i]
        lim2 = cut_points[i + 1]

        part[data_sorted[lim1:lim2]] = current_cluster
        current_cluster += 1

    return part


@njit(cache=True, nogil=True)
def get_range_n_clusters(
    n_features: int, internal_n_clusters: Iterable[int] = None
) -> NDArray[np.uint8]:
    """
    Given the number of features it returns a tuple of k values to cluster those
    features into. By default, it generates a tuple of k values from 2 to
    int(np.round(np.sqrt(n_features))) (inclusive). For example, for 25 features,
    it will generate this tuple: (2, 3, 4, 5).

    Args:
        n_features: a positive number representing the number of features that
            will be clustered into different groups/clusters.
        internal_n_clusters: it allows to force a different list of clusters. It
            must be a list of integers. Repeated or invalid values will be dropped,
            such as values lesser than 2 (a singleton partition is not allowed).

    Returns:
        A numpy array with integer values representing numbers of clusters.
    """

    if internal_n_clusters is not None:
        # remove k values that are invalid
        clusters_range_list = list(
            set([int(x) for x in internal_n_clusters if 1 < x < n_features])
        )
    else:
        # default behavior if no internal_n_clusters is given: return range from
        # 2 to sqrt(n_features)
        n_sqrt = int(np.round(np.sqrt(n_features)))
        n_sqrt = min((n_sqrt, 10))
        clusters_range_list = list(range(2, n_sqrt + 1))

    return np.array(clusters_range_list, dtype=np.uint16)


@njit(cache=True, nogil=True)
def get_parts(
    data: NDArray, range_n_clusters: tuple[int], data_is_numerical: bool = True
) -> NDArray[np.int16]:
    """
    Given a 1d data array, it computes a partition for each k value in the given
    range of clusters. This function only supports numerical data, and it
    always runs run_run_quantile_clustering with the different k values.
    If partitions with only one cluster are returned (singletons), then the
    returned array will have negative values.

    Args:
        data: a 1d data vector. It is assumed that there are no nans.
        range_n_clusters: a tuple with the number of clusters.
        data_is_numerical: indicates whether data is numerical (True) or categorical (False)

    Returns:
        A numpy array with shape (number of clusters, data rows) with
        partitions of data.

        Partitions could have negative values in some scenarios, with different
        meanings: -1 is used for categorical data, where only one partition is generated
        and the rest (-1) are marked as "empty". -2 is used when singletons have been
        detected (partitions with one cluster), usually because of problems with the
        input data (it has all the same values, for example).
    """
    parts = np.zeros((len(range_n_clusters), data.shape[0]), dtype=np.int16) - 1

    if data_is_numerical:
        for idx in range(len(range_n_clusters)):
            k = range_n_clusters[idx]
            parts[idx] = run_quantile_clustering(data, k)

        # remove singletons by putting a -2 as values
        partitions_ks = np.array([len(np.unique(p)) for p in parts])
        parts[partitions_ks == 1, :] = -2
    else:
        # if the data is categorical, then the encoded feature is already the partition
        # only the first partition is filled, the rest will be -1 (missing)
        parts[0] = data.astype(np.int16)

    return parts


def test_get_parts_simple():
    # Declare a literal 3D ND.array parts with shape (3, 4, 5)
    parts = np.array(
        [
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ],
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ],
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ],
        ],
        dtype=np.int16,
    )

    res = compute_coef(parts, 2, 3, 4)
    print(res)
