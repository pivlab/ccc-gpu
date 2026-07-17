"""
Contains function that implement the Clustermatch Correlation Coefficient (CCC).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import ccc_cuda_ext
import numpy as np
from numba import njit
from numba.typed import List
from numpy.typing import NDArray

from ccc.scipy.stats import rank
from ccc.utils import chunker


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
) -> NDArray[np.int16]:
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
    range of clusters. If partitions with only one cluster are returned (singletons),
    then the returned array will have negative values.

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


def get_feature_parts(params):
    """
    Given a list of parameters, it returns the partitions for each feature. The goal
    of this function is to parallelize the partitioning step (get_parts function).

    Args:
        params: a list of tuples with three elements: 1) a tuple with the feature
            index, the cluster index and the number of clusters (k), 2) the data for the
            feature, and 3) a boolean indicating whether the feature is numerical or not.

    Returns:
        A 2d array with the partitions (rows) for the selected features and number of
        clusters.
    """
    # Let's say we have:
    # 2 features: one numerical (temperature) and one categorical (color)
    # 3 data points
    # Want to try k=2 and k=3 clusters
    # Example setup
    # X = [
    #     np.array([20, 25, 30]),         # temperature (numerical)
    #     np.array(['red', 'blue', 'red']) # color (categorical)
    # ]
    # X_numerical_type = [True, False]  # temperature is numerical, color is categorical

    # # This would create params like this:
    # params = [
    #     # Chunk 1 (temperature feature)
    #     [
    #         (
    #             (0, 0, 2),              # (feature_idx=0, cluster_idx=0, k=2)
    #             np.array([20, 25, 30]),  # temperature data
    #             True                     # is numerical
    #         ),
    #         (
    #             (0, 1, 3),              # (feature_idx=0, cluster_idx=1, k=3)
    #             np.array([20, 25, 30]),  # temperature data
    #             True                     # is numerical
    #         )
    #     ],
    #     # Chunk 2 (color feature)
    #     [
    #         (
    #             (1, 0, 2),                           # (feature_idx=1, cluster_idx=0, k=2)
    #             np.array(['red', 'blue', 'red']),    # color data
    #             False                                # is categorical
    #         ),
    #         (
    #             (1, 1, 3),                           # (feature_idx=1, cluster_idx=1, k=3)
    #             np.array(['red', 'blue', 'red']),    # color data
    #             False                                # is categorical
    #         )
    #     ]
    # ]

    # # The function would process this and might return something like:
    # parts = [
    #     [0, 1, 1],    # temperature split into 2 clusters
    #     [0, 1, 2],    # temperature split into 3 clusters
    #     [0, 1, 0],    # color split into 2 clusters (categorical)
    #     [-1, -1, -1]  # ignored because categorical features only need one partition
    # ]

    n_objects = params[0][1].shape[0]
    parts = np.zeros((len(params), n_objects), dtype=np.int16) - 1

    # iterate over a list of tuples that indicate a feature-k pair
    for p_idx, p in enumerate(params):
        # the first element is a tuple with the feature index, the cluster index and the
        # number of clusters (k)
        info = p[0]
        # f_idx = info[0]
        c_idx = info[1]
        c = info[2]
        range_n_clusters = np.array([c], dtype=np.uint16)

        # the second element is the data for the feature
        data = p[1]

        # the third element is a boolean indicating whether the feature is numerical
        numerical_data_type = p[2]

        # if the feature is categorical, then only the first partition is filled
        if not numerical_data_type and c_idx > 0:
            continue

        parts[p_idx] = get_parts(data, range_n_clusters, numerical_data_type)

    return parts


def get_chunks(
    iterable: int | Iterable, n_threads: int, ratio: float = 1
) -> Iterable[Iterable[int]]:
    """
    It splits elements in an iterable in chunks according to the number of
    CPU cores available for parallel processing.

    Args:
        iterable: an iterable to be split in chunks. If it is an integer, it
            will split the iterable given by np.arange(iterable).
        n_threads: number of threads available for parallelization.
        ratio: a ratio that allows to increase the number of splits given
            n_threads. For example, with ratio=1, the function will just split
            the iterable in n_threads chunks. If ratio is larger than 1, then
            it will split in n_threads * ratio chunks.

    Results:
        Another iterable with chunks according to the arguments given. For
        example, if iterable is [0, 1, 2, 3, 4, 5] and n_threads is 2, it will
        return [[0, 1, 2], [3, 4, 5]].
    """
    if isinstance(iterable, int):
        iterable = np.arange(iterable)

    n = len(iterable)
    expected_n_chunks = n_threads * ratio

    res = list(chunker(iterable, int(np.ceil(n / expected_n_chunks))))

    while len(res) < expected_n_chunks <= n:
        # look for an element in res that can be split in two
        idx = 0
        while len(res[idx]) == 1:
            idx = idx + 1

        new_chunk = get_chunks(res[idx], 2)
        res[idx] = new_chunk[0]
        res.insert(idx + 1, new_chunk[1])

    return res


def get_feature_type_and_encode(feature_data: NDArray) -> tuple[NDArray, bool]:
    """
    Given the data of one feature as a 1d numpy array (it could also be a pandas.Series),
    it returns the same data if it is numerical (float, signed or unsigned integer) or an
    encoded version if it is categorical (each category value has a unique integer starting from
    zero).

    Args:
        feature_data: a 1d array with data.

    Returns:
        A tuple with two elements:
          1. the feature data: same as input if numerical, encoded version if not numerical.
          2. A boolean indicating whether the feature data is numerical or not.
    """
    data_type_is_numerical = feature_data.dtype.kind in ("f", "i", "u")
    if data_type_is_numerical:
        return feature_data, data_type_is_numerical

    # here np.unique with return_inverse encodes categorical values into numerical ones
    return np.unique(feature_data, return_inverse=True)[1], data_type_is_numerical


def get_n_workers(n_jobs: int | None) -> int:
    """
    Helper function to get the number of workers for parallel processing.

    Args:
        n_jobs: value specified by the main ccc function.
    Returns:
        The number of workers to use for parallel processing
    """
    n_cpu_cores = os.cpu_count()
    if n_cpu_cores is None:
        raise ValueError(
            "Could not determine the number of CPU cores. Please specify a positive value of n_jobs"
        )

    n_workers = n_cpu_cores
    if n_jobs is None:
        return n_workers

    n_workers = os.cpu_count() + n_jobs if n_jobs < 0 else n_jobs

    if n_workers < 1:
        raise ValueError(
            f"The number of threads/processes to use must be greater than 0. Got {n_workers}."
            "Please check the n_jobs argument provided"
        )

    return n_workers


def ccc(
    x: NDArray,
    y: NDArray = None,
    internal_n_clusters: int | Iterable[int] = None,
    return_parts: bool = False,
    n_chunks_threads_ratio: int = 1,
    n_jobs: int = 1,
    pvalue_n_perms: int = None,
    partitioning_executor: str = "thread",
) -> float | NDArray[np.float64] | tuple:
    """
    This is the main function that computes the Clustermatch Correlation
    Coefficient (CCC) between two arrays. The implementation supports numerical
    and categorical data.

    This is the GPU-accelerated implementation; the coefficient computation runs
    on the GPU (values are computed in float32, so they may differ from the CPU
    implementation in ``ccc.coef.impl`` by a small tolerance).

    Args:
        x: 1d or 2d numerical array with the data. NaN are not supported.
          If it is 2d, then the coefficient is computed for each pair of rows
          (in case x is a numpy.array) or each pair of columns (pandas.DataFrame).
        y: an optional 1d numerical array. If x is 1d and y is given, it computes
          the coefficient between x and y.
        internal_n_clusters: this parameter can be an integer (the maximum number
          of clusters used to split x and y, starting from k=2) or a list of
          integer values (a custom list of k values).
        return_parts: if True, for each object pair, it returns the partitions
          that maximized the coefficient.
        n_chunks_threads_ratio: allows to modify how pairwise comparisons are
          split across different threads. It's given as the ratio parameter of
          function get_chunks.
        n_jobs: number of CPU cores/threads to use for parallelization. The value
          None will use all available cores (`os.cpu_count()`), and negative
          values will use `os.cpu_count() + n_jobs` (exception will be raised
          if this expression yields a result less than 1). Default is 1.
        pvalue_n_perms: if given (an integer > 0), also estimate a p-value for
            each coefficient with a one-sided permutation test (computed on the
            GPU). One of the two features' partitions is randomly shuffled
            ``pvalue_n_perms`` times and the CCC is recomputed each time; the
            p-value is the fraction of permuted coefficients greater than or equal
            to the observed one, with add-one (Laplace) smoothing::

                p = (#{permuted CCC >= observed CCC} + 1) / (pvalue_n_perms + 1)

            The test is one-sided: a *smaller* p-value is stronger evidence that
            the association is not due to chance. Because of the ``+1`` in the
            numerator and denominator, the smallest resolvable p-value is
            ``1 / (pvalue_n_perms + 1)`` (e.g. ~1e-3 for 999 permutations), so
            pick ``pvalue_n_perms`` for the resolution you need. If ``None`` or
            ``0`` (the default), no p-value is computed. (The GPU permutation
            p-values were corrected in the ``fix-cuda-correctness`` change, so
            values may differ from pre-fix releases.)

            Cost warning: for a 2d input the p-value is computed independently
            for every one of the ``n * (n - 1) / 2`` feature pairs, each costing
            ``pvalue_n_perms`` extra CCC evaluations -- a total of about
            ``n * (n - 1) / 2 * pvalue_n_perms`` additional coefficient
            computations, which can be orders of magnitude more expensive than
            the point estimate alone.
        partitioning_executor: Executor type used for partitioning the data. It
            can be either "thread" (default) or "process". If "thread", it will use
            ThreadPoolExecutor for parallelization, which uses less memory. If
            "process", it will use ProcessPoolExecutor, which might be faster. If
            anything else, it will not parallelize the partitioning step.


    Returns:
        The return type is polymorphic; it depends on the input shape and on the
        ``pvalue_n_perms`` / ``return_parts`` flags:

        - 1d ``x`` and ``y`` (a single feature pair): the coefficient is a scalar
          ``float``.
        - 2d ``x`` (``n`` features/rows): the coefficients are a 1d condensed
          array ``cm_values`` of length ``n * (n - 1) / 2`` (the upper triangle of
          the pairwise matrix, compatible with
          ``scipy.spatial.distance.squareform``).

        When ``pvalue_n_perms`` is an integer greater than 0, the coefficient
        result is replaced by a 2-tuple ``(cm_values, cm_pvalues)`` whose elements
        have matching shapes (two scalars for a single pair; two 1d arrays for a
        2d input).

        When ``return_parts`` is True, a 3-tuple ``(coefficients, max_parts,
        parts)`` is returned instead of the coefficients alone -- and
        ``coefficients`` is itself the ``(cm_values, cm_pvalues)`` tuple described
        above when ``pvalue_n_perms`` was given.

        cm_values: the CCC coefficient(s). Each value is between 0 and 1
            (inclusive), or ``np.nan`` when one of the two variables has no
            variation (all values are the same) so the coefficient is undefined.

        cm_pvalues: present only when ``pvalue_n_perms`` > 0. The one-sided
            permutation p-value(s), aligned with ``cm_values`` (same shape).

        max_parts: an array with ``n * (n - 1) / 2`` rows (one for each object
            pair) and two columns. It has the indexes pointing to each object's
            partition (parts, see below) that maximized the ARI. If
            cm_values[idx] is nan, then max_parts[idx] will be meaningless.

        parts: a 3d array that contains all the internal partitions generated
            for each object in data. parts[i] has the partitions for object i,
            whereas parts[i,j] has the partition j generated for object i. The
            third dimension is the number of columns in x (if 2d) or elements in
            x/y (if 1d). For example, if you want to access the pair of
            partitions that maximized the CCC given x and y
            (a pair of objects), then max_parts[0] and max_parts[1] have the
            partition indexes in parts, respectively: parts[0][max_parts[0]]
            points to the partition for x, and parts[1][max_parts[1]] points to
            the partition for y. Values could be negative in case
            singleton cases were found (-2; usually because input data has all the same
            value) or for categorical features (-1).
    """
    n_objects = None
    n_features = None
    # this is a boolean array of size n_features with True if the feature is numerical and False otherwise
    X_numerical_type = None
    if x.ndim == 1 and (y is not None and y.ndim == 1):
        # both x and y are 1d arrays
        if not x.shape == y.shape:
            raise ValueError("x and y need to be of the same size")
        n_objects = x.shape[0]
        n_features = 2

        X = np.zeros((n_features, n_objects))
        X_numerical_type = np.full((n_features,), True, dtype=bool)

        X[0, :], X_numerical_type[0] = get_feature_type_and_encode(x)
        X[1, :], X_numerical_type[1] = get_feature_type_and_encode(y)
    elif x.ndim == 2 and y is None:
        # x is a 2d array; two things could happen: 1) this is an numpy array,
        # in that case, features are in rows, objects are in columns; 2) or this is a
        # pandas dataframe, which is the opposite (features in columns and objects in rows),
        # plus we have the features data type (numerical, categorical, etc)

        if isinstance(x, np.ndarray):
            if not get_feature_type_and_encode(x[0, :])[1]:
                raise ValueError(
                    "If data is a 2d numpy array, it has to be numerical. Use pandas.DataFrame if "
                    "you need to mix features with different data types"
                )
            n_objects = x.shape[1]
            n_features = x.shape[0]

            X = x
            X_numerical_type = np.full((n_features,), True, dtype=bool)
        elif hasattr(x, "to_numpy"):
            # Here I assume that if x has the attribute "to_numpy" is of type pandas.DataFrame
            # Using isinstance(x, pandas.DataFrame) would be more appropriate, but I dont want to
            # have pandas as a dependency just for that
            n_objects = x.shape[0]
            n_features = x.shape[1]

            X = np.zeros((n_features, n_objects))
            X_numerical_type = np.full((n_features,), True, dtype=bool)

            for f_idx in range(n_features):
                X[f_idx, :], X_numerical_type[f_idx] = get_feature_type_and_encode(
                    x.iloc[:, f_idx]
                )
    else:
        raise ValueError("Wrong combination of parameters x and y")

    # Note: Categorical data is now handled by the GPU implementation

    # get number of cores to use
    n_workers = get_n_workers(n_jobs)

    if internal_n_clusters is not None:
        _tmp_list = List()

        if isinstance(internal_n_clusters, int):
            # this interprets internal_n_clusters as the maximum k
            internal_n_clusters = range(2, internal_n_clusters + 1)

        for x in internal_n_clusters:
            _tmp_list.append(x)
        internal_n_clusters = _tmp_list

    # get matrix of partitions for each object pair
    range_n_clusters = get_range_n_clusters(n_objects, internal_n_clusters)
    n_clusters = range_n_clusters.shape[0]

    if n_clusters == 0:
        raise ValueError(f"Data has too few objects: {n_objects}")

    # store a set of partitions per row (object) in X as a multidimensional
    # array, where the second dimension is the number of partitions per object.
    parts = np.zeros((n_features, n_clusters, n_objects), dtype=np.int16) - 1

    # cm_values stores the CCC coefficients
    n_features_comp = (n_features * (n_features - 1)) // 2
    cm_values = np.full(n_features_comp, np.nan)
    cm_pvalues = np.full(n_features_comp, np.nan)

    # for each object pair being compared, max_parts has the indexes of the
    # partitions that maximized the ARI
    max_parts = np.zeros((n_features_comp, 2), dtype=np.uint64)

    with (
        ThreadPoolExecutor(max_workers=n_workers) as executor,
        ProcessPoolExecutor(max_workers=n_workers) as pexecutor,
    ):
        map_func = map
        if n_workers > 1:
            if partitioning_executor == "thread":
                map_func = executor.map
            elif partitioning_executor == "process":
                map_func = pexecutor.map

        # pre-compute the internal partitions for each object in parallel

        # first, create a list with features-k pairs that will be used to parallelize
        # the partitioning step
        inputs = get_chunks(
            [
                (f_idx, c_idx, c)
                for f_idx in range(n_features)
                for c_idx, c in enumerate(range_n_clusters)
            ],
            n_workers,
            n_chunks_threads_ratio,
        )
        # For example, if you have:
        # 2 features (n_features = 2)
        # range_n_clusters = [2, 3, 4]
        # You'll get pairs like:
        # [
        #     (0, 0, 2),  # (feature 0, first k, k=2)
        #     (0, 1, 3),  # (feature 0, second k, k=3)
        #     (0, 2, 4),  # (feature 0, third k, k=4)
        #     (1, 0, 2),  # (feature 1, first k, k=2)
        #     (1, 1, 3),  # (feature 1, second k, k=3)
        #     (1, 2, 4)   # (feature 1, third k, k=4)
        # ]

        # then, flatten the list of features-k tuples into a list that is divided into
        # chunks that will be used to parallelize the partitioning step.
        inputs = [
            [
                (
                    feature_k_pair,  # Original (f_idx, c_idx, k) tuple
                    X[feature_k_pair[0]],  # Actual feature data using f_idx
                    X_numerical_type[
                        feature_k_pair[0]
                    ],  # Data type info for this feature
                )
                for feature_k_pair in chunk
            ]
            for chunk in inputs
        ]
        # Let's say we have:
        # 2 features (indices 0, 1)
        # range_n_clusters = [2, 3]
        # 2 chunks for parallel processing
        # The inputs would look like this:
        # inputs = [
        # Chunk 1
        #     [
        #         (
        #             (0, 0, 2),           # (feature 0, first k, k=2)
        #             X[0],                # Data for feature 0
        #             X_numerical_type[0]  # Type info for feature 0
        #         ),
        #         (
        #             (0, 1, 3),           # (feature 0, second k, k=3)
        #             X[0],                # Data for feature 0
        #             X_numerical_type[0]  # Type info for feature 0
        #         )
        #     ],
        # Chunk 2
        #     [
        #         (
        #             (1, 0, 2),           # (feature 1, first k, k=2)
        #             X[1],                # Data for feature 1
        #             X_numerical_type[1]  # Type info for feature 1
        #         ),
        #         (
        #             (1, 1, 3),           # (feature 1, second k, k=3)
        #             X[1],                # Data for feature 1
        #             X_numerical_type[1]  # Type info for feature 1
        #         )
        #     ]
        # ]

        for params, ps in zip(inputs, map_func(get_feature_parts, inputs)):
            # get the set of feature indexes and cluster indexes
            f_idxs = [p[0][0] for p in params]
            c_idxs = [p[0][1] for p in params]

            # update the partitions for each feature-k pair
            parts[f_idxs, c_idxs] = ps

    # Compute the CCC coefficient for all feature pairs
    # Use the GPU implementation for all data types (numerical and categorical)
    coef = ccc_cuda_ext.compute_coef(
        parts, n_features, n_clusters, n_objects, return_parts, pvalue_n_perms
    )
    cm_values, cm_pvalues, max_parts = coef

    # return an array of values or a single scalar, depending on the input data
    if cm_values.shape[0] == 1:
        if return_parts:
            if pvalue_n_perms is not None and pvalue_n_perms > 0:
                return (cm_values[0], cm_pvalues[0]), max_parts[0], parts
            return cm_values[0], max_parts[0], parts
        if pvalue_n_perms is not None and pvalue_n_perms > 0:
            return cm_values[0], cm_pvalues[0]
        return cm_values[0]

    if return_parts:
        if pvalue_n_perms is not None and pvalue_n_perms > 0:
            return (cm_values, cm_pvalues), max_parts, parts
        return cm_values, max_parts, parts
    if pvalue_n_perms is not None and pvalue_n_perms > 0:
        return cm_values, cm_pvalues
    return cm_values
