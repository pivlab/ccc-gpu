Algorithms
==========

This section explains the underlying algorithms and mathematical foundations of the Clustermatch Correlation Coefficient (CCC) and its GPU implementation.

Clustermatch Correlation Coefficient (CCC)
-------------------------------------------

The Clustermatch Correlation Coefficient is based on the simple yet powerful idea of clustering data points and then computing the Adjusted Rand Index (ARI) between the two clusterings. This approach allows CCC to capture both linear and non-linear relationships effectively.

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

Given two variables X and Y with n observations each, CCC works as follows:

1. **Quantile Clustering**: For each variable, create multiple partitions using quantile-based clustering with different numbers of clusters k
2. **Partition Generation**: Generate partitions for k = 2, 3, ..., √n clusters (customizable)
3. **ARI Computation**: Calculate the Adjusted Rand Index between all partition pairs
4. **Maximization**: Return the maximum ARI value across all partition combinations

The ARI is defined as:

.. math::

    ARI = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}

where RI is the Rand Index and E[RI] is its expected value under random partitioning.

Quantile Clustering Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantile clustering procedure partitions data into k clusters based on percentiles:

1. **Rank Computation**: Convert data values to ranks using ``rank(data)``
2. **Percentile Calculation**: Compute percentiles that divide data into k equal-sized groups
3. **Partition Assignment**: Assign each data point to a cluster based on its percentile position

.. code-block:: python

    def get_perc_from_k(k):
        """Get percentiles for k clusters"""
        return [(1.0 / k) * i for i in range(1, k)]

    def run_quantile_clustering(data, k):
        """Perform quantile clustering on 1D data"""
        data_rank = rank(data)
        data_perc = data_rank / len(data)
        percentiles = [0.0] + get_perc_from_k(k) + [1.0]
        # Assign clusters based on percentile boundaries
        return partition

Adjusted Rand Index (ARI)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ARI measures the similarity between two clusterings, corrected for chance:

.. math::

    ARI(X, Y) = \frac{2(ad - bc)}{(a + b)(b + d) + (a + c)(c + d)}

where a, b, c, d are counts from the contingency table of the two partitions.

ARI Properties:

- **Range**: [-1, 1] where 1 indicates perfect agreement
- **Chance Correction**: Expected value is 0 for random partitions
- **Symmetric**: ARI(X, Y) = ARI(Y, X)

GPU Implementation Details
--------------------------

CCC-GPU leverages CUDA to accelerate the most computationally intensive parts of the algorithm:

CUDA Kernel Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

The GPU implementation uses several specialized CUDA kernels:

1. **ARI Computation Kernel**: Parallelizes ARI calculations across feature pairs
2. **Max-Finding Kernel**: Uses CUB primitives for efficient maximum value detection
3. **Partition Processing**: Handles multiple partition comparisons in parallel

.. code-block:: cuda

    template <typename T>
    __global__ void findMaxAriKernel(const T *aris,
                                     uint8_t *max_parts,
                                     T *cm_values,
                                     const int n_partitions,
                                     const int reduction_range)

Memory Management
~~~~~~~~~~~~~~~~~

The GPU implementation employs sophisticated memory management:

- **Device Memory**: Stores partition data and intermediate results on GPU
- **Thrust Containers**: Uses ``thrust::device_vector`` for RAII memory management  
- **Memory Coalescing**: Optimizes memory access patterns for maximum bandwidth
- **Batch Processing**: Processes data in batches to handle memory constraints

Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Several optimizations contribute to the significant speedups:

1. **Parallel ARI Computation**: Each thread block handles one feature comparison
2. **CUB Primitives**: Leverages highly-optimized reduction operations
3. **Memory Hierarchy**: Strategic use of shared memory and registers
4. **Warp-Level Primitives**: Exploits SIMD execution within warps

Algorithmic Complexity
----------------------

Time Complexity
~~~~~~~~~~~~~~~

For n samples and m features:

- **CPU Implementation**: O(m² × k² × n log n) where k is max clusters
- **GPU Implementation**: O(m² × k² × n log n / p) where p is parallelization factor
- **Memory Complexity**: O(m × k × n) for storing all partitions

The GPU implementation achieves 16x-74x speedups depending on dataset size and hardware.

Space Complexity
~~~~~~~~~~~~~~~~

The algorithm requires storage for:

- **Original Data**: O(m × n) 
- **Partitions**: O(m × k × n) for all clustering variations
- **ARI Matrix**: O(m² × k²) for pairwise comparisons
- **Results**: O(m²) for final correlation matrix

Categorical Data Support
------------------------

CCC naturally handles categorical data through specialized encoding:

1. **Categorical Encoding**: Each unique category becomes a cluster label
2. **Single Partition**: Categorical features generate only one meaningful partition  
3. **Mixed Data Types**: Seamlessly combines numerical and categorical features

.. code-block:: python

    def get_feature_type_and_encode(feature_data):
        """Detect and encode feature data type"""
        is_numerical = feature_data.dtype.kind in ("f", "i", "u")
        if is_numerical:
            return feature_data, True
        # Encode categorical as integers
        return np.unique(feature_data, return_inverse=True)[1], False

Implementation Considerations
-----------------------------

Numerical Stability
~~~~~~~~~~~~~~~~~~~

The implementation includes several measures for numerical stability:

- **Singleton Detection**: Handles features with zero variance (all same values)
- **NaN Handling**: Proper treatment of invalid computations
- **Precision Control**: Uses appropriate floating-point precision for calculations

Error Handling
~~~~~~~~~~~~~~

Robust error handling for edge cases:

- **Insufficient Data**: Requires minimum number of samples for clustering
- **GPU Memory Limits**: Graceful handling of memory constraints
- **CUDA Errors**: Proper error checking and cleanup

The algorithm is designed to be both mathematically sound and computationally efficient, making it suitable for large-scale genomics and machine learning applications.
