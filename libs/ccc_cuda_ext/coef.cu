#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <curand_kernel.h>
#include <spdlog/spdlog.h>

#include <execution>
#include <iostream>
#include <iomanip>
#include <limits>
#include <optional>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "coef.cuh"
#include "metrics.cuh"
#include "math.cuh"
#include "utils.cuh"
namespace py = pybind11;


/**
 * @brief CUDA kernel to find maximum ARI values and their corresponding partition pairs
 *
 * This kernel processes a range of ARIs for a single feature comparison and finds:
 * 1. The maximum ARI value
 * 2. The partition pair (m,n) that achieved this maximum
 *
 * @tparam T The floating-point type for ARI values (float or double)
 * @param aris Input array of ARI values
 * @param max_parts Output array for partition pairs that achieved maximum ARIs
 * @param cm_values Output array for maximum ARI values
 * @param n_partitions Number of partitions to consider
 * @param reduction_range Number of partition pairs to process per feature comparison
 */
template <typename T>
__global__ void findMaxAriKernel(const T *aris,
                                 uint8_t *max_parts,
                                 T *cm_values,
                                 const int n_partitions,
                                 const int reduction_range)
{
    /*
     * Thread and Block Setup
     * --------------------
     * Each block handles one feature comparison, with threads collaboratively
     * processing all partition pairs for that comparison.
     */
    const uint64_t comp_idx = blockIdx.x;
    const uint64_t reduce_start_idx = comp_idx * reduction_range;

    /*
     * Thread-local Reduction Variables
     * -----------------------------
     * Each thread maintains its own maximum value and index,
     * which will be reduced across the block later.
     * Using a key-value pair for proper ArgMax reduction.
     */
    typedef cub::KeyValuePair<int, T> KeyValuePairT;
    KeyValuePairT thread_data;
    thread_data.key = -1;  // Initialize to invalid index
    thread_data.value = -1.0f;  // Initialize to very small value
    bool has_nan = false;

    /*
     * Initial Reduction Phase
     * ---------------------
     * Each thread processes a subset of partition pairs to find local maximum.
     * Handles NaN values by marking them and skipping in the reduction.
     */
    for (uint64_t i = threadIdx.x; i < reduction_range; i += blockDim.x)
    {
        uint64_t idx = reduce_start_idx + i;
        T val = aris[idx];

        // Check for NaN
        if (isnan(val))
        {
            has_nan = true;
            continue;
        }

        if (val > thread_data.value)
        {
            thread_data.value = val;
            thread_data.key = i;  // Store the local index
        }
    }

    /*
     * NaN Handling
     * -----------
     * If any thread found a NaN, the entire feature comparison is marked as invalid.
     */
    // Use shared memory to communicate NaN status
    __shared__ bool block_has_nan;
    if (threadIdx.x == 0)
    {
        block_has_nan = false;
    }
    __syncthreads();
    
    if (has_nan)
    {
        block_has_nan = true;
    }
    __syncthreads();
    
    if (block_has_nan)
    {
        if (threadIdx.x == 0)
        {
            cm_values[comp_idx] = NAN;
            max_parts[comp_idx * 2] = 0;
            max_parts[comp_idx * 2 + 1] = 0;
        }
        return;
    }

    /*
     * Block-level Reduction Setup
     * -------------------------
     * Use CUB's ArgMax to find both the maximum value and its index in one operation.
     */
    typedef cub::BlockReduce<KeyValuePairT, 128> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    // Use standard ArgMax but rely on CUB's tie-breaking behavior
    KeyValuePairT aggregate = BlockReduceT(temp_storage).Reduce(
        thread_data, 
        cub::ArgMax()
    );

    /*
     * Result Writing
     * ------------
     * Thread 0 writes the final results for this feature comparison.
     * Converts the linear index back to partition pair (m,n).
     */
    if (threadIdx.x == 0)
    {
        // Store the maximum ARI value
        cm_values[comp_idx] = aggregate.value > 0.0f ? aggregate.value : 0.0f;

        // Convert linear index to partition pair. Guard against a negative key
        // (no valid maximum found): a negative aggregate.key would produce a
        // bogus column index via unsigned modulo and then truncate into the
        // uint8_t max_parts buffer. Fall back to (0, 0) in that case.
        if (aggregate.key < 0)
        {
            max_parts[comp_idx * 2] = 0;
            max_parts[comp_idx * 2 + 1] = 0;
        }
        else
        {
            unsigned int m = aggregate.key / n_partitions;
            unsigned int n = aggregate.key % n_partitions;

            // Store the partition pair
            max_parts[comp_idx * 2] = m;
            max_parts[comp_idx * 2 + 1] = n;
        }
    }
}

/**
 * @brief CUDA kernel to initialize cuRAND states for p-value computation
 * @param states Array of cuRAND states to initialize
 * @param n_states Number of states to initialize
 * @param seed Random seed for initialization
 */
__global__ void initRandomStates(curandState *states, const uint32_t n_states, const uint32_t seed)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * @brief CUDA kernel to generate permutation indices for p-value computation
 * @param states Array of cuRAND states
 * @param perm_indices Output array for permutation indices
 * @param n_perms Number of permutations
 * @param n_objects Number of objects to permute
 */
__global__ void generatePermutations(curandState *states, uint32_t *perm_indices, 
                                   const uint32_t n_perms, const uint32_t n_objects)
{
    uint32_t perm_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (perm_idx < n_perms)
    {
        curandState *state = &states[perm_idx];
        uint32_t *perm = &perm_indices[perm_idx * n_objects];
        
        // Initialize sequential indices
        for (uint32_t i = 0; i < n_objects; ++i)
        {
            perm[i] = i;
        }
        
        // Fisher-Yates shuffle
        for (uint32_t i = n_objects - 1; i > 0; --i)
        {
            uint32_t j = curand(state) % (i + 1);
            uint32_t temp = perm[i];
            perm[i] = perm[j];
            perm[j] = temp;
        }
    }
}

/**
 * @brief Device function to compute ARI between two partitions with permutation
 *
 * Uses a caller-provided global-memory scratch region for the k x k contingency
 * matrix and the row/column sum arrays, so there is no fixed cluster-count cap
 * (the old MAX_CLUSTERS=16 thread-local arrays silently returned 0.0 for k>16).
 * The contingency counts stay in int (bounded by n_objects, matching the main
 * kernel) while every squared/combination accumulator is 64-bit to avoid the
 * overflow that occurs once a cluster holds more than ~46k objects.
 *
 * @param part_i First partition (fixed)
 * @param part_j Second partition (permuted through @p perm)
 * @param perm Permutation indices
 * @param n_objects Number of objects
 * @param k Cluster-count upper bound (max cluster id + 1) used to size scratch
 * @param scratch Global scratch: k*k ints (contingency) + k ints (sum_rows) +
 *                k ints (sum_cols), unique to the calling thread
 * @return ARI value
 */
template <typename T, typename R>
__device__ R computePermutedARI(const T *part_i, const T *part_j,
                                const uint32_t *perm, const uint32_t n_objects,
                                const int k, int *scratch)
{
    // Quick validation
    if (n_objects == 0) return 0.0f;
    if (k <= 0) return 0.0f;

    int *contingency = scratch;          // k * k ints
    int *sum_rows = contingency + k * k; // k ints
    int *sum_cols = sum_rows + k;        // k ints

    // Initialize scratch arrays
    for (int i = 0; i < k * k; ++i) contingency[i] = 0;
    for (int i = 0; i < k; ++i) { sum_rows[i] = 0; sum_cols[i] = 0; }

    // Build contingency matrix and compute sum of squares in a single pass.
    long long sum_squares = 0;
    for (uint32_t idx = 0; idx < n_objects; ++idx)
    {
        T cluster_i = part_i[idx];
        T cluster_j = part_j[perm[idx]];

        if (cluster_i >= 0 && cluster_j >= 0 && cluster_i < k && cluster_j < k)
        {
            int cont_idx = cluster_i * k + cluster_j;
            contingency[cont_idx]++;
            int val = contingency[cont_idx];

            // Update sum_squares incrementally: val^2 - (val-1)^2 = 2*val - 1
            sum_squares += 2LL * val - 1;
        }
    }

    // Compute row and column sums
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int val = contingency[i * k + j];
            sum_rows[i] += val;
            sum_cols[j] += val;
        }
    }

    // Compute combination sums using 64-bit arithmetic
    long long sum_comb_c = 0, sum_comb_k = 0;
    for (int i = 0; i < k; ++i)
    {
        if (sum_rows[i] > 1)
            sum_comb_c += (static_cast<long long>(sum_rows[i]) * (sum_rows[i] - 1)) / 2;
        if (sum_cols[i] > 1)
            sum_comb_k += (static_cast<long long>(sum_cols[i]) * (sum_cols[i] - 1)) / 2;
    }

    long long sum_comb_ck = (sum_squares - static_cast<long long>(n_objects)) / 2;

    // Compute ARI with improved numerical stability
    if (sum_comb_c == 0 && sum_comb_k == 0)
    {
        return (sum_comb_ck == 0) ? 1.0f : 0.0f;
    }

    // Use double precision for intermediate calculations to avoid overflow
    double n_choose_2 = static_cast<double>(n_objects) * (n_objects - 1) / 2.0;
    double expected_index = static_cast<double>(sum_comb_c) * static_cast<double>(sum_comb_k) / n_choose_2;
    double max_index = (static_cast<double>(sum_comb_c) + static_cast<double>(sum_comb_k)) / 2.0;

    if (fabs(max_index - expected_index) < 1e-10)
    {
        return 0.0f;
    }

    double ari = (static_cast<double>(sum_comb_ck) - expected_index) / (max_index - expected_index);
    return static_cast<R>(ari);
}

/**
 * @brief CUDA kernel to compute CCC values for permuted partitions
 *
 * Each thread handles one permutation and reduces (max) over all partition
 * pairs, applying the SAME invalid-partition semantics as the observed
 * statistic (ari_kernel + findMaxAriKernel):
 *   - categorical pair (-1 marker): contributes ARI 0.0,
 *   - singleton pair (-2 marker): yields NaN which poisons the comparison,
 *   - the reduced value is clamped to >= 0.0 (matching the observed clamp and
 *     the CPU reference's max(., 0.0)).
 * This keeps the permutation null distribution consistent with the observed
 * coefficient, removing the previous bias (negative permuted values, and
 * silent skips of categorical partitions).
 *
 * @param parts_i Partitions for the fixed feature
 * @param parts_j Partitions for the permuted feature
 * @param perm_indices Permutation indices
 * @param perm_ccc_values Output array for permutation CCC values
 * @param n_perms Number of permutations
 * @param n_partitions Number of partitions per feature
 * @param n_objects Number of objects
 * @param k Cluster-count upper bound used to size the per-thread scratch
 * @param scratch Global scratch (n_perms * scratch_stride ints)
 * @param scratch_stride Ints per thread: k*k + 2*k
 */
template <typename T, typename R>
__global__ void computePermutationCCC(const T *parts_i, const T *parts_j,
                                     const uint32_t *perm_indices, R *perm_ccc_values,
                                     const uint32_t perm_offset, const uint32_t perm_count,
                                     const uint32_t n_partitions, const uint32_t n_objects,
                                     const int k, int *scratch, const uint64_t scratch_stride)
{
    const uint32_t local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= perm_count) return;
    const uint32_t perm_idx = perm_offset + local_idx;

    // Permutation for this thread. Scratch is indexed by the LOCAL id so the
    // scratch buffer only needs `perm_count` slices — bounding memory to the
    // permutation sub-batch regardless of the cluster count k.
    const uint32_t *perm = &perm_indices[static_cast<uint64_t>(perm_idx) * n_objects];
    int *thread_scratch = scratch + static_cast<uint64_t>(local_idx) * scratch_stride;

    R max_ari = 0.0f;
    bool found_valid_ari = false;
    bool has_singleton = false;

    // Compute ARI for all partition pairs with this permutation
    for (uint32_t i = 0; i < n_partitions; ++i)
    {
        for (uint32_t j = 0; j < n_partitions; ++j)
        {
            const T *part_i = &parts_i[i * n_objects];
            const T *part_j = &parts_j[j * n_objects];

            const PartPairValidity validity = classify_partition_pair(part_i[0], part_j[0]);
            if (validity == PartPairValidity::SINGLETON)
            {
                // Singleton partition -> NaN, mirroring the observed statistic
                // whose max reduction poisons the whole comparison to NaN.
                has_singleton = true;
                continue;
            }

            R ari_value;
            if (validity == PartPairValidity::CATEGORICAL)
            {
                // Categorical marker contributes 0.0 (matches ari_kernel).
                ari_value = 0.0f;
            }
            else
            {
                ari_value = computePermutedARI<T, R>(part_i, part_j, perm, n_objects, k, thread_scratch);
                if (!isfinite(ari_value))
                {
                    continue;
                }
            }

            if (!found_valid_ari || ari_value > max_ari)
            {
                max_ari = ari_value;
                found_valid_ari = true;
            }
        }
    }

    R result;
    if (has_singleton)
    {
        // Consistent with the observed path: any singleton pair -> NaN.
        result = NAN;
    }
    else if (found_valid_ari)
    {
        // Clamp to >= 0.0, matching findMaxAriKernel and the CPU reference.
        result = max_ari > 0.0f ? max_ari : 0.0f;
    }
    else
    {
        result = 0.0f;
    }
    perm_ccc_values[perm_idx] = result;
}

/**
 * @brief CUDA kernel to compute p-values from permutation results
 * @param perm_ccc_values Array of CCC values from permutations
 * @param observed_ccc_values Array of observed CCC values
 * @param pvalues Output array for p-values
 * @param n_comparisons Number of feature comparisons
 * @param n_perms Number of permutations per comparison
 */
template <typename R>
__global__ void computePValues(const R *perm_ccc_values, const R *observed_ccc_values,
                              R *pvalues, const uint64_t n_comparisons, const uint32_t n_perms)
{
    const uint64_t comp_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (comp_idx < n_comparisons)
    {
        const R observed_value = observed_ccc_values[comp_idx];
        // 64-bit offset: comp_idx * n_perms can exceed 2^32 for large inputs.
        const R *perm_values = &perm_ccc_values[comp_idx * static_cast<uint64_t>(n_perms)];

        uint64_t count = 0;
        for (uint32_t i = 0; i < n_perms; ++i)
        {
            if (perm_values[i] >= observed_value)
            {
                count++;
            }
        }

        // Standard permutation test p-value formula
        pvalues[comp_idx] = static_cast<R>(count + 1) / static_cast<R>(n_perms + 1);
    }
}

/**
 * @brief Check if feature comparison count would exceed maximum representable value
 *
 * @param n_features Number of features
 * @throws std::range_error if calculation would exceed maximum representable value
 */
void check_feature_comp_bounds(const size_t n_features)
{
    if (n_features > 1 && n_features > UINT64_MAX / (n_features - 1))
    {
        throw std::range_error("Feature comparison count would exceed maximum representable value: n_features too large");
    }
}

/**
 * @brief Check if ARI count would exceed maximum representable value
 *
 * @param n_feature_comp Number of feature comparisons
 * @param n_partitions Number of partitions
 * @throws std::range_error if calculation would exceed maximum representable value
 */
void check_ari_count_bounds(const uint64_t n_feature_comp, const size_t n_partitions)
{
    if (n_feature_comp > UINT64_MAX / n_partitions)
    {
        throw std::range_error("ARI count would exceed maximum representable value: n_feature_comp * n_partitions too large");
    }
    const uint64_t temp = n_feature_comp * n_partitions;
    if (temp > UINT64_MAX / n_partitions)
    {
        throw std::range_error("ARI count would exceed maximum representable value: n_feature_comp * n_partitions * n_partitions too large");
    }
}

/**
 * @brief Calculate the number of feature comparisons
 *
 * @param n_features Number of features
 * @return uint64_t Number of feature comparisons
 * @throws std::range_error if calculation would exceed maximum representable value
 */
uint64_t calculate_feature_comparisons(const size_t n_features)
{
    check_feature_comp_bounds(n_features);
    return n_features * (n_features - 1) / 2;
}

/**
 * @brief Calculate the total number of ARIs to process
 *
 * @param n_feature_comp Number of feature comparisons
 * @param n_partitions Number of partitions
 * @return uint64_t Total number of ARIs
 * @throws std::range_error if calculation would exceed maximum representable value
 */
uint64_t calculate_total_aris(const uint64_t n_feature_comp, const size_t n_partitions)
{
    check_ari_count_bounds(n_feature_comp, n_partitions);
    return n_feature_comp * n_partitions * n_partitions;
}

template <typename T, typename R>
auto compute_coef(const py::array_t<T, py::array::c_style> &parts,
                  const size_t n_features,
                  const size_t n_partitions,
                  const size_t n_objects,
                  const bool return_parts,
                  std::optional<uint32_t> pvalue_n_perms) -> py::object
{
    /*
     * Input validation (before any device work)
     * -----------------------------------------
     * Reject mismatched shapes and out-of-range partition counts with a Python
     * ValueError instead of launching kernels on inconsistent dimensions.
     */
    {
        py::buffer_info buffer = parts.request();
        if (buffer.format != py::format_descriptor<T>::format())
        {
            throw py::value_error(
                std::string("Partitions array has an incompatible dtype: expected numpy format '") +
                py::format_descriptor<T>::format() + "', got '" + buffer.format + "'");
        }
        if (buffer.ndim != 3 ||
            buffer.shape[0] != static_cast<py::ssize_t>(n_features) ||
            buffer.shape[1] != static_cast<py::ssize_t>(n_partitions) ||
            buffer.shape[2] != static_cast<py::ssize_t>(n_objects))
        {
            std::string got = buffer.ndim == 3
                                  ? ("(" + std::to_string(buffer.shape[0]) + ", " +
                                     std::to_string(buffer.shape[1]) + ", " +
                                     std::to_string(buffer.shape[2]) + ")")
                                  : ("ndim=" + std::to_string(buffer.ndim));
            throw py::value_error(
                "Partitions array shape mismatch: expected (n_features, n_partitions, n_objects) = (" +
                std::to_string(n_features) + ", " + std::to_string(n_partitions) + ", " +
                std::to_string(n_objects) + ") but got " + got);
        }
    }
    if (n_partitions == 0)
    {
        throw py::value_error("n_partitions must be greater than 0");
    }
    // max_parts stores partition indices in a uint8_t buffer, so the number of
    // partitions per feature must be representable in a uint8_t.
    if (n_partitions > 255)
    {
        throw py::value_error("n_partitions must be <= 255 (partition indices are stored as uint8); got " +
                              std::to_string(n_partitions));
    }

    // Check for CCC_GPU_LOGGING environment variable to enable debug logging
    const char* logging_env = std::getenv("CCC_GPU_LOGGING");
    if (logging_env != nullptr) {
        spdlog::set_level(spdlog::level::debug);
        // Check CUDA info
        spdlog::debug("CUDA Device Info:");
        print_cuda_device_info();
        spdlog::debug("CUDA Memory Info:");
        print_cuda_memory_info();
    } else {
        // Disable debug logging by default
        spdlog::set_level(spdlog::level::err);
    }

    /*
     * Configuration and Constants
     * --------------------------
     * These values determine the batch processing parameters and memory allocation sizes.
     * They should be tuned based on available GPU memory and performance requirements.
     */
    const uint64_t batch_n_features = 5000;
    const uint64_t batch_n_parts = n_partitions;
    const uint64_t batch_n_feature_comp = batch_n_features * (batch_n_features - 1) / 2;
    const uint64_t batch_n_aris = batch_n_feature_comp * batch_n_parts * batch_n_parts;

    /*
     * Pre-computation of Array Sizes
     * -----------------------------
     * Calculate the total number of comparisons and ARIs to be processed.
     * Includes overflow checks to prevent undefined behavior.
     */
    const uint64_t n_feature_comp = calculate_feature_comparisons(n_features);
    const uint64_t n_aris = calculate_total_aris(n_feature_comp, n_partitions);
    const uint64_t reduction_range = n_partitions * n_partitions;

    spdlog::debug("Debug Info:");
    spdlog::debug("  n_features: {}", n_features);
    spdlog::debug("  n_partitions: {}", n_partitions);
    spdlog::debug("  n_objects: {}", n_objects);
    spdlog::debug("  n_feature_comp: {}", n_feature_comp);
    spdlog::debug("  n_aris: {}", n_aris);
    spdlog::debug("  batch_n_aris: {}", batch_n_aris);

    /*
     * Host-side Memory Allocation
     * --------------------------
     * Allocate memory for the final results that will be returned to Python.
     * These arrays store the maximum ARI values and their corresponding partition pairs.
     */
    spdlog::debug("Allocating host memory...");
    spdlog::debug("  Memory before allocation: ");
    size_t before_host_mem = print_host_memory_info();

    // Main result containers
    std::vector<R> cm_values(n_feature_comp, std::numeric_limits<R>::quiet_NaN());
    std::vector<uint8_t> max_parts(n_feature_comp * 2, UINT8_MAX);

    // Optional p-value container
    std::vector<R> cm_pvalues;
    if (pvalue_n_perms.has_value())
    {
        cm_pvalues.resize(n_feature_comp, std::numeric_limits<R>::quiet_NaN());
    }

    spdlog::debug("  Memory after allocation: ");
    size_t after_host_mem = print_host_memory_info();
    spdlog::debug("  Memory used: {} MB", (after_host_mem - before_host_mem));

    /*
     * Device-side Memory Allocation
     * ---------------------------
     * Pre-allocate device memory for batch processing.
     * These vectors are reused for each batch to minimize memory allocation overhead.
     */
    const uint64_t max_batch_feature_comp = batch_n_feature_comp;
    spdlog::debug("Allocating device memory...");
    spdlog::debug("  max_batch_feature_comp: {}", max_batch_feature_comp);
    spdlog::debug("  Memory before allocation: ");
    size_t before_mem = print_cuda_memory_info();

    // Device vectors for batch processing
    thrust::device_vector<R> d_cm_values(max_batch_feature_comp, std::numeric_limits<R>::quiet_NaN());
    thrust::device_vector<uint8_t> d_max_parts(max_batch_feature_comp * 2, UINT8_MAX);

    // Host vectors for batch results
    std::vector<R> batch_cm_values(max_batch_feature_comp);
    std::vector<uint8_t> batch_max_parts(max_batch_feature_comp * 2);

    spdlog::debug("  Memory after allocation: ");
    size_t after_mem = print_cuda_memory_info();
    spdlog::debug("  Memory used: {} MB", (before_mem - after_mem) / 1024 / 1024);

    /*
     * Batch Processing Loop
     * -------------------
     * Process ARIs in batches to manage memory usage and improve performance.
     * Each batch computes a subset of feature comparisons.
     */
    for (uint64_t batch_start = 0; batch_start < n_aris; batch_start += batch_n_aris)
    {
        spdlog::debug("Processing batch {} of {}",
                      (batch_start / batch_n_aris + 1),
                      (n_aris + batch_n_aris - 1) / batch_n_aris);
        spdlog::debug("  Start index: {}", batch_start);
        spdlog::debug("  Batch size: {}", batch_n_aris);
        spdlog::debug("  Memory before batch: ");
        before_mem = print_cuda_memory_info();

        // Calculate the actual batch size for this iteration
        const uint64_t current_batch_size = std::min(batch_n_aris, n_aris - batch_start);
        spdlog::debug("  Current batch size: {}", current_batch_size);

        // Compute ARIs for this batch
        const auto d_aris = ari_core_device<T, R>(
            parts, n_features, n_partitions, n_objects, batch_start, current_batch_size);

        // Configure kernel launch parameters
        const int threadsPerBlock = 128;
        const int numBlocks = current_batch_size / (n_partitions * n_partitions);
        spdlog::debug("  Launching reduction kernel with {} blocks, {} threads per block",
                      numBlocks, threadsPerBlock);

        // Launch kernel to find maximum values and their partition pairs
        findMaxAriKernel<R><<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_aris->data()),
            thrust::raw_pointer_cast(d_max_parts.data()),
            thrust::raw_pointer_cast(d_cm_values.data()),
            n_partitions,
            reduction_range);

        // Check for kernel errors
        cudaError_t kernelError = cudaGetLastError();
        if (kernelError != cudaSuccess)
        {
            throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(kernelError)));
        }

        // Synchronize to ensure kernel completion
        cudaError_t syncError = cudaDeviceSynchronize();
        if (syncError != cudaSuccess)
        {
            throw std::runtime_error("Device synchronization failed: " + std::string(cudaGetErrorString(syncError)));
        }

        // Copy batch results back to host
        thrust::copy(d_cm_values.begin(), d_cm_values.begin() + current_batch_size / (n_partitions * n_partitions),
                     batch_cm_values.begin());
        thrust::copy(d_max_parts.begin(), d_max_parts.begin() + (current_batch_size / (n_partitions * n_partitions)) * 2,
                     batch_max_parts.begin());

        // Update main result arrays with batch results
        for (uint64_t i = 0; i < current_batch_size / (n_partitions * n_partitions); ++i)
        {
            const uint64_t global_idx = batch_start / (n_partitions * n_partitions) + i;
            if (global_idx < n_feature_comp)
            {
                cm_values[global_idx] = batch_cm_values[i];
                max_parts[global_idx * 2] = batch_max_parts[i * 2];
                max_parts[global_idx * 2 + 1] = batch_max_parts[i * 2 + 1];
            }
        }

        spdlog::debug("  Memory after batch: ");
        size_t after_mem = print_cuda_memory_info();
        spdlog::debug("  Memory used in batch: {} MB", (before_mem - after_mem) / 1024 / 1024);
    }

    /*
     * P-Value Computation
     * -------------------
     * Compute p-values using permutation testing if requested.
     */
    if (pvalue_n_perms.has_value() && pvalue_n_perms.value() > 0)
    {
        spdlog::debug("Computing p-values with {} permutations", pvalue_n_perms.value());

        const uint32_t n_perms = pvalue_n_perms.value();
        const uint32_t rand_seed = 42; // Fixed seed for reproducibility

        // Copy partition data to device for p-value computation
        thrust::device_vector<T> d_parts(parts.data(), parts.data() + (n_features * n_partitions * n_objects));

        // Determine the global cluster-count bound (max cluster id + 1) so the
        // permutation kernel's contingency scratch is sized to fit any k. This
        // removes the old MAX_CLUSTERS=16 cliff that silently returned ARI 0.0.
        int k_global = static_cast<int>(
            thrust::reduce(d_parts.begin(), d_parts.end(), static_cast<T>(-1), thrust::maximum<T>()) + 1);
        if (k_global < 1) k_global = 1;
        const uint64_t scratch_stride = static_cast<uint64_t>(k_global) * k_global + 2ULL * k_global;

        // Per-permutation contingency/sum scratch. Its size scales as k^2 per
        // permutation, so sizing it by the full n_perms could OOM for large
        // cluster counts. Instead bound it to a memory budget and process the
        // permutations in sub-batches of `perm_batch` (see the launch loop
        // below); large k simply takes more passes rather than exhausting memory.
        size_t free_scratch = 0, total_scratch = 0;
        cudaMemGetInfo(&free_scratch, &total_scratch);
        const uint64_t scratch_elem_bytes = std::max<uint64_t>(scratch_stride * sizeof(int), 1);
        const uint64_t scratch_budget =
            std::min<uint64_t>(free_scratch / 4, static_cast<uint64_t>(1) << 30); // cap at 1 GiB
        uint32_t perm_batch =
            static_cast<uint32_t>(std::max<uint64_t>(1, scratch_budget / scratch_elem_bytes));
        if (perm_batch > n_perms) perm_batch = n_perms;
        thrust::device_vector<int> d_perm_scratch(static_cast<size_t>(perm_batch) * scratch_stride);

        // Allocate device memory for p-value computation
        thrust::device_vector<curandState> d_rand_states(n_perms);
        thrust::device_vector<uint32_t> d_perm_indices(static_cast<size_t>(n_perms) * n_objects);
        thrust::device_vector<R> d_observed_ccc_values(cm_values.begin(), cm_values.end());
        thrust::device_vector<R> d_computed_pvalues(n_feature_comp);

        // Initialize random states
        const uint32_t block_size = 256;
        const uint32_t grid_size_states = (n_perms + block_size - 1) / block_size;

        initRandomStates<<<grid_size_states, block_size>>>(
            thrust::raw_pointer_cast(d_rand_states.data()),
            n_perms,
            rand_seed
        );
        CUDA_CHECK_KERNEL("initRandomStates");

        // Generate permutation indices
        generatePermutations<<<grid_size_states, block_size>>>(
            thrust::raw_pointer_cast(d_rand_states.data()),
            thrust::raw_pointer_cast(d_perm_indices.data()),
            n_perms,
            n_objects
        );
        CUDA_CHECK_KERNEL("generatePermutations");

        /*
         * Memory-bounded permutation storage
         * ----------------------------------
         * Storing every comparison's n_perms permuted CCC values at once costs
         * n_feature_comp * n_perms * sizeof(R), which OOMs for large inputs that
         * the coefficient-only path handles fine. Batch the comparisons into
         * chunks bounded by available memory and compute p-values per chunk.
         */
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const size_t perm_value_bytes = std::max<size_t>(static_cast<size_t>(n_perms) * sizeof(R), 1);
        const size_t budget = std::min<size_t>(free_mem / 4, static_cast<size_t>(512) * 1024 * 1024);
        uint64_t chunk_comps = static_cast<uint64_t>(budget / perm_value_bytes);
        if (chunk_comps < 1) chunk_comps = 1;
        if (chunk_comps > n_feature_comp) chunk_comps = n_feature_comp;

        thrust::device_vector<R> d_perm_ccc_values(static_cast<size_t>(chunk_comps) * n_perms);

        spdlog::debug("Computing permutation CCC values for {} feature comparisons ({} per chunk, k={})",
                      n_feature_comp, chunk_comps, k_global);

        for (uint64_t chunk_start = 0; chunk_start < n_feature_comp; chunk_start += chunk_comps)
        {
            const uint64_t chunk_len = std::min(chunk_comps, n_feature_comp - chunk_start);

            for (uint64_t local = 0; local < chunk_len; ++local)
            {
                const uint64_t comp_idx = chunk_start + local;

                // Get feature indices for this comparison (64-bit to avoid
                // truncating comp_idx once n_feature_comp exceeds 2^32).
                uint64_t feat_i, feat_j;
                get_coords_from_index(static_cast<uint64_t>(n_features), comp_idx, feat_i, feat_j);

                R *out_slice = thrust::raw_pointer_cast(d_perm_ccc_values.data()) +
                               local * static_cast<uint64_t>(n_perms);

                // Validate feature indices to prevent memory corruption. This
                // should not trigger for a valid comp_idx; fill the slice so the
                // stale reuse of the batch buffer cannot corrupt the p-value.
                if (feat_i >= n_features || feat_j >= n_features)
                {
                    spdlog::error("Invalid feature indices: feat_i={}, feat_j={}, n_features={}",
                                  feat_i, feat_j, n_features);
                    thrust::fill(d_perm_ccc_values.begin() + local * static_cast<uint64_t>(n_perms),
                                 d_perm_ccc_values.begin() + (local + 1) * static_cast<uint64_t>(n_perms),
                                 static_cast<R>(0));
                    continue;
                }

                const T* d_parts_i = thrust::raw_pointer_cast(d_parts.data()) + feat_i * n_partitions * n_objects;
                const T* d_parts_j = thrust::raw_pointer_cast(d_parts.data()) + feat_j * n_partitions * n_objects;

                // Count valid partitions on host (small operation)
                uint32_t valid_count_i = 0, valid_count_j = 0;
                const T* host_parts_i = parts.data() + feat_i * n_partitions * n_objects;
                const T* host_parts_j = parts.data() + feat_j * n_partitions * n_objects;
                for (uint32_t p = 0; p < n_partitions; ++p)
                {
                    if (host_parts_i[p * n_objects] >= 0) valid_count_i++;
                    if (host_parts_j[p * n_objects] >= 0) valid_count_j++;
                }

                // Permute the feature that generated MORE valid partitions,
                // matching the CPU reference (impl.py compute_coef).
                const T* d_parts_to_permute = (valid_count_i > valid_count_j) ? d_parts_i : d_parts_j;
                const T* d_parts_fixed      = (valid_count_i > valid_count_j) ? d_parts_j : d_parts_i;

                // Process permutations in sub-batches bounded by the scratch budget.
                for (uint32_t p_off = 0; p_off < n_perms; p_off += perm_batch)
                {
                    const uint32_t p_cnt = std::min(perm_batch, n_perms - p_off);
                    const uint32_t p_grid = (p_cnt + block_size - 1) / block_size;
                    computePermutationCCC<<<p_grid, block_size>>>(
                        d_parts_fixed,
                        d_parts_to_permute,
                        thrust::raw_pointer_cast(d_perm_indices.data()),
                        out_slice,
                        p_off,
                        p_cnt,
                        n_partitions,
                        n_objects,
                        k_global,
                        thrust::raw_pointer_cast(d_perm_scratch.data()),
                        scratch_stride
                    );
                    CUDA_CHECK_KERNEL("computePermutationCCC");
                }
            }

            // Compute p-values for the comparisons in this chunk
            const uint32_t pval_grid_size = static_cast<uint32_t>((chunk_len + block_size - 1) / block_size);
            computePValues<<<pval_grid_size, block_size>>>(
                thrust::raw_pointer_cast(d_perm_ccc_values.data()),
                thrust::raw_pointer_cast(d_observed_ccc_values.data()) + chunk_start,
                thrust::raw_pointer_cast(d_computed_pvalues.data()) + chunk_start,
                chunk_len,
                n_perms
            );
            CUDA_CHECK_KERNEL("computePValues");
        }

        // Copy p-values back to host
        thrust::copy(d_computed_pvalues.begin(), d_computed_pvalues.end(), cm_pvalues.begin());

        // Set p-values to NaN where corresponding CCC values are NaN
        for (uint64_t i = 0; i < n_feature_comp; ++i)
        {
            if (std::isnan(cm_values[i]))
            {
                cm_pvalues[i] = std::numeric_limits<R>::quiet_NaN();
            }
        }

        spdlog::debug("P-value computation completed successfully");
    }

    /*
     * Prepare Return Values
     * -------------------
     * Convert the results to numpy arrays and return them as a tuple.
     */
    const auto cm_values_py = py::array_t<R>(cm_values.size(), cm_values.data());
    const auto cm_pvalues_py = pvalue_n_perms.has_value()
                                   ? py::object(py::array_t<R>(cm_pvalues.size(), cm_pvalues.data()))
                                   : py::object(py::none());
    const auto max_parts_py = py::array_t<uint8_t>(max_parts.size(), max_parts.data()).reshape({n_feature_comp, static_cast<uint64_t>(2)});

    return py::make_tuple(
        cm_values_py,
        cm_pvalues_py,
        max_parts_py);
}

// Below is the explicit instantiation of the ari template function.
//
// Generally people would write the implementation of template classes and functions in the header file. However, we
// separate the implementation into a .cpp file to make things clearer. In order to make the compiler know the
// implementation of the template functions, we need to explicitly instantiate them here, so that they can be picked up
// by the linker.
template auto compute_coef<int16_t, float>(const py::array_t<int16_t, py::array::c_style> &parts,
                                          const size_t n_features,
                                          const size_t n_partitions,
                                          const size_t n_objects,
                                          const bool return_parts,
                                          std::optional<unsigned int> pvalue_n_perms) -> py::object;
