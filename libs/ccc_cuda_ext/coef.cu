#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
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

        // Convert linear index to partition pair
        unsigned int m = aggregate.key / n_partitions;
        unsigned int n = aggregate.key % n_partitions;

        // Store the partition pair
        max_parts[comp_idx * 2] = m;
        max_parts[comp_idx * 2 + 1] = n;
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
 * Optimized for GPU execution with smaller memory footprint
 * @param part_i First partition
 * @param part_j Second partition (will be permuted)
 * @param perm Permutation indices
 * @param n_objects Number of objects
 * @return ARI value
 */
template <typename T, typename R>
__device__ R computePermutedARI(const T *part_i, const T *part_j, 
                               const uint32_t *perm, const uint32_t n_objects)
{
    // Quick validation
    if (n_objects == 0) return 0.0f;
    
    // For GPU efficiency, limit the maximum number of clusters
    const int MAX_CLUSTERS = 16;
    
    // Find the maximum cluster IDs
    T max_i = 0, max_j = 0;
    for (uint32_t idx = 0; idx < n_objects; ++idx)
    {
        T cluster_i = part_i[idx];
        T cluster_j = part_j[perm[idx]];
        
        if (cluster_i >= 0 && cluster_i > max_i) max_i = cluster_i;
        if (cluster_j >= 0 && cluster_j > max_j) max_j = cluster_j;
    }
    
    const int k = max(max_i, max_j) + 1;
    if (k <= 0 || k > MAX_CLUSTERS) return 0.0f;
    
    // Use a more memory-efficient approach with smaller arrays
    int contingency[MAX_CLUSTERS * MAX_CLUSTERS]; // Flatten the matrix
    int sum_rows[MAX_CLUSTERS];
    int sum_cols[MAX_CLUSTERS];
    
    // Initialize arrays
    for (int i = 0; i < k * k; ++i) contingency[i] = 0;
    for (int i = 0; i < k; ++i) { sum_rows[i] = 0; sum_cols[i] = 0; }
    
    // Build contingency matrix and compute sums in single pass
    int sum_squares = 0;
    for (uint32_t idx = 0; idx < n_objects; ++idx)
    {
        T cluster_i = part_i[idx];
        T cluster_j = part_j[perm[idx]];
        
        if (cluster_i >= 0 && cluster_j >= 0 && cluster_i < k && cluster_j < k)
        {
            int cont_idx = cluster_i * k + cluster_j;
            contingency[cont_idx]++;
            int val = contingency[cont_idx];
            
            // Update sum_squares incrementally
            sum_squares += 2 * val - 1; // val^2 - (val-1)^2 = 2*val - 1
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
    
    // Compute combination sums
    int sum_comb_c = 0, sum_comb_k = 0;
    for (int i = 0; i < k; ++i)
    {
        if (sum_rows[i] > 1)
            sum_comb_c += (sum_rows[i] * (sum_rows[i] - 1)) / 2;
        if (sum_cols[i] > 1)
            sum_comb_k += (sum_cols[i] * (sum_cols[i] - 1)) / 2;
    }
    
    int sum_comb_ck = (sum_squares - n_objects) / 2;
    
    // Compute ARI with improved numerical stability
    if (sum_comb_c == 0 && sum_comb_k == 0)
    {
        return (sum_comb_ck == 0) ? 1.0f : 0.0f;
    }
    
    // Use double precision for intermediate calculations to avoid overflow
    double n_choose_2 = static_cast<double>(n_objects) * (n_objects - 1) / 2.0;
    double expected_index = static_cast<double>(sum_comb_c) * sum_comb_k / n_choose_2;
    double max_index = (static_cast<double>(sum_comb_c) + sum_comb_k) / 2.0;
    
    if (fabs(max_index - expected_index) < 1e-10)
    {
        return 0.0f;
    }
    
    double ari = (static_cast<double>(sum_comb_ck) - expected_index) / (max_index - expected_index);
    return static_cast<R>(ari);
}

/**
 * @brief CUDA kernel to compute CCC values for permuted partitions
 * Optimized with better error handling and bounds checking
 * @param parts_i Partitions for feature i
 * @param parts_j Partitions for feature j
 * @param perm_indices Permutation indices
 * @param perm_ccc_values Output array for permutation CCC values
 * @param n_perms Number of permutations
 * @param n_partitions Number of partitions per feature
 * @param n_objects Number of objects
 */
template <typename T, typename R>
__global__ void computePermutationCCC(const T *parts_i, const T *parts_j, 
                                     const uint32_t *perm_indices, R *perm_ccc_values,
                                     const uint32_t n_perms, const uint32_t n_partitions,
                                     const uint32_t n_objects)
{
    uint32_t perm_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (perm_idx >= n_perms) return;
    
    // Initialize with a reasonable default value
    R max_ari = 0.0f;
    bool found_valid_ari = false;
    
    // Get permutation for this thread
    const uint32_t *perm = &perm_indices[perm_idx * n_objects];
    
    // Validate permutation indices (basic sanity check)
    bool valid_perm = true;
    for (uint32_t k = 0; k < min(n_objects, 10u); ++k) // Check first 10 for efficiency
    {
        if (perm[k] >= n_objects)
        {
            valid_perm = false;
            break;
        }
    }
    
    if (!valid_perm)
    {
        if (perm_idx < n_perms)  // Add bounds check
            perm_ccc_values[perm_idx] = 0.0f;
        return;
    }
    
    // Compute ARI for all partition pairs with this permutation
    for (uint32_t i = 0; i < n_partitions; ++i)
    {
        for (uint32_t j = 0; j < n_partitions; ++j)
        {
            const T *part_i = &parts_i[i * n_objects];
            const T *part_j = &parts_j[j * n_objects];
            
            // Check for invalid partitions (categorical markers or singletons)
            if (part_i[0] < 0 || part_j[0] < 0)
            {
                continue;
            }
            
            // Compute ARI between part_i and permuted part_j
            R ari_value = computePermutedARI<T, R>(part_i, part_j, perm, n_objects);
            
            // Handle NaN/inf values
            if (isfinite(ari_value))
            {
                if (!found_valid_ari || ari_value > max_ari)
                {
                    max_ari = ari_value;
                    found_valid_ari = true;
                }
            }
        }
    }
    
    // Store result (0.0 if no valid ARI was computed)
    if (perm_idx < n_perms)  // Add bounds check
        perm_ccc_values[perm_idx] = found_valid_ari ? max_ari : 0.0f;
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
                              R *pvalues, const uint32_t n_comparisons, const uint32_t n_perms)
{
    uint32_t comp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp_idx < n_comparisons)
    {
        const R observed_value = observed_ccc_values[comp_idx];
        const R *perm_values = &perm_ccc_values[comp_idx * n_perms];
        
        uint32_t count = 0;
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
        spdlog::set_level(spdlog::level::error);
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
        
        // Allocate device memory for p-value computation
        thrust::device_vector<curandState> d_rand_states(n_perms);
        thrust::device_vector<uint32_t> d_perm_indices(n_perms * n_objects);
        thrust::device_vector<R> d_perm_ccc_values(n_feature_comp * n_perms);
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
        CUDA_CHECK_MANDATORY(cudaDeviceSynchronize());
        
        // Generate permutation indices
        generatePermutations<<<grid_size_states, block_size>>>(
            thrust::raw_pointer_cast(d_rand_states.data()),
            thrust::raw_pointer_cast(d_perm_indices.data()),
            n_perms,
            n_objects
        );
        CUDA_CHECK_MANDATORY(cudaDeviceSynchronize());
        
        // Compute CCC values for each feature pair with permutations
        spdlog::debug("Computing permutation CCC values for {} feature comparisons", n_feature_comp);
        
        for (uint64_t comp_idx = 0; comp_idx < n_feature_comp; ++comp_idx)
        {
            // Get feature indices for this comparison
            unsigned int feat_i, feat_j;
            get_coords_from_index(static_cast<unsigned int>(n_features), 
                                static_cast<unsigned int>(comp_idx), feat_i, feat_j);
            
            // Calculate device pointers for the two features
            const T* d_parts_i = thrust::raw_pointer_cast(d_parts.data()) + feat_i * n_partitions * n_objects;
            const T* d_parts_j = thrust::raw_pointer_cast(d_parts.data()) + feat_j * n_partitions * n_objects;
            
            // Count valid partitions on host (since this is a small operation)
            uint32_t valid_count_i = 0, valid_count_j = 0;
            const T* host_parts_i = parts.data() + feat_i * n_partitions * n_objects;
            const T* host_parts_j = parts.data() + feat_j * n_partitions * n_objects;
            
            // Validate feature indices to prevent memory corruption
            if (feat_i >= n_features || feat_j >= n_features)
            {
                spdlog::error("Invalid feature indices: feat_i={}, feat_j={}, n_features={}", 
                             feat_i, feat_j, n_features);
                continue; // Skip this comparison
            }
            
            for (uint32_t p = 0; p < n_partitions; ++p)
            {
                if (host_parts_i[p * n_objects] >= 0) valid_count_i++;
                if (host_parts_j[p * n_objects] >= 0) valid_count_j++;
            }
            
            // Select feature with more valid partitions for permutation (CPU logic)
            const T* d_parts_to_permute = (valid_count_i > valid_count_j) ? d_parts_j : d_parts_i;
            const T* d_parts_fixed = (valid_count_i > valid_count_j) ? d_parts_i : d_parts_j;
            
            // Compute CCC values for all permutations of this feature pair
            const uint32_t perm_grid_size = (n_perms + block_size - 1) / block_size;
            
            computePermutationCCC<<<perm_grid_size, block_size>>>(
                d_parts_fixed,
                d_parts_to_permute,
                thrust::raw_pointer_cast(d_perm_indices.data()),
                thrust::raw_pointer_cast(d_perm_ccc_values.data()) + comp_idx * n_perms,
                n_perms,
                n_partitions,
                n_objects
            );
            CUDA_CHECK_MANDATORY(cudaDeviceSynchronize());
        }
        
        // Compute p-values from permutation results
        spdlog::debug("Computing final p-values");
        const uint32_t pval_grid_size = (n_feature_comp + block_size - 1) / block_size;
        
        computePValues<<<pval_grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_perm_ccc_values.data()),
            thrust::raw_pointer_cast(d_observed_ccc_values.data()),
            thrust::raw_pointer_cast(d_computed_pvalues.data()),
            n_feature_comp,
            n_perms
        );
        CUDA_CHECK_MANDATORY(cudaDeviceSynchronize());
        
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

auto example_return_optional_vectors(bool include_first,
                                     bool include_second,
                                     bool include_third) -> py::object
{
    // Example vectors
    std::optional<std::vector<float>> vec1;
    std::optional<std::vector<int>> vec2;
    std::optional<std::vector<double>> vec3;

    // Fill vectors if included
    if (include_first)
    {
        vec1 = std::vector<float>{1.0f, 2.0f, 3.0f};
    }
    if (include_second)
    {
        vec2 = std::vector<int>{4, 5, 6};
    }
    if (include_third)
    {
        vec3 = std::vector<double>{7.0, 8.0, 9.0};
    }

    // Convert to Python objects
    py::object py_vec1 = vec1.has_value() ? py::cast(vec1.value()) : py::none();
    py::object py_vec2 = vec2.has_value() ? py::cast(vec2.value()) : py::none();
    py::object py_vec3 = vec3.has_value() ? py::cast(vec3.value()) : py::none();

    // Return as tuple
    return py::make_tuple(py_vec1, py_vec2, py_vec3);
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
