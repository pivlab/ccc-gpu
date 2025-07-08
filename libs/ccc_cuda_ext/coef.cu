#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <spdlog/spdlog.h>

#include <execution>
#include <iostream>
#include <iomanip>
#include <limits>
#include <optional>
#include <vector>
#include <algorithm>
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
            max_parts[comp_idx * 2] = UINT8_MAX;
            max_parts[comp_idx * 2 + 1] = UINT8_MAX;
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

    // Perform ArgMax reduction - this finds the maximum value and its corresponding index
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
    spdlog::set_level(spdlog::level::debug);
    // Check CUDA info
    spdlog::debug("CUDA Device Info:");
    print_cuda_device_info();
    spdlog::debug("CUDA Memory Info:");
    print_cuda_memory_info();

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
template auto compute_coef<int8_t, float>(const py::array_t<int8_t, py::array::c_style> &parts,
                                          const size_t n_features,
                                          const size_t n_partitions,
                                          const size_t n_objects,
                                          const bool return_parts,
                                          std::optional<unsigned int> pvalue_n_perms) -> py::object;
