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

// Debug mode macro - set to 1 to enable debug output, 0 to disable
#define DEBUG_MODE 1

template <typename T>
__global__ void findMaxAriKernel(const T *aris,
                                 // unsigned int* max_parts,
                                 T *cm_values,
                                 const int n_partitions,
                                 const int reduction_range)
{
    // Each block handles one feature comparison
    const uint64_t comp_idx = blockIdx.x;

    // Calculate start index for this feature comparison
    const uint64_t reduce_start_idx = comp_idx * reduction_range;

    // Thread-local variables for reduction
    uint64_t max_idx = UINT64_MAX;
    T max_val = -1.0f;
    bool has_nan = false;

    // Have threads collaboratively process all partition pairs
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

        if (val > max_val)
        {
            max_val = val;
            max_idx = i;
        }
    }

    // If thread 0 found NaN, set output to NaN and return
    if (threadIdx.x == 0 && has_nan)
    {
        cm_values[comp_idx] = NAN;
        return;
    }

    // Shared memory for block reduction
    __shared__ typename cub::BlockReduce<T, 128>::TempStorage temp_storage_val;
    __shared__ typename cub::BlockReduce<uint64_t, 128>::TempStorage temp_storage_idx;

    // Pair-wise reduction within the block
    struct
    {
        T val;
        uint64_t idx;
    } in, out;

    in.val = max_val;
    in.idx = max_idx;

    // Find the maximum value and its index within the block
    T max_block_val = cub::BlockReduce<T, 128>(temp_storage_val).Reduce(in.val, cub::Max());

    // Only threads with the max value participate in index selection
    __syncthreads();

    uint64_t selected_idx = UINT64_MAX;
    if (in.val == max_block_val)
    {
        selected_idx = in.idx;
    }

    // Get the smallest valid index of the max value
    uint64_t min_idx = cub::BlockReduce<uint64_t, 128>(temp_storage_idx).Reduce(selected_idx, [](uint64_t a, uint64_t b)
                                                                                { return (a == UINT64_MAX) ? b : ((b == UINT64_MAX) ? a : min(a, b)); });

    // Thread 0 writes the results
    if (threadIdx.x == 0)
    {

        cm_values[comp_idx] = max_block_val > 0.0f ? max_block_val : 0.0f;

        // Unravel the index to get partition indices
        // unsigned int m, n;
        // m = min_idx / n_partitions;
        // n = min_idx % n_partitions;

        // max_parts[comp_idx >> 1] = m;
        // max_parts[comp_idx >> 1 + 1] = n;
    }
}

// TODO: Add mode check to decide whether to do batch processing or not
template <typename T, typename R>
auto compute_coef(const py::array_t<T, py::array::c_style> &parts,
                  const size_t n_features,
                  const size_t n_partitions,
                  const size_t n_objects,
                  const bool return_parts,
                  std::optional<unsigned int> pvalue_n_perms) -> py::object
{
    // Check CUDA info
    spdlog::debug("CUDA Device Info:");
    print_cuda_device_info();
    spdlog::debug("CUDA Memory Info:");
    print_cuda_memory_info();

    // Batch-computing configs, to be tuned and dynamically set based on the GPU memory size
    const uint64_t batch_n_features = 5000;
    const uint64_t batch_n_parts = n_partitions; // k from 2 to 10
    const uint64_t batch_n_feature_comp = batch_n_features * (batch_n_features - 1) / 2;
    const uint64_t batch_n_aris = batch_n_feature_comp * batch_n_parts * batch_n_parts;

    // Pre-computation
    // Check for overflow in n_feature_comp calculation
    if (n_features > 1 && n_features > UINT64_MAX / (n_features - 1))
    {
        throw std::overflow_error("Overflow in n_feature_comp calculation: n_features too large");
    }
    const uint64_t n_feature_comp = n_features * (n_features - 1) / 2;

    // Check for overflow in n_aris calculation
    if (n_feature_comp > UINT64_MAX / n_partitions)
    {
        throw std::overflow_error("Overflow in n_aris calculation: n_feature_comp * n_partitions too large");
    }
    const uint64_t temp = n_feature_comp * n_partitions;
    if (temp > UINT64_MAX / n_partitions)
    {
        throw std::overflow_error("Overflow in n_aris calculation: n_feature_comp * n_partitions * n_partitions too large");
    }
    const uint64_t n_aris = temp * n_partitions;
    const uint64_t reduction_range = n_partitions * n_partitions;

    spdlog::debug("Debug Info:");
    spdlog::debug("  n_features: {}", n_features);
    spdlog::debug("  n_partitions: {}", n_partitions);
    spdlog::debug("  n_objects: {}", n_objects);
    spdlog::debug("  n_feature_comp: {}", n_feature_comp);
    spdlog::debug("  n_aris: {}", n_aris);
    spdlog::debug("  batch_n_aris: {}", batch_n_aris);

    // Allocate host memory for results
    spdlog::debug("Allocating host memory...");
    spdlog::debug("  Memory before allocation: ");
    size_t before_host_mem = print_host_memory_info();

    std::vector<R> cm_values(n_feature_comp, std::numeric_limits<R>::quiet_NaN());
    std::vector<R> cm_pvalues;

    if (pvalue_n_perms.has_value())
    {
        cm_pvalues.resize(n_feature_comp, std::numeric_limits<R>::quiet_NaN());
    }

    spdlog::debug("  Memory after allocation: ");
    size_t after_host_mem = print_host_memory_info();
    spdlog::debug("  Memory used: {} MB", (after_host_mem - before_host_mem));

    // Pre-allocate device memory for the maximum batch size
    const uint64_t max_batch_feature_comp = batch_n_feature_comp;
    spdlog::debug("Allocating device memory...");
    spdlog::debug("  max_batch_feature_comp: {}", max_batch_feature_comp);
    spdlog::debug("  Memory before allocation: ");
    size_t before_mem = print_cuda_memory_info();

    thrust::device_vector<R> d_cm_values(max_batch_feature_comp, std::numeric_limits<R>::quiet_NaN());
    std::vector<R> batch_cm_values(max_batch_feature_comp);

    spdlog::debug("  Memory after allocation: ");
    size_t after_mem = print_cuda_memory_info();
    spdlog::debug("  Memory used: {} MB", (before_mem - after_mem) / 1024 / 1024);

    // Process ARIs in batches
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

        // Compute the ARIs for this batch
        const auto d_aris = ari_core_device<T, R>(
            parts, n_features, n_partitions, n_objects, batch_start, current_batch_size);

        // Configure kernel launch parameters for this batch
        const int threadsPerBlock = 128;
        const int numBlocks = current_batch_size / (n_partitions * n_partitions);
        spdlog::debug("  Launching reduction kernel with {} blocks, {} threads per block",
                        numBlocks, threadsPerBlock);

        // Launch kernel to find maximum values on device for this batch
        findMaxAriKernel<R><<<numBlocks, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_aris->data()),
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

        // Copy reduced results back to host
        thrust::copy(d_cm_values.begin(), d_cm_values.begin() + current_batch_size / (n_partitions * n_partitions),
                        batch_cm_values.begin());

        // Update the main cm_values array with the batch results
        for (uint64_t i = 0; i < current_batch_size / (n_partitions * n_partitions); ++i)
        {
            const uint64_t global_idx = batch_start / (n_partitions * n_partitions) + i;
            if (global_idx < n_feature_comp)
            {
                cm_values[global_idx] = batch_cm_values[i];
            }
        }

        spdlog::debug("  Memory after batch: ");
        size_t after_mem = print_cuda_memory_info();
        spdlog::debug("  Memory used in batch: {} MB", (before_mem - after_mem) / 1024 / 1024);
    }

    // Allocate py::arrays for the results
    const auto cm_values_py = py::array_t<R>(cm_values.size(), cm_values.data());
    const auto cm_pvalues_py = pvalue_n_perms.has_value()
                                   ? py::object(py::array_t<R>(cm_pvalues.size(), cm_pvalues.data()))
                                   : py::object(py::none());

    // Return the results as a tuple
    return py::make_tuple(
        cm_values_py,
        cm_pvalues_py,
        py::object(py::none()));
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
