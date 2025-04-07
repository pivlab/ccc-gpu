#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>

#include <iostream>
#include <limits>
#include <optional>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "coef.cuh"
#include "metrics.cuh"
#include "math.cuh"

namespace py = pybind11;

template <typename T>
__global__ void findMaxAriKernel(const T* aris, unsigned int* max_parts, T* cm_values,
                                const int n_partitions, const int reduce_range) {
    // Each block handles one feature comparison
    const int comp_idx = blockIdx.x;

    // Calculate start index for this feature comparison
    const int reduce_start_idx = comp_idx * reduce_range;

    // Thread-local variables for reduction
    int max_idx = -1;
    T max_val = -1.0f;

    // Have threads collaboratively process all partition pairs
    for (int i = threadIdx.x; i < reduce_range; i += blockDim.x) {
        int idx = reduce_start_idx + i;
        T val = aris[idx];

        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // Shared memory for block reduction
    __shared__ typename cub::BlockReduce<T, 128>::TempStorage temp_storage_val;
    __shared__ typename cub::BlockReduce<int, 128>::TempStorage temp_storage_idx;

    // Pair-wise reduction within the block
    struct {
        T val;
        int idx;
    } in, out;

    in.val = max_val;
    in.idx = max_idx;

    // Find the maximum value and its index within the block
    T max_block_val = cub::BlockReduce<T, 128>(temp_storage_val).Reduce(in.val, cub::Max());

    // Only threads with the max value participate in index selection
    // (using __syncthreads to ensure all threads have computed max_block_val)
    __syncthreads();

    int selected_idx = -1;
    if (in.val == max_block_val) {
        selected_idx = in.idx;
    }

    // Get the smallest valid index of the max value
    int min_idx = cub::BlockReduce<int, 128>(temp_storage_idx).Reduce(
        selected_idx,
        [](int a, int b) { return (a == -1) ? b : ((b == -1) ? a : min(a, b)); }
    );

    // Thread 0 writes the results
    if (threadIdx.x == 0) {
        cm_values[comp_idx] = max_block_val > 0.0f ? max_block_val : 0.0f;

        // Unravel the index to get partition indices
        unsigned int m, n;
        m = min_idx / n_partitions;
        n = min_idx % n_partitions;

        max_parts[comp_idx >> 1] = m;
        max_parts[comp_idx >> 1 + 1] = n;
    }
}

template <typename T>
auto compute_coef(const py::array_t<T, py::array::c_style> &parts,
                  const size_t n_features,
                  const size_t n_partitions,
                  const size_t n_objects,
                  const bool return_parts,
                  std::optional<unsigned int> pvalue_n_perms) -> py::object
{
    // Pre-computation
    using parts_dtype = T;
    using out_dtype = float;
    const int n_feature_comp = n_features * (n_features - 1) / 2;
    const int n_aris = n_feature_comp * n_partitions * n_partitions;
    const auto reduce_range = n_partitions * n_partitions;

    // Compute the aris across all features
    const auto d_aris = ari_core_device<parts_dtype, out_dtype>(parts, n_features, n_partitions, n_objects);

    // Allocate device memory for results
    thrust::device_vector<out_dtype> d_cm_values(n_feature_comp);
    thrust::device_vector<unsigned int> d_max_parts(n_feature_comp * 2);

    // Configure kernel launch parameters
    const int threadsPerBlock = 128;
    const int numBlocks = n_feature_comp;

    // Launch kernel to find maximum values on device
    findMaxAriKernel<<<numBlocks, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_aris->data()),
        thrust::raw_pointer_cast(d_max_parts.data()),
        thrust::raw_pointer_cast(d_cm_values.data()),
        n_partitions,
        reduce_range
    );

    // Allocate host memory for results
    std::vector<out_dtype> cm_values(n_feature_comp);
    std::vector<unsigned int> max_parts(n_feature_comp * 2);
    std::vector<out_dtype> cm_pvalues;

    if (pvalue_n_perms.has_value()) {
        cm_pvalues.resize(n_feature_comp, std::numeric_limits<out_dtype>::quiet_NaN());
    }

    // Copy reduced results back to host
    thrust::copy(d_cm_values.begin(), d_cm_values.end(), cm_values.begin());
    thrust::copy(d_max_parts.begin(), d_max_parts.end(), max_parts.begin());

    // Allocate py::arrays for the results
    const auto max_parts_py = py::array_t<unsigned int>(max_parts.size(), max_parts.data()).reshape({n_feature_comp, 2});
    const auto cm_values_py = py::array_t<out_dtype>(cm_values.size(), cm_values.data());
    const auto cm_pvalues_py = pvalue_n_perms.has_value()
        ? py::object(py::array_t<out_dtype>(cm_pvalues.size(), cm_pvalues.data()))
        : py::object(py::none());

    // Return the results as a tuple
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
template auto compute_coef<int>(const py::array_t<int, py::array::c_style> &parts,
                                const size_t n_features,
                                const size_t n_partitions,
                                const size_t n_objects,
                                const bool return_parts,
                                std::optional<unsigned int> pvalue_n_perms) -> py::object;
