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
#include "utils.cuh"
namespace py = pybind11;

template <typename T>
__global__ void findMaxAriKernel(const T* aris,
                                // unsigned int* max_parts,
                                T* cm_values,
                                const int n_partitions,
                                const int reduction_range) {
    // Each block handles one feature comparison
    const int comp_idx = blockIdx.x;

    // Calculate start index for this feature comparison
    const int reduce_start_idx = comp_idx * reduction_range;

    // Thread-local variables for reduction
    int max_idx = -1;
    T max_val = -1.0f;

    // Have threads collaboratively process all partition pairs
    for (int i = threadIdx.x; i < reduction_range; i += blockDim.x) {
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
        // unsigned int m, n;
        // m = min_idx / n_partitions;
        // n = min_idx % n_partitions;

        // max_parts[comp_idx >> 1] = m;
        // max_parts[comp_idx >> 1 + 1] = n;
    }
}


/**
 * @brief Helper function to process and validate input numpy array
 * @param parts Input numpy array to process
 * @return Pointer to the underlying data
 */
template <typename T>
T *process_input_array(const py::array_t<T, py::array::c_style> &parts)
{
    py::buffer_info buffer = parts.request();
    if (buffer.format != py::format_descriptor<T>::format())
    {
        throw std::runtime_error("Incompatible format: expected an int array!");
    }
    if (buffer.ndim != 3)
    {
        throw std::runtime_error("Incompatible buffer dimension!");
    }
    return static_cast<T *>(buffer.ptr);
}

template <typename T>
auto compute_coef(const py::array_t<T, py::array::c_style> &parts,
                  const size_t n_features,
                  const size_t n_partitions,
                  const size_t n_objects,
                  const size_t max_k,
                  const bool return_parts,
                  std::optional<unsigned int> pvalue_n_perms) -> py::object
{
    // Pre-computation
    using parts_dtype = T;
    using out_dtype = float;
    const int n_feature_comp = n_features * (n_features - 1) / 2;
    const int n_aris = n_feature_comp * n_partitions * n_partitions;
    const auto n_elems_per_feat = n_partitions * n_objects;
    const auto reduction_range = n_partitions * n_partitions;
    // Parse the input parts
    const auto parts_ptr = process_input_array(parts);
    // Input validation
    if (!parts_ptr || n_features == 0 || n_partitions == 0 || n_objects == 0)
    {
        throw std::invalid_argument("Invalid input parameters");
    }


    // Allocate host memory for results
    std::vector<out_dtype> cm_values(n_feature_comp);
    std::vector<int32_t> max_parts(n_feature_comp * 2);
    std::vector<out_dtype> cm_pvalues;

    const size_t n_live_reductions = 0; // Number of stream groups running concurrently, we need to sync them to get partial results for cm_values

    const auto n_streams = n_partitions * n_partitions; // k * k partition aris
    // Each stream group is responsible for all ARI computations between two features
    std::vector<cudaStream_t> streams(n_streams);
    for (int s = 0; s < n_streams; s++)
    {
        // Create streams, each stream is responsible for one ARI computation between two features
        CUDA_CHECK_MANDATORY(cudaStreamCreate(&streams[s]));
    }

    // Compute the aris across all features and perform reduction on the go
    for (size_t range_ari_idx = 0; range_ari_idx < n_aris; range_ari_idx+=reduction_range) {
        // Allocate page-locked memory for ARI values
        const parts_dtype *h_aris;
        // Host page-locked memory
        std::vector<parts_dtype *> h_part0s(n_streams);
        std::vector<parts_dtype *> h_part1s(n_streams);
        // Device memory
        // TODO: OPTIMIZE: maybe it's better to put the whole parts arrays on the device
        std::vector<parts_dtype *> d_part0s(n_streams);
        std::vector<parts_dtype *> d_part1s(n_streams);
        // std::vector<out_dtype *> d_aris(n_streams);
        CUDA_CHECK_MANDATORY(cudaHostAlloc((void **)&h_aris, reduction_range * sizeof(out_dtype), cudaHostAllocDefault));
        // Copy part0 and part1 to each stream
        for (int s = 0; s < n_streams; ++s)
        {
            auto h_part0 = h_part0s[s];
            auto h_part1 = h_part1s[s];
            auto d_part0 = d_part0s[s];
            auto d_part1 = d_part1s[s];
            // auto d_ari = d_aris[s];
            // Allocate page-locked memory for part0 and part1
            CUDA_CHECK_MANDATORY(cudaHostAlloc((void **)&h_part0, reduction_range * sizeof(parts_dtype), cudaHostAllocDefault));
            CUDA_CHECK_MANDATORY(cudaHostAlloc((void **)&h_part1, reduction_range * sizeof(parts_dtype), cudaHostAllocDefault));
            CUDA_CHECK_MANDATORY(cudaHostAlloc((void **)&h_aris, reduction_range * sizeof(out_dtype), cudaHostAllocDefault));
            CUDA_CHECK_MANDATORY(cudaMalloc((void **)&d_part0, reduction_range * sizeof(parts_dtype)));
            CUDA_CHECK_MANDATORY(cudaMalloc((void **)&d_part1, reduction_range * sizeof(parts_dtype)));
            // Single ARI value per stream. We can also try one stream for all ARI values within the reduction range
            // CUDA_CHECK_MANDATORY(cudaMalloc((void **)&d_aris, 1 * sizeof(out_dtype)));
            // Compute indices
            const auto feature_comp_flat_idx = range_ari_idx;
            const auto part_pair_flat_idx = s;
            uint32_t i, j;
            get_coords_from_index(n_features, feature_comp_flat_idx, &i, &j);
            uint32_t m, n;
            unravel_index(part_pair_flat_idx, n_partitions, &m, &n);
            // Copy data from parts to page-locked memory
            const auto part0_start_idx = parts_ptr + i * n_elems_per_feat + m * n_objects;
            const auto part1_start_idx = parts_ptr + j * n_elems_per_feat + n * n_objects;
            for (int k = 0; k < n_objects; ++k) {
                h_part0[k] = part0_start_idx[k];
                h_part1[k] = part1_start_idx[k];
            }
            // Copy the locked memory to the device, async
            CUDA_CHECK_MANDATORY(cudaMemcpyAsync(d_part0, h_part0, reduction_range * sizeof(parts_dtype), cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK_MANDATORY(cudaMemcpyAsync(d_part1, h_part1, reduction_range * sizeof(parts_dtype), cudaMemcpyHostToDevice, streams[s]));
        }
        // Invoke the kernel
        for (int s = 0; s < n_streams; ++s) {
            auto d_part0 = d_part0s[s];
            auto d_part1 = d_part1s[s];
            ari_core_scalar(d_part0, d_part1, n_objects, max_k, s, streams[s], h_aris);
        }

        // Wait for all streams to finish
        for (int s = 0; s < n_streams; ++s) {
            CUDA_CHECK_MANDATORY(cudaStreamSynchronize(streams[s]));
        }

        // Get the maximum ARI value and its index in array h_aris
        out_dtype max_ari = -1.0f;
        int32_t max_ari_idx = -1;
        for (int s = 0; s < n_streams; ++s) {
            if (h_aris[s] > max_ari) {
                max_ari = h_aris[s];
                max_ari_idx = s;
            }
        }
        cm_values[range_ari_idx] = max_ari;
        // Unravel the index to get partition indices
        uint32_t m, n;
        unravel_index(max_ari_idx, n_partitions, &m, &n);
        max_parts[range_ari_idx >> 1] = m;
        max_parts[range_ari_idx >> 1 + 1] = n;

        // Clean up
        for (int s = 0; s < n_streams; ++s) {
            // Destroy the stream
            CUDA_CHECK_MANDATORY(cudaStreamDestroy(streams[s]));
            // Free the memory
            CUDA_CHECK_MANDATORY(cudaFreeHost(h_part0s[s]));
            CUDA_CHECK_MANDATORY(cudaFreeHost(h_part1s[s]));
            // CUDA_CHECK_MANDATORY(cudaFreeHost(h_aris[s]));
            CUDA_CHECK_MANDATORY(cudaFree(d_part0s[s]));
            CUDA_CHECK_MANDATORY(cudaFree(d_part1s[s]));
            // CUDA_CHECK_MANDATORY(cudaFree(d_aris[s]));
        }
    }

    // P-valued results
    if (pvalue_n_perms.has_value()) {
        cm_pvalues.resize(n_feature_comp, std::numeric_limits<out_dtype>::quiet_NaN());
    }

    // Allocate py::arrays for the results
    // const auto max_parts_py = py::array_t<unsigned int>(max_parts.size(), max_parts.data()).reshape({n_feature_comp, 2});
    const auto cm_values_py = py::array_t<out_dtype>(cm_values.size(), cm_values.data());
    const auto cm_pvalues_py = pvalue_n_perms.has_value()
        ? py::object(py::array_t<out_dtype>(cm_pvalues.size(), cm_pvalues.data()))
        : py::object(py::none());

    // Return the results as a tuple
    return py::make_tuple(
        cm_values_py,
        cm_pvalues_py,
        py::object(py::none())
        // max_parts_py
    );
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
template auto compute_coef<int8_t>(const py::array_t<int8_t, py::array::c_style> &parts,
                                const size_t n_features,
                                const size_t n_partitions,
                                const size_t n_objects,
                                const size_t max_k,
                                const bool return_parts,
                                std::optional<unsigned int> pvalue_n_perms) -> py::object;
