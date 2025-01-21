#include <cuda_runtime.h>
#include <cub/block/block_load.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <iostream>
#include <cmath>
#include <limits>
#include <optional>
#include <assert.h>
#include "utils.cuh"
#include "math.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <vector>

namespace py = pybind11;

// /**
//  * @brief API exposed to Python for computing the correlation coefficient and p values using CUDA
//  * @param parts 3D Numpy.NDArray of partitions with shape of (n_features, n_parts, n_objs)
//  * @throws std::invalid_argument if "parts" is invalid
//  * @return float ARI value for each pair of partitions
//  */
// template <typename T>
// auto compute_coef(const py::array_t<T, py::array::c_style> &parts,
//                  const size_t n_features,
//                  const size_t n_parts,
//                  const size_t n_objs,
//                  const bool return_parts,
//                  std::optional<unsigned int> pvalue_n_perms = std::nullopt) -> void
// {
//     /*
//      * Pre-computation
//      */
//     using parts_dtype = T;
//     using out_dtype = float;
//     const auto n_feature_comp = n_features * (n_features - 1) / 2;
//     const auto n_aris = n_feature_comp * n_parts * n_parts;

//     /*
//      * Memory Allocation
//      */
//     // Allocate host memory
//     thrust::host_vector<out_dtype> h_aris(n_aris);
//     thrust::host_vector<out_dtype> cm_values(n_feature_comp, std::numeric_limits<out_dtype>::quiet_NaN());
//     thrust::host_vector<out_dtype> cm_pvalues(n_feature_comp, std::numeric_limits<out_dtype>::quiet_NaN());
//     thrust::host_vector<unsigned int> max_parts(n_feature_comp * 2, 0);

//     /*
//      * Compute the CCC values
//      */
//     // Compute the aris across all features
//     const auto d_aris = ari_core_device<parts_dtype, out_dtype>(parts, n_features, n_parts, n_objs);

//     // Copy the aris to the host
//     thrust::copy(d_aris->begin(), d_aris->end(), h_aris.begin());

//     const auto reduce_range = n_objs * n_objs;
//     for (unsigned int comp_idx = 0; comp_idx < n_feature_comp; comp_idx++)
//     {
//         const auto reduce_start_idx = comp_idx * reduce_range;
//         const auto reduce_start_iter = h_aris.begin() + reduce_start_idx;
//         const auto reduce_end_iter = h_aris.begin() + reduce_start_idx + reduce_range;

//         // Compute the maximum ARI value for the current feature pair
//         const auto max_ari = thrust::reduce(reduce_start_iter, reduce_end_iter, std::numeric_limits<out_dtype>::quiet_NaN(), thrust::maximum<out_dtype>());
//         // Get the flattened index of the maximum ARI value
//         const auto max_part_pair_flat_idx = thrust::distance(h_aris.begin(), thrust::max_element(reduce_start_iter, reduce_end_iter));
//         cm_values[comp_idx] = max_ari;
//         // Get the unraveled indices of the partitions
//         unsigned int m, n;
//         unravel_index(max_part_pair_flat_idx, n_parts, &m, &n);
//         max_parts[comp_idx * 2] = m;
//         max_parts[comp_idx * 2 + 1] = n;
//     }
// }

auto example_return_optional_vectors(bool include_first = true,
                                        bool include_second = true,
                                        bool include_third = true) -> py::object{
    // Example vectors
    std::optional<std::vector<float>> vec1;
    std::optional<std::vector<int>> vec2;
    std::optional<std::vector<double>> vec3;

    // Fill vectors if included
    if (include_first) {
        vec1 = std::vector<float>{1.0f, 2.0f, 3.0f};
    }
    if (include_second) {
        vec2 = std::vector<int>{4, 5, 6};
    }
    if (include_third) {
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
