#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <iostream>
#include <cmath>
#include <limits>
#include <optional>
#include <optional>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "coef.cuh"
#include "metrics.cuh"
// #include "utils.cuh"
#include "math.cuh"

namespace py = pybind11;

// /**
//  * @brief API exposed to Python for computing the correlation coefficient and p values using CUDA
//  * @param parts 3D Numpy.NDArray of partitions with shape of (n_features, n_parts, n_objs)
//  * @throws std::invalid_argument if "parts" is invalid
//  * @return float ARI value for each pair of partitions
//  */
template <typename T>
auto compute_coef(const py::array_t<T, py::array::c_style> &parts,
                  const size_t n_features,
                  const size_t n_parts,
                  const size_t n_objs,
                  const bool return_parts,
                  std::optional<unsigned int> pvalue_n_perms) -> py::object
{
    // Pre-computation
    using parts_dtype = T;
    using out_dtype = float;
    const auto n_feature_comp = n_features * (n_features - 1) / 2;
    const auto n_aris = n_feature_comp * n_parts * n_parts;

    // Compute the aris across all features
    const auto d_aris = ari_core_device<parts_dtype, out_dtype>(parts, n_features, n_parts, n_objs);

    // Allocate host memory
    std::vector<out_dtype> h_aris(n_aris);
    std::vector<out_dtype> cm_values(n_feature_comp, std::numeric_limits<out_dtype>::quiet_NaN());
    std::vector<out_dtype> cm_pvalues(n_feature_comp, std::numeric_limits<out_dtype>::quiet_NaN());
    std::vector<unsigned int> max_parts(n_feature_comp * 2, 0);
    // Copy data back to host using -> operator since d_aris is a unique_ptr
    thrust::copy(d_aris->begin(), d_aris->end(), h_aris.begin());

    // Iterate over all feature comparison pairs
    const auto reduce_range = n_parts * n_parts;
    for (unsigned int comp_idx = 0; comp_idx < n_feature_comp; comp_idx++)
    {
        const auto reduce_start_idx = comp_idx * reduce_range;
        const auto reduce_start_iter = h_aris.begin() + reduce_start_idx;
        const auto reduce_end_iter = reduce_start_iter + reduce_range;

        // Compute the maximum ARI value for the current feature pair
        const auto max_ari = std::max_element(reduce_start_iter, reduce_end_iter);
        // Get the flattened index of the maximum ARI value
        const auto max_part_pair_flat_idx = std::distance(reduce_start_iter, max_ari);
        cm_values[comp_idx] = *max_ari;
        // Get the unraveled indices of the partitions
        unsigned int m, n;
        unravel_index(max_part_pair_flat_idx, n_parts, m, n);
        max_parts[comp_idx * 2] = m;
        max_parts[comp_idx * 2 + 1] = n;
    }

    // Return the results as a tuple
    return py::make_tuple(
        py::cast(cm_values),
        py::cast(cm_pvalues),
        py::cast(max_parts));
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
                                const size_t n_parts,
                                const size_t n_objs,
                                const bool return_parts,
                                std::optional<unsigned int> pvalue_n_perms) -> py::object;
