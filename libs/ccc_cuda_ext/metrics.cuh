#pragma once

#include <vector>
#include <memory>
#include <pybind11/numpy.h>
#include <thrust/device_vector.h>

namespace py = pybind11;

template <typename T>
auto ari(const py::array_t<T, py::array::c_style>& parts,
         const size_t n_features,
         const size_t n_parts,
         const size_t n_objs) -> std::vector<float>;

// Used for internal c++ testing
template <typename T>
auto ari_core(const T* parts,
         const size_t n_features,
         const size_t n_parts,
         const size_t n_objs) -> std::vector<float>;

// Declaration of the device function
template <typename T, typename R>
auto ari_core_device(const T *parts,
                    const size_t n_features,
                    const size_t n_parts,
                    const size_t n_objs) -> std::unique_ptr<thrust::device_vector<R>>;
