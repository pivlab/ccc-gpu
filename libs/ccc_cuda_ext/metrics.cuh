#pragma once

#include <vector>
#include <memory>
#include <pybind11/numpy.h>
#include <thrust/device_vector.h>

namespace py = pybind11;

// Used for external python testing
template <typename T>
auto ari(const py::array_t<T, py::array::c_style> &parts,
         const size_t n_features,
         const size_t n_parts,
         const size_t n_objs) -> std::vector<float>;

// Used for internal c++ testing
template <typename T>
auto ari_core_host(const T *parts,
                   const size_t n_features,
                   const size_t n_parts,
                   const size_t n_objs) -> std::vector<float>;

// Used in the coef API
template <typename T, typename R>
auto ari_core_device(const py::array_t<T, py::array::c_style> &parts,
                     const size_t n_features,
                     const size_t n_parts,
                     const size_t n_objs) -> std::unique_ptr<thrust::device_vector<R>>;

template <typename T, typename R>
auto ari_core_scalar(T *const d_part0,
                     T *const d_part1,
                     const size_t n_objs,
                     const size_t max_k,
                     const size_t stream_idx,
                     const cudaStream_t stream,
                     R *const h_aris) -> void;
