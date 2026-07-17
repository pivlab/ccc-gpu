#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T, typename R>
auto compute_coef(const py::array_t<T, py::array::c_style> &parts, const size_t n_features, const size_t n_parts,
                  const size_t n_objs, const bool return_parts, std::optional<uint32_t> pvalue_n_perms = std::nullopt)
    -> py::object;
