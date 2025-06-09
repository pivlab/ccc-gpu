#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

auto example_return_optional_vectors(
    bool include_first = true,
    bool include_second = true,
    bool include_third = true
) -> py::object;


template <typename T, typename R>
auto compute_coef(const py::array_t<T, py::array::c_style> &parts,
                 const size_t n_features,
                 const size_t n_parts,
                 const size_t n_objs,
                 const bool return_parts,
                 std::optional<uint32_t> pvalue_n_perms = std::nullopt) -> py::object;
