#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

auto example_return_optional_vectors(
    bool include_first = true,
    bool include_second = true,
    bool include_third = true
) -> py::object;
