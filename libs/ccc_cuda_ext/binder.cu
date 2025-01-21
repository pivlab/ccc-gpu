
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "metrics.cuh"
#include "coef.cuh"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(ccc_cuda_ext, m) {
    m.doc() = "CUDA extension module for CCC";
    m.def("ari_int32", &ari<int>, "CUDA version of Adjusted Rand Index (ARI) calculation",
        "parts"_a, "n_features"_a, "n_parts"_a, "n_objs"_a);    // _a is a shorter notation for named arguments
    m.def("example_return_optional_vectors", &example_return_optional_vectors,
          py::arg("include_first") = true,
          py::arg("include_second") = true,
          py::arg("include_third") = true,
          "Returns a tuple of three optional vectors");
}
