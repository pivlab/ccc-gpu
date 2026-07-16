#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "metrics.cuh"
#include "coef.cuh"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(ccc_cuda_ext, m)
{
    m.doc() = "CUDA extension module for CCC";
    m.def("ari_int32", &ari<int>, "CUDA version of Adjusted Rand Index (ARI) calculation",
          "parts"_a, "n_features"_a, "n_parts"_a, "n_objs"_a, "batch_start"_a = 0, "batch_size"_a = 0); // _a is a shorter notation for named arguments
    m.def("compute_coef", &compute_coef<int16_t, float>,
          "CUDA version of CCC coefficient calculation",
          "parts"_a,      // numpy array of partitions
          "n_features"_a, // number of features
          "n_parts"_a,    // number of partitions per feature
          "n_objs"_a,     // number of objects
          // return_parts is accepted for API compatibility but is unused by the
          // extension: the partitions are assembled Python-side in ccc.coef.impl_gpu.ccc.
          "return_parts"_a = false,
          "pvalue_n_perms"_a = std::nullopt // optional number of permutations for p-value
    );
}
