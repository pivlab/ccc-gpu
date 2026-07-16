#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <thrust/device_vector.h>

namespace py = pybind11;

/**
 * @brief Validity class of a partition pair, shared by the observed-statistic
 *        ARI kernels and the permutation (p-value) kernel.
 *
 * A partition's first element encodes special markers produced by get_parts:
 *   -1 -> categorical marker ("empty"/unused partition of a categorical
 *          feature): the pair contributes an ARI of 0.0.
 *   -2 -> singleton marker (a partition collapsed to a single cluster, usually
 *          constant input): the pair yields NaN, which poisons the whole
 *          feature comparison to NaN in the max reduction.
 * Anything else is a valid partition and the ARI is computed normally.
 */
enum class PartPairValidity : int
{
    VALID = 0,
    CATEGORICAL = 1, // -1 marker -> ARI 0.0
    SINGLETON = 2    // -2 marker -> NaN
};

/**
 * @brief Classify a partition pair from the first element of each partition.
 *
 * The categorical (-1) check takes precedence over the singleton (-2) check so
 * the behavior matches ari_kernel (which tests -1 before -2). Because get_parts
 * fills an entire partition row with the same marker, inspecting index 0 is
 * sufficient and permutation of object order does not change the class.
 */
template <typename T>
__device__ __host__ inline PartPairValidity classify_partition_pair(T a0, T b0)
{
    if (a0 == static_cast<T>(-1) || b0 == static_cast<T>(-1))
    {
        return PartPairValidity::CATEGORICAL;
    }
    if (a0 == static_cast<T>(-2) || b0 == static_cast<T>(-2))
    {
        return PartPairValidity::SINGLETON;
    }
    return PartPairValidity::VALID;
}

// Used for external python testing
template <typename T>
auto ari(const py::array_t<T, py::array::c_style> &parts,
         const size_t n_features,
         const size_t n_parts,
         const size_t n_objs,
         const uint64_t batch_start = 0,
         const uint64_t batch_size = 0) -> std::vector<float>;

// Used for internal c++ testing
template <typename T>
auto ari_core_host(const T *parts,
                   const size_t n_features,
                   const size_t n_parts,
                   const size_t n_objs,
                   const uint64_t batch_start = 0,
                   const uint64_t batch_size = 0) -> std::vector<float>;

// Used in the coef API
template <typename T, typename R>
auto ari_core_device(const py::array_t<T, py::array::c_style> &parts,
                     const uint64_t n_features,
                     const uint64_t n_parts,
                     const uint64_t n_objs,
                     const uint64_t batch_start = 0,
                     const uint64_t batch_size = 0) -> std::unique_ptr<thrust::device_vector<R>>;
