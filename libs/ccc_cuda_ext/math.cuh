/**
 * @file math.cuh
 * @brief Internal athematical utility functions for array and matrix operations
 *
 * @note All functions in this header are marked with __device__ __host__
 *       qualifiers to enable their use in both CPU and GPU code.
 *
 * @warning Care should be taken when using these functions with large indices
 *          to avoid integer overflow.
 */

#pragma once

#include <cuda_runtime.h>

/**
 * @brief Unravel a flat (linear) index into corresponding 2D indices
 *
 * @details This function converts a single linear index into its corresponding row
 *          and column indices in a 2D array. This is particularly useful when working
 *          with flattened 2D arrays in CUDA kernels or host code.
 *
 * @param[in] flat_idx The flat index to unravel (must be non-negative)
 * @param[in] num_cols Number of columns in the 2D array (must be positive)
 * @param[out] row Reference to store the calculated row index
 * @param[out] col Reference to store the calculated column index
 *
 * @note The function assumes that the input flat_idx is valid for the given dimensions
 * @note This function can be called from both device (GPU) and host (CPU) code
 *
 * @example
 *     int row, col;
 *     unravel_index(5, 3, &row, &col);  // For a 3-column array, index 5 -> row=1, col=2
 */
__device__ __host__ inline void unravel_index(unsigned int flat_idx, unsigned int num_cols,
                                              unsigned int &row, unsigned int &col)
{
    // change int to uint32_t
    row = flat_idx / num_cols; // Compute row index
    col = flat_idx % num_cols; // Compute column index
}

/**
 * @brief Calculates coordinates in a symmetric matrix from a condensed flat index
 *
 * @details This function converts a linear index from a condensed array representing
 *          the upper triangular part of a symmetric matrix into its corresponding
 *          2D coordinates. The condensed array stores only unique elements, excluding
 *          the diagonal and redundant symmetric elements.
 *
 * @param[in] n_obj The size of one dimension of the square symmetric matrix (must be > 1)
 * @param[in] idx The flat index from the condensed array (must be valid for given n_obj)
 * @param[out] x Reference to store the calculated row coordinate
 * @param[out] y Reference to store the calculated column coordinate
 *
 * @note The function uses the quadratic formula to solve for coordinates
 * @note The resulting coordinates always satisfy x < y to ensure upper triangular access
 * @note This function can be called from both device (GPU) and host (CPU) code
 *
 * @example
 *     For a 4x4 symmetric matrix:
 *     [ - 0 1 2 ]    The condensed array would be [0,1,2,3,4,5]
 *     [ 0 - 3 4 ]    where each number represents the flat index
 *     [ 1 3 - 5 ]    For idx=3, the function returns x=1, y=2
 *     [ 2 4 5 - ]    representing the position in the original matrix
 */
__device__ __host__ inline void get_coords_from_index(unsigned int n_obj, unsigned int idx,
                                                      unsigned int &x, unsigned int &y)
{
    // Prevent overflow by using int64_t for intermediate calculations
    int64_t n_obj_64 = static_cast<int64_t>(n_obj);
    int64_t idx_64 = static_cast<int64_t>(idx);
    
    // Calculate 'b' based on the input n_obj
    int64_t b = 1 - 2 * n_obj_64;
    
    // Calculate discriminant using double precision to avoid overflow
    double discriminant = static_cast<double>(b * b) - 8.0 * static_cast<double>(idx_64);
    
    // Check for negative discriminant (invalid input)
    if (discriminant < 0) {
        x = 0;
        y = 0;
        return;
    }
    
    // Calculate 'x' using the quadratic formula part
    double x_float = (-static_cast<double>(b) - sqrt(discriminant)) / 2.0;
    int64_t x_64 = static_cast<int64_t>(floor(x_float));
    
    // Bounds checking for x
    if (x_64 < 0 || x_64 > UINT32_MAX) {
        x = 0;
        y = 0;
        return;
    }
    
    x = static_cast<unsigned int>(x_64);
    
    // Calculate 'y' based on 'x' and the index
    int64_t y_64 = idx_64 + x_64 * (b + x_64 + 2) / 2 + 1;
    
    // Bounds checking for y
    if (y_64 < 0 || y_64 > UINT32_MAX) {
        x = 0;
        y = 0;
        return;
    }
    
    y = static_cast<unsigned int>(y_64);
}
