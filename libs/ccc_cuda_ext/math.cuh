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
 * @brief Unravel a flat index to the corresponding 2D indicis
 * @param[in] flat_idx The flat index to unravel
 * @param[in] num_cols Number of columns in the 2D array
 * @param[out] row Pointer to the row index
 * @param[out] col Pointer to the column index
 */
__device__ __host__ inline void unravel_index(uint32_t flat_idx, uint32_t num_cols, uint32_t *row, uint32_t *col)
{
    // change int to uint32_t
    *row = flat_idx / num_cols; // Compute row index
    *col = flat_idx % num_cols; // Compute column index
}

/**
 * @brief Given the number of objects and an index, this function calculates
 *        the coordinates in a symmetric matrix from a flat index.
 *        For example, if there are n_obj objects (such as genes), a condensed
 *        1D array can be created with pairwise comparisons between these
 *        objects, which corresponds to a symmetric 2D matrix. This function
 *        calculates the 2D coordinates (x, y) in the symmetric matrix that
 *        corresponds to the given flat index.
 *
 * @param[in] n_obj The total number of objects (i.e., the size of one dimension
 *                  of the square symmetric matrix).
 * @param[in] idx The flat index from the condensed pairwise array.
 * @param[out] x Pointer to the calculated row coordinate in the symmetric matrix.
 * @param[out] y Pointer to the calculated column coordinate in the symmetric matrix.
 */
__device__ __host__ inline void get_coords_from_index(int n_obj, int idx, uint32_t *x, uint32_t *y)
{
    // Use int64_t to prevent overflow in intermediate calculations
    int64_t n_obj_64 = static_cast<int64_t>(n_obj);
    int64_t idx_64 = static_cast<int64_t>(idx);

    // Calculate 'b' using 64-bit arithmetic
    int64_t b = 1 - 2 * n_obj_64;

    // Calculate discriminant using 64-bit arithmetic
    // Use double for floating point to maintain precision
    double b_squared = static_cast<double>(b) * b;
    double idx_term = 8.0 * static_cast<double>(idx_64);
    double discriminant = b_squared - idx_term;

    // Calculate x using double precision
    double x_float = (-b - sqrt(discriminant)) / 2.0;

    // Floor and convert to uint32_t, with bounds checking
    int64_t x_64 = static_cast<int64_t>(floor(x_float));
    if (x_64 < 0 || x_64 > UINT32_MAX)
    {
        // Handle error condition - could throw error or set to max/min value
        *x = 0;
        *y = 0;
        return;
    }
    *x = static_cast<uint32_t>(x_64);

    // Calculate y using 64-bit arithmetic to prevent overflow
    int64_t y_term1 = idx_64;
    int64_t y_term2 = x_64 * (b + x_64 + 2) / 2;
    int64_t y_64 = y_term1 + y_term2 + 1;

    // Bounds checking for y
    if (y_64 < 0 || y_64 > UINT32_MAX)
    {
        // Handle error condition
        *x = 0;
        *y = 0;
        return;
    }
    *y = static_cast<uint32_t>(y_64);
}
