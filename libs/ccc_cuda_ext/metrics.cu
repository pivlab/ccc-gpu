#include <cuda_runtime.h>
#include <cub/block/block_load.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <iostream>
#include <cmath>
#include <assert.h>
#include "metrics.cuh"
#include "utils.cuh"

namespace py = pybind11;

/**
 * Future optimizations
 * 1. use narrower data types
 * 2. optimized on locality
 * 3. use warp-level reduction
 */

/**
 * Future optimizations
 * 1. GPU memory is not enough to store the partitions -> split the partitions into smaller chunks
 *    and do stream processing
 * 2.
 */

/**
 * @brief Unravel a flat index to the corresponding 2D indicis
 * @param[in] flat_idx The flat index to unravel
 * @param[in] num_cols Number of columns in the 2D array
 * @param[out] row Pointer to the row index
 * @param[out] col Pointer to the column index
 */
__device__ __host__ inline void unravel_index(int flat_idx, int num_cols, int *row, int *col)
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

/**
 * @brief Compute the contingency matrix for two partitions using shared memory
 * @param[in] part0 Pointer to the first partition array, global memory
 * @param[in] part1 Pointer to the second partition array, global memory
 * @param[in] n_objs Number of elements in each partition array
 * @param[out] shared_cont_mat Pointer to shared memory for storing the contingency matrix
 * @param[in] k Maximum number of clusters (size of contingency matrix is k x k)
 */
__device__ void get_contingency_matrix(int *part0, int *part1, int n_objs, int *shared_cont_mat, int k)
{
    const int tid = threadIdx.x;
    const int n_block_threads = blockDim.x;
    const int cont_mat_size = k * k;

    // Initialize shared memory
    if (tid < cont_mat_size)
    {
        shared_cont_mat[tid] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = tid; i < n_objs; i += n_block_threads)
    {
        // Directly load row/col info from global memory into registers, no need to load into shared memory
        const int row = part0[i];
        const int col = part1[i];

        // Add bounds checking
        // assert(row >= 0 && row < k && col >= 0 && col < k);
        // OPT: can we use shared memory to avoid atomicAdd?
        atomicAdd(&shared_cont_mat[row * k + col], 1);
    }
    // __syncthreads();
}

/**
 * @brief CUDA device function to compute the pair confusion matrix
 * @param[in] contingency Pointer to the contingency matrix
 * @param[in] sum_rows Pointer to the sum of rows in the contingency matrix
 * @param[in] sum_cols Pointer to the sum of columns in the contingency matrix
 * @param[in] n_objs Number of objects in each partition
 * @param[in] k Number of clusters (assuming k is the max of clusters in part0 and part1)
 * @param[out] C Pointer to the output pair confusion matrix (2x2)
 */
__device__ void get_pair_confusion_matrix(
    const int *__restrict__ contingency,
    int *sum_rows,
    int *sum_cols,
    const int n_objs,
    const int k,
    int *C)
{
    // TODO: use block-level reduction

    const int tid = threadIdx.x;
    const int n_block_threads = blockDim.x;

    // Initialize sum_rows and sum_cols
    for (int i = tid; i < k; i += n_block_threads)
    {
        sum_rows[i] = 0;
        sum_cols[i] = 0;
    }
    __syncthreads();

    // Compute sum_rows and sum_cols
    for (int i = tid; i < k * k; i += n_block_threads)
    {
        int row = i / k;
        int col = i % k;
        int val = contingency[i];
        atomicAdd(&sum_cols[col], val);
        atomicAdd(&sum_rows[row], val);
    }
    __syncthreads();

    // Compute sum_squares
    int sum_squares;
    if (tid == 0)
    {
        sum_squares = 0;
        for (int i = 0; i < k * k; ++i)
        {
            sum_squares += (contingency[i] * contingency[i]);
        }
    }
    __syncthreads();

    // Use different warps to compute C[1,1], C[0,1], C[1,0], and C[0,0]
    if (tid == 0)
    {
        C[3] = sum_squares - n_objs; // C[1,1]

        int temp = 0;
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                temp += (contingency[i * k + j]) * sum_cols[j];
            }
        }
        C[1] = temp - sum_squares; // C[0,1]

        temp = 0;
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                temp += (contingency[j * k + i]) * sum_rows[j];
            }
        }
        C[2] = temp - sum_squares; // C[1,0]

        C[0] = n_objs * n_objs - C[1] - C[2] - sum_squares; // C[0,0]
    }
}

/**
 * @brief Main ARI kernel. Now only compare a pair of ARIs
 * @param parts Device pointer to the 3D Array of partitions with shape of (n_features, n_parts, n_objs)
 * @param n_aris Number of ARIs to compute
 * @param n_features Number of features
 * @param n_parts Number of partitions of each feature
 * @param n_objs Number of objects in each partitions
 * @param n_elems_per_feat Number of elements for each feature, i.e., part[i].x * part[i].y
 * @param n_part_mat_elems Number of elements in the square partition matrix
 * @param k The max value of cluster number + 1
 * @param out Output array of ARIs
 */
// TODO: Parameterize the int type to allow using narrower int types for memory efficiency
extern "C" __global__ void ari_kernel(int *parts,
                                      const int n_aris,
                                      const int n_features,
                                      const int n_parts,
                                      const int n_objs,
                                      const int n_elems_per_feat,
                                      const int n_part_mat_elems,
                                      const int k,
                                      float *out)
{
    /*
     * Step 0: Compute shared memory addresses
     */
    extern __shared__ int shared_mem[];
    int *s_contingency = shared_mem;               // k * k elements
    int *s_sum_rows = s_contingency + (k * k);     // k elements
    int *s_sum_cols = s_sum_rows + k;              // k elements
    int *s_pair_confusion_matrix = s_sum_cols + k; // 4 elements

    /*
     * Step 1: Each thead, unravel flat indices and load the corresponding data into shared memory
     */
    // each block is responsible for one ARI computation
    int ari_block_idx = blockIdx.x;
    // obtain the corresponding parts and unique counts
    int feature_comp_flat_idx = ari_block_idx / n_part_mat_elems; // flat comparison pair index for two features
    int part_pair_flat_idx = ari_block_idx % n_part_mat_elems;    // flat comparison pair index for two partitions of one feature pair
    uint32_t i, j;

    // Unravel the feature indices
    // For example, if n_features = 3, n_feature_comp = n_features * (n_features - 1) / 2 = 3
    // The feature indices of the pair being compared are (0, 1), (0, 2), (1, 2)
    // i.e., the pairs being compared are feature0-feature1, feature0-feature2, feature1-feature2
    // The range of the flattened index is [0, n_feature_comp - 1] = [0, 2]
    // Given the flat index, we compute the corresponding feature indices
    get_coords_from_index(n_features, feature_comp_flat_idx, &i, &j);
    // assert(i < n_features && j < n_features);
    // assert(i >= 0 && j >= 0);

    // Unravel the partition indices within the feature pair
    // For example, if n_parts = 3, n_part_mat_elems = n_parts * n_parts = 9
    // The partition indices of the pair being compared are (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
    // i.e., the pairs being compared are part0-part1, part0-part2, part1-part0, part1-part1, part1-part2, part2-part0, part2-part1, part2-part2
    // The range of the flattened index is [0, n_part_mat_elems - 1] = [0, 8]
    // Given the flat index, we compute the corresponding partition indices
    int m, n;
    unravel_index(part_pair_flat_idx, n_parts, &m, &n);
    // Make pointers to select the partitions from `parts` and unique counts for the feature pair
    // Todo: Use int4*?
    // Prefix `t_` for data hold by a thread
    int *t_data_part0 = parts + i * n_elems_per_feat + m * n_objs;
    int *t_data_part1 = parts + j * n_elems_per_feat + n * n_objs;

    // Check on categorical partition marker, if the first object of either partition is -1 (actually all the objects are -1),
    // then skip the computation for this feature pair. The final coef output will still have a slot for this pair, with a default value of -1.
    if (t_data_part0[0] == -1 || t_data_part1[0] == -1)
    {
        return;
    }

    /*
     * Step 2: Compute contingency matrix within the block
     */
    // shared mem address for the contingency matrix
    // int *s_contingency = shared_mem + 2 * n_objs;
    get_contingency_matrix(t_data_part0, t_data_part1, n_objs, s_contingency, k);

    /*
     * Step 3: Construct pair confusion matrix
     */
    get_pair_confusion_matrix(s_contingency, s_sum_rows, s_sum_cols, n_objs, k, s_pair_confusion_matrix);

    /*
     * Step 4: Compute ARI and write to global memory
     */
    if (threadIdx.x == 0)
    {
        float tn = s_pair_confusion_matrix[0];
        float fp = s_pair_confusion_matrix[1];
        float fn = s_pair_confusion_matrix[2];
        float tp = s_pair_confusion_matrix[3];
        float ari = 0.0f;
        if (fn == 0 && fp == 0)
        {
            ari = 1.0f;
        }
        else
        {
            ari = 2.0f * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
        }
        out[ari_block_idx] = ari;
    }
    __syncthreads();
}

/**
 * @brief Helper function to process and validate input numpy array
 * @param parts Input numpy array to process
 * @return Pointer to the underlying data
 */
template <typename T>
T *process_input_array(const py::array_t<T, py::array::c_style> &parts)
{
    py::buffer_info buffer = parts.request();
    if (buffer.format != py::format_descriptor<T>::format())
    {
        throw std::runtime_error("Incompatible format: expected an int array!");
    }
    if (buffer.ndim != 3)
    {
        throw std::runtime_error("Incompatible buffer dimension!");
    }
    return static_cast<T *>(buffer.ptr);
}

/**
 * @brief Internal lower-level ARI computation, returns a pointer to the ARI values on the device
 * @param parts pointer to the 3D Array of partitions with shape of (n_features, n_parts, n_objs)
 * @throws std::invalid_argument if "parts" is invalid
 * @return std::unique_ptr to thrust device vector containing ARI values with type R
 */
template <typename T, typename R>
auto ari_core_device(const T *parts,
                     const size_t n_features,
                     const size_t n_parts,
                     const size_t n_objs) -> std::unique_ptr<thrust::device_vector<R>>
{
    /*
     * Show debugging and device information
     */
    // printf("Max shared memory per block: %zu bytes\n", get_max_shared_memory_per_block());

    // Input validation
    if (!parts || n_features == 0 || n_parts == 0 || n_objs == 0)
    {
        throw std::invalid_argument("Invalid input parameters");
    }

    /*
     * Pre-computation
     */
    using parts_dtype = T;
    using out_dtype = R;
    const auto n_feature_comp = n_features * (n_features - 1) / 2;
    const auto n_aris = n_feature_comp * n_parts * n_parts;

    /*
     * Memory Allocation
     */
    // Create device vectors using unique_ptr
    const auto n_elems = n_features * n_parts * n_objs;
    auto d_parts = std::make_unique<thrust::device_vector<parts_dtype>>(parts, parts + n_elems);
    auto d_out = std::make_unique<thrust::device_vector<out_dtype>>(n_aris, 0.0f);

    // Define shared memory size for each block
    // Pre-compute the max value of the partitions
    // TODO: Each block could compute the max value of the partition pairs to eliminate this global reduction
    // Also, potentially we can spare some smem for better occupancy. But the issue is we need to dynamically allocate smem for each block
    const auto k = thrust::reduce(d_parts->begin(), d_parts->end(), -1, thrust::maximum<parts_dtype>()) + 1;
    const auto sz_parts_dtype = sizeof(parts_dtype);
    // Compute shared memory size
    // FIXME: Partition pair size should be fixed. Stream processing should be used for large input
    // NOTE: Use global memory to fix the issue for now and then optimize with shared memory
    // auto s_mem_size = 2 * n_objs * sz_parts_dtype;  // For the partition pair to be compared
    auto s_mem_size = 0;
    s_mem_size += k * k * sz_parts_dtype; // For contingency matrix
    s_mem_size += 2 * k * sz_parts_dtype; // For the internal sum arrays, FIXME: should be fixed?
    s_mem_size += 4 * sz_parts_dtype;     // For the pair confusion matrix

    // Check if shared memory size exceeds device limits
    auto [is_valid, message] = check_shared_memory_size(s_mem_size);
    if (!is_valid)
    {
        throw std::runtime_error(message);
    }

    /*
     * Launch the kernel
     */
    // Each logical block is responsible for one ARI computation
    const auto grid_size = n_aris;
    // Todo: change block_size to template parameter for performance tuning
    // For now, with 128 threads per block, we have 4 warps to compute the 4 elements of the confusion matrix.
    // Future optimizations should consider how to reduce the number of idle warps.
    const auto block_size = 128;
    // Launch the kernel
    ari_kernel<<<grid_size, block_size, s_mem_size>>>(
        thrust::raw_pointer_cast(d_parts->data()),
        n_aris,
        n_features,
        n_parts,
        n_objs,
        n_parts * n_objs,
        n_parts * n_parts,
        k,
        thrust::raw_pointer_cast(d_out->data()));
    // Return the device vector
    return d_out;
}

/**
 * @brief Overloaded ari_core_device function. Takes a numpy.ndarray as input
 * @param parts 3D Numpy.NDArray of partitions with shape of (n_features, n_parts, n_objs)
 * @throws std::invalid_argument if "parts" is invalid
 * @return std::unique_ptr to thrust device vector containing ARI values
 */
template <typename T, typename R>
auto ari_core_device(const py::array_t<int, py::array::c_style> &parts,
                     const size_t n_features,
                     const size_t n_parts,
                     const size_t n_objs) -> std::unique_ptr<thrust::device_vector<R>>
{
    const auto parts_ptr = process_input_array(parts);
    return ari_core_device<T, R>(parts_ptr, n_features, n_parts, n_objs);
}

/**
 * @brief Internal lower-level ARI computation
 * @param parts pointer to the 3D Array of partitions with shape of (n_features, n_parts, n_objs)
 * @throws std::invalid_argument if "parts" is invalid
 * @return std::vector<float> ARI values for each pair of partitions stored in host memory
 */
template <typename T>
auto ari_core_host(const T *parts,
                   const size_t n_features,
                   const size_t n_parts,
                   const size_t n_objs) -> std::vector<float>
{
    /*
     * Pre-computation
     */
    using parts_dtype = T;
    using out_dtype = float;
    const auto n_feature_comp = n_features * (n_features - 1) / 2;
    const auto n_aris = n_feature_comp * n_parts * n_parts;

    /*
     * Memory Allocation
     */
    // Allocate host memory
    thrust::host_vector<out_dtype> h_out(n_aris);
    // thrust::host_vector<parts_dtype> h_parts_pairs(n_aris * 2 * n_objs);

    // Call the device function ari_core_device
    auto d_out = ari_core_device<parts_dtype, out_dtype>(parts, n_features, n_parts, n_objs);

    // Copy data back to host using -> operator since d_out is a unique_ptr
    thrust::copy(d_out->begin(), d_out->end(), h_out.begin());

    // Copy data to std::vector
    std::vector<out_dtype> res(n_aris);
    thrust::copy(h_out.begin(), h_out.end(), res.begin());

    // Return the ARI values
    return res;
}

/**********************
  API Implementations
 **********************/

/**
 * @brief API exposed to Python for computing ARI using CUDA upon a 3D Numpy NDArray of partitions
 * @param parts 3D Numpy.NDArray of partitions with shape of (n_features, n_parts, n_objs)
 * @throws std::invalid_argument if "parts" is invalid
 * @return std::vector<float> All ARI values for each pair of partitions
 */
template <typename T>
auto ari(const py::array_t<T, py::array::c_style> &parts,
         const size_t n_features,
         const size_t n_parts,
         const size_t n_objs) -> std::vector<float>
{
    const auto parts_ptr = process_input_array(parts);
    return ari_core_host(parts_ptr, n_features, n_parts, n_objs);
}

/**
 * @brief API exposed to Python for computing ARI using CUDA upon a 3D Numpy NDArray of partitions
 * @param parts 3D Numpy.NDArray of partitions with shape of (n_features, n_parts, n_objs)
 * @throws std::invalid_argument if "parts" is invalid
 * @return std::vector<float> Reduced(max) ARI value for each pair of partitions
 */
template <typename T>
auto ari_reduced(const py::array_t<T, py::array::c_style> &parts,
                 const size_t n_features,
                 const size_t n_parts,
                 const size_t n_objs) -> std::vector<float>
{
    const auto parts_ptr = process_input_array(parts);
    throw std::logic_error("Function not yet implemented");
}

// Below is the explicit instantiation of the ari template function.
//
// Generally people would write the implementation of template classes and functions in the header file. However, we
// separate the implementation into a .cpp file to make things clearer. In order to make the compiler know the
// implementation of the template functions, we need to explicitly instantiate them here, so that they can be picked up
// by the linker.

// Used for external python testing
template auto ari<int>(
    const py::array_t<int, py::array::c_style> &parts,
    const size_t n_features,
    const size_t n_parts,
    const size_t n_objs) -> std::vector<float>;

// Used for internal c++ testing
template auto ari_core_host<int>(
    const int *parts,
    const size_t n_features,
    const size_t n_parts,
    const size_t n_objs) -> std::vector<float>;

// Used in the coef API
template auto ari_core_device<int, float>(
    const py::array_t<int, py::array::c_style> &parts,
    const size_t n_features,
    const size_t n_parts,
    const size_t n_objs) -> std::unique_ptr<thrust::device_vector<float>>;
