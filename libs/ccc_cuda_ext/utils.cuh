/**
 * @file utils.cuh
 * @brief CUDA utility functions and macros for error checking and device management
 *
 * This header provides common CUDA utilities including:
 * - Error checking macros and functions
 * - Device verification
 * - Common CUDA helper functions
 *
 * Usage example:
 * @code
 * float* d_data;
 * CUDA_CHECK_MANDATORY(cudaMalloc(&d_data, size * sizeof(float)));
 * // ... use d_data ...
 * CUDA_CHECK_OPTIONAL(cudaFree(d_data));
 * @endcode
 */

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief Main error checking macro for CUDA operations
 *
 * @param ans The CUDA operation to check (e.g., cudaMalloc, cudaMemcpy)
 * @param abort If true, program will exit on error; if false, will only print error
 *
 * Example:
 * @code
 * CUDA_CHECK(cudaMalloc(&ptr, size), true);  // Will exit on failure
 * CUDA_CHECK(cudaDeviceSynchronize(), false); // Will only print error on failure
 * @endcode
 */
#define CUDA_CHECK(ans, abort)                       \
    {                                                \
        gpuAssert((ans), __FILE__, __LINE__, abort); \
    }

/**
 * @brief Mandatory error checking macro that always aborts on failure
 * Use for critical operations where failure is not acceptable
 *
 * @param ans The CUDA operation to check
 */
#define CUDA_CHECK_MANDATORY(ans) CUDA_CHECK(ans, true)

/**
 * @brief Optional error checking macro that only prints errors
 * Use for non-critical operations where failure can be handled
 *
 * @param ans The CUDA operation to check
 */
#define CUDA_CHECK_OPTIONAL(ans) CUDA_CHECK(ans, false)

/**
 * @brief Internal function for CUDA error checking
 *
 * This function is called by the CUDA_CHECK macros to handle error checking.
 * It prints detailed error information and optionally terminates the program.
 *
 * @param code The CUDA error code to check
 * @param file Source file where the error occurred
 * @param line Line number where the error occurred
 * @param abort Whether to terminate the program on error
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

/**
 * @brief Verifies that a CUDA-capable device is available
 *
 * This function checks if any CUDA devices are present and accessible.
 * It should be called at the start of any CUDA-dependent program.
 *
 * @throws Exits with error code -1 if no CUDA devices are found
 *
 * Example:
 * @code
 * int main() {
 *     checkCudaDevice();  // Verify CUDA device availability
 *     // ... rest of the program
 * }
 * @endcode
 */
inline void checkCudaDevice()
{
    int deviceCount;
    CUDA_CHECK_MANDATORY(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA devices available\n");
        exit(-1);
    }
}

/**
 * @brief Checks if requested shared memory size is within device limits
 *
 * Verifies that the requested shared memory size doesn't exceed the maximum
 * available shared memory per block for the current CUDA device.
 *
 * @param requested_size The amount of shared memory needed (in bytes)
 * @param device_id The CUDA device ID to check (defaults to 0)
 * @return true if the requested size is within limits, false otherwise
 *
 * @throws Exits if unable to query device properties
 *
 * Example:
 * @code
 * size_t needed_shared_mem = 1024 * sizeof(float);
 * if (!checkSharedMemorySize(needed_shared_mem)) {
 *     fprintf(stderr, "Insufficient shared memory available\n");
 *     return false;
 * }
 * @endcode
 */
inline bool checkSharedMemorySize(const size_t requested_size, int device_id = 0)
{
    cudaDeviceProp prop;
    CUDA_CHECK_MANDATORY(cudaGetDeviceProperties(&prop, device_id));

    const size_t max_shared_mem = prop.sharedMemPerBlock;

    if (requested_size > max_shared_mem)
    {
        fprintf(stderr, "Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n",
                requested_size, max_shared_mem);
        return false;
    }

    // Optionally warn if close to limit (e.g., using more than 90%)
    if (requested_size > (max_shared_mem * 0.9))
    {
        fprintf(stderr, "Warning: Shared memory usage (%zu bytes) is close to device limit (%zu bytes)\n",
                requested_size, max_shared_mem);
    }

    return true;
}

/**
 * @brief Gets the maximum shared memory per block for the current device
 *
 * @param device_id The CUDA device ID to query (defaults to 0)
 * @return size_t Maximum shared memory per block in bytes
 *
 * @throws Exits if unable to query device properties
 */
inline size_t getMaxSharedMemoryPerBlock(int device_id = 0)
{
    cudaDeviceProp prop;
    CUDA_CHECK_MANDATORY(cudaGetDeviceProperties(&prop, device_id));
    return prop.sharedMemPerBlock;
}

// TODO: Add more utility functions such as:
// - Memory management helpers
// - Device property queries
// - Stream management utilities
// - Kernel launch helpers
