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
#include <tuple>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <spdlog/spdlog.h>

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
#define CUDA_CHECK(ans, abort)                        \
    {                                                 \
        gpu_assert((ans), __FILE__, __LINE__, abort); \
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
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        spdlog::error("CUDA Error: {} at {}:{}", cudaGetErrorString(code), file, line);
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
 *     check_cuda_device();  // Verify CUDA device availability
 *     // ... rest of the program
 * }
 * @endcode
 */
inline void check_cuda_device()
{
    int deviceCount;
    CUDA_CHECK_MANDATORY(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        spdlog::error("No CUDA devices available");
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
 * @return std::tuple<bool, std::string> containing validation status and any warning/error message
 *
 * @throws Exits if unable to query device properties
 *
 * Example:
 * @code
 * size_t needed_shared_mem = 1024 * sizeof(float);
 * auto [is_valid, message] = check_shared_memory_size(needed_shared_mem);
 * if (!is_valid) {
 *     throw std::runtime_error(message);
 * }
 * if (!message.empty()) {
 *     std::cerr << message << std::endl;  // Handle warning
 * }
 * @endcode
 */
inline std::tuple<bool, std::string> check_shared_memory_size(const size_t requested_size, int device_id = 0)
{
    cudaDeviceProp prop;
    CUDA_CHECK_MANDATORY(cudaGetDeviceProperties(&prop, device_id));

    const size_t max_shared_mem = prop.sharedMemPerBlock;

    if (requested_size > max_shared_mem)
    {
        return std::make_tuple(false,
                               std::string("Required shared memory (") + std::to_string(requested_size) +
                                   " bytes) exceeds device limit (" + std::to_string(max_shared_mem) + " bytes)");
    }

    // Optionally warn if close to limit (e.g., using more than 90%)
    if (requested_size > (max_shared_mem * 0.9))
    {
        return std::make_tuple(true,
                               std::string("Warning: Shared memory usage (") + std::to_string(requested_size) +
                                   " bytes) is close to device limit (" + std::to_string(max_shared_mem) + " bytes)");
    }

    return std::make_tuple(true, "");
}

/**
 * @brief Gets the maximum shared memory per block for the current device
 *
 * @param device_id The CUDA device ID to query (defaults to 0)
 * @return size_t Maximum shared memory per block in bytes
 *
 * @throws Exits if unable to query device properties
 */
inline size_t get_max_shared_memory_per_block(int device_id = 0)
{
    cudaDeviceProp prop;
    CUDA_CHECK_MANDATORY(cudaGetDeviceProperties(&prop, device_id));
    return prop.sharedMemPerBlock;
}

// get the total global memory size of the current device
inline size_t get_total_global_memory_size(int device_id = 0)
{
    cudaDeviceProp prop;
    CUDA_CHECK_MANDATORY(cudaGetDeviceProperties(&prop, device_id));
    return prop.totalGlobalMem;
}

/**
 * @brief Prints detailed information about the specified CUDA device
 *
 * This function prints comprehensive information about a CUDA device including:
 * - Device name
 * - CUDA driver and runtime versions
 * - Compute capability
 * - Memory information (global memory, clock rates, bus width)
 * - Current memory usage
 *
 * @param device_id The CUDA device ID to query (defaults to 0)
 *
 * Example usage:
 * @code
 * print_cuda_device_info();  // Print info for device 0
 * print_cuda_device_info(1); // Print info for device 1
 * @endcode
 */
inline void print_cuda_device_info(int device_id = 0)
{
    int driverVersion = 0, runtimeVersion = 0;
    CUDA_CHECK_MANDATORY(cudaSetDevice(device_id));
    cudaDeviceProp deviceProp;
    CUDA_CHECK_MANDATORY(cudaGetDeviceProperties(&deviceProp, device_id));
    spdlog::debug("Device {}: \"{}\"", device_id, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    spdlog::debug("  CUDA Driver Version / Runtime Version          {}.{} / {}.{}",
                 driverVersion / 1000, (driverVersion % 100) / 10,
                 runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    spdlog::debug("  CUDA Capability Major/Minor version number:    {}.{}",
                 deviceProp.major, deviceProp.minor);
    spdlog::debug("  Total amount of global memory:                 {:.2f} GBytes ({} bytes)",
                 (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
                 (unsigned long long)deviceProp.totalGlobalMem);
    spdlog::debug("  GPU Clock rate:                                {:.0f} MHz ({:.2f} GHz)",
                 deviceProp.clockRate * 1e-3f,
                 deviceProp.clockRate * 1e-6f);
    spdlog::debug("  Memory Clock rate:                             {:.0f} Mhz",
                 deviceProp.memoryClockRate * 1e-3f);
    spdlog::debug("  Memory Bus Width:                              {}-bit",
                 deviceProp.memoryBusWidth);
}

/**
 * @brief Prints current memory usage information for the specified CUDA device
 *
 * This function prints the current free and total memory available on the device.
 * It automatically chooses the most appropriate unit (KB, MB, or GB) based on the memory size.
 *
 * @param device_id The CUDA device ID to query (defaults to 0)
 * @return size_t The amount of free memory in bytes
 *
 * Example usage:
 * @code
 * print_cuda_memory_info();  // Print memory info for device 0
 * print_cuda_memory_info(1); // Print memory info for device 1
 * @endcode
 */
inline size_t print_cuda_memory_info(int device_id = 0)
{
    CUDA_CHECK_MANDATORY(cudaSetDevice(device_id));
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Helper function to format memory size with appropriate unit
    auto format_memory = [](size_t bytes) -> std::string
    {
        const size_t KB = 1024;
        const size_t MB = KB * 1024;
        const size_t GB = MB * 1024;

        if (bytes >= GB)
        {
            return std::to_string(static_cast<double>(bytes) / GB) + " GB";
        }
        else if (bytes >= MB)
        {
            return std::to_string(static_cast<double>(bytes) / MB) + " MB";
        }
        else if (bytes >= KB)
        {
            return std::to_string(static_cast<double>(bytes) / KB) + " KB";
        }
        else
        {
            return std::to_string(bytes) + " bytes";
        }
    };

    spdlog::debug("Free memory: {}, Total memory: {}",
                 format_memory(free_mem), format_memory(total_mem));
    return free_mem;
}

/**
 * @brief Prints current host memory usage information
 *
 * This function prints the current memory usage of the process.
 * It's useful for monitoring memory usage during program execution.
 *
 * Example usage:
 * @code
 * print_host_memory_info();  // Print memory info for the current process
 * @endcode
 */
inline size_t print_host_memory_info()
{
    // Get process memory usage
    FILE *file = fopen("/proc/self/status", "r");
    if (!file)
    {
        spdlog::error("Failed to open /proc/self/status");
        return 0;
    }

    size_t vm_size = 0;
    size_t vm_rss = 0;
    char line[128];

    while (fgets(line, 128, file) != NULL)
    {
        if (strncmp(line, "VmSize:", 7) == 0)
        {
            sscanf(line + 7, "%lu", &vm_size);
        }
        if (strncmp(line, "VmRSS:", 6) == 0)
        {
            sscanf(line + 6, "%lu", &vm_rss);
        }
    }
    fclose(file);

    // Convert to MB
    vm_size = vm_size / 1024;
    vm_rss = vm_rss / 1024;

    spdlog::debug("Host Memory Usage: {} MB (RSS), {} MB (Virtual)", vm_rss, vm_size);
    return vm_rss;
}

// TODO: Add more utility functions such as:
// - Memory management helpers
// - Device property queries
// - Stream management utilities
// - Kernel launch helpers
