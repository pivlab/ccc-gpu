cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CUDA CXX)

# Add this near the top of your file, after project()
# Define the include directories for the whole project
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libs)
# Set extention name and source directory
set(CUDA_EXT_MODULE_NAME ccc_cuda_ext)
set(CUDA_EXT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/${CUDA_EXT_MODULE_NAME})

file(GLOB headers ${CUDA_EXT_DIR}/*.cuh)
file(GLOB sources ${CUDA_EXT_DIR}/*.cu)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set Python Standard
# Get Python version dynamically
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    OUTPUT_VARIABLE PYTHON_VERSION_FULL
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Set the paths using the detected version
set(Python_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python${PYTHON_VERSION_FULL})
set(PYTHON_LIBRARY $ENV{CONDA_PREFIX}/lib/libpython${PYTHON_VERSION_FULL}.so)


# Set up pybind11
find_package(Python REQUIRED Development)
find_package(pybind11 CONFIG REQUIRED)
set(PYBIND11_NEWPYTHON ON)

# Download and configure Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2  # Adjust version as needed
)
FetchContent_MakeAvailable(googletest)
# Download and configure spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
)
FetchContent_MakeAvailable(spdlog)

# include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Setup Gtest
enable_testing()
# Function to automatically add tests from a directory
function(add_tests_from_directory TEST_DIR)
    # Find all test files in the directory
    file(GLOB_RECURSE TEST_FILES
        "${TEST_DIR}/*_test.cu"    # Files ending with _test.cu
        "${TEST_DIR}/test_*.cu"    # Files starting with test_
    )

    # Loop through each test file
    foreach(TEST_FILE ${TEST_FILES})
        # Get the filename without extension
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

        # Create an executable for this test
        add_executable(${TEST_NAME} ${TEST_FILE} ${headers} ${sources})

        # target_include_directories(${TEST_NAME} PRIVATE
        #     ${PROJECT_INCLUDE_DIR}  # Add this line
        #     ${Python_INCLUDE_DIRS}
        # )

        target_link_libraries(${TEST_NAME} PRIVATE
            GTest::gtest_main
            GTest::gtest
            pybind11::headers
            pybind11::embed
            Python::Python
            # Add your other project libraries here
            # project_lib
        )

        # Add the test to CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

        # Set test properties (optional)
        # Set test properties (optional)
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 10  # Timeout in seconds
            WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )
    endforeach()
endfunction()

# Specify your test directory and call the function
# add_tests_from_directory(${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Optional: Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# Set up binding then do compilation and installation
pybind11_add_module(${CUDA_EXT_MODULE_NAME}
    ${CUDA_EXT_DIR}/binder.cu
    ${CUDA_EXT_DIR}/metrics.cu
    ${CUDA_EXT_DIR}/coef.cu
)

target_link_libraries(${CUDA_EXT_MODULE_NAME} PRIVATE
    spdlog::spdlog
)

install(TARGETS ${CUDA_EXT_MODULE_NAME} LIBRARY DESTINATION .)
