cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CUDA CXX)

# When ON, build the CUDA C++ gtests under tests/cuda_ext/ and register them with
# ctest. OFF by default so the wheel build never fetches googletest or compiles
# the native tests. Enable with: cmake -S . -B build -DCCC_BUILD_TESTS=ON
option(CCC_BUILD_TESTS "Build the CUDA C++ gtests and register them with ctest" OFF)

# Export compile_commands.json so clang-tidy (and editors/LSP) can analyze the
# CUDA/C++ sources. Run clang-tidy host-only, e.g.:
#   clang-tidy -p build --extra-arg=--cuda-host-only libs/ccc_cuda_ext/*.cu
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# Note: Conda-specific paths removed for manylinux compatibility
# CMake's FindPython will automatically detect Python in manylinux containers
# Uncomment below if building locally with conda:
# execute_process(
#     COMMAND "${Python_EXECUTABLE}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
#     OUTPUT_VARIABLE PYTHON_VERSION_FULL
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# set(Python_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
# set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python${PYTHON_VERSION_FULL})
# set(PYTHON_LIBRARY $ENV{CONDA_PREFIX}/lib/libpython${PYTHON_VERSION_FULL}.so)


# Set up pybind11
# Use Development.Module for manylinux compatibility (not Development.Embed)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
set(PYBIND11_NEWPYTHON ON)

# Google Test is fetched only when building the native tests (CCC_BUILD_TESTS);
# the default wheel build must not touch the network for gtest. See the guarded
# test block near the end of this file.
include(FetchContent)
# Download and configure spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
)
# Build spdlog with -fPIC for compatibility with shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(spdlog)

# include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

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

# --------------------------------------------------------------------------- #
# CUDA C++ gtests (opt-in). Guarded by CCC_BUILD_TESTS so the wheel build never
# fetches googletest or compiles the native tests.
#   cmake -S . -B build -DCCC_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
# --------------------------------------------------------------------------- #
if(CCC_BUILD_TESTS)
    message(STATUS "CCC_BUILD_TESTS=ON: configuring CUDA C++ gtests")

    # Embedding a Python interpreter (test_ari_random.cu) needs Development.Embed,
    # which the default wheel build (Development.Module only) does not require.
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed Development.Module)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    enable_testing()

    # The gtests exercise the ARI kernels in metrics.cu. Compile that translation
    # unit once into a static library and link it into each test executable so we
    # do not recompile the (slow) .cu file per test. coef.cu / binder.cu are not
    # needed by the current tests and are intentionally excluded.
    add_library(ccc_cuda_ext_testlib STATIC ${CUDA_EXT_DIR}/metrics.cu)
    target_link_libraries(ccc_cuda_ext_testlib PUBLIC
        spdlog::spdlog
        pybind11::headers
        Python::Python
    )
    set_target_properties(ccc_cuda_ext_testlib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    # Discover the gtest source files. test_ari_py.cu is a legacy manual driver
    # with its own main() (not a gtest); it would clash with gtest_main, so skip it.
    file(GLOB CCC_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_ext/test_*.cu")
    foreach(TEST_FILE ${CCC_TEST_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        if(TEST_NAME STREQUAL "test_ari_py")
            continue()
        endif()

        add_executable(${TEST_NAME} ${TEST_FILE})
        target_link_libraries(${TEST_NAME} PRIVATE
            ccc_cuda_ext_testlib
            GTest::gtest_main
            GTest::gtest
            pybind11::headers
            pybind11::embed
            Python::Python
        )

        # gtest_discover_tests registers each TEST_P case with ctest. The default
        # 10s ctest timeout was too tight for the Python-reference tests; raise it.
        gtest_discover_tests(${TEST_NAME}
            PROPERTIES
                TIMEOUT 300
                WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            DISCOVERY_TIMEOUT 120
        )
    endforeach()
endif()
