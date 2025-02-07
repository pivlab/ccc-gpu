Testing
===============

All the tests are located in the `tests` folder.

Folder `tests/gpu` contains python tests for the GPU-related functionalities. They invokes Python APIs for corresponding CUDA C++ APIs and compares the results against those from the CPU-based implementation.

Folder `tests/gpu/exclueded` contains deprecated python tests for the GPU-related functionalities. They are excluded from all test suites, and are kept for future refactoring.

Folder `tests/cuda_ext` contains unit tests for the CUDA extension module written in C++. They contain redundant tests in regard to the Python tests. The reason for keeping them is to test the binding creation logic and other features provided by pybind11.

There's one testing script for running specific tests:

.. code-block:: bash

    bash ./scripts/run_tests.sh [python | cpp | all]

Make sure all the tests pass before pushing the changes. A git pre-push hook will be added to force executing the tests before pushing.
