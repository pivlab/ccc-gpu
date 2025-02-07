Build the C++ CUDA extension module
===============================================

`Scikit-build <https://scikit-build-core.readthedocs.io/en/latest/getting_started.html>`_ is used to build the C++ CUDA extension module and its tests.

How to install this CUDA module
------------------------------------------------------

At the root of the repository, run:

.. code-block:: bash

    conda activate ccc-gpu
    # This will build the c++ module and install it with the Python package in the current environment
    pip install .


How to build the C++ CUDA extension module separately
-------------------------------------------------------

.. code-block:: bash

    # Clean up the build directory
    rm -rf build
    # Read ./CMakeLists.txt, configure the project, generate the build system files in the ./build directory
    cmake -S . -B build
    # Compile the project, generate the executable files in the ./build directory
    cmake --build build
