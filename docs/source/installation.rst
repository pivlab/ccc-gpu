Installation
============

Quick Install with pip
----------------------

The ``cccgpu`` package is now available for installation via pip from test PyPI:

However, note that cccgpu depends on `libstdc++`. For a smooth installation, we recommend using a wrapper conda environment to install it:

.. code-block:: bash

    mamba create -n ccc-gpu-toolchain-env -c conda-forge python=3.10 pip pytest libstdcxx-ng && conda activate ccc-gpu-toolchain-env

Support for more Python versions and architectures requires extra effort, and will be added soon.

Then, install the package in the toolchain environment:

.. code-block:: bash

    pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                --only-binary=cccgpu cccgpu

Then try running some tests to verify the installation:

.. code-block:: bash

    python -c "from ccc.coef.impl_gpu import ccc as ccc_gpu; import numpy as np; print(ccc_gpu(np.random.rand(100), np.random.rand(100)))"


**Command options explained:**

- ``--index-url https://test.pypi.org/simple/``: Specifies test PyPI as the primary package index to search for ``cccgpu``
- ``--extra-index-url https://pypi.org/simple/``: Adds the main PyPI repository as a fallback to install dependencies (numpy, scipy, numba, etc.) that may not be available on test PyPI
- ``--only-binary=cccgpu``: Ensures that only binary wheels are installed for ``cccgpu`` package, so you don't need to compile it from source
- ``cccgpu``: The package name to install

.. note::
   This installs from test PyPI while the package is in testing phase. Once stable, it will be available from the main PyPI repository with a simple ``pip install cccgpu`` command.


Install from Source
-------------------

For now, install from source using the provided conda-lock environment:

1. Install Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

First, install Mamba (recommended) and conda-lock:

.. code-block:: bash

    # Install MiniForge (includes Mamba)
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b

    # Install conda-lock
    pip install conda-lock

2. Clone and Setup Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-username/ccc-gpu.git
    cd ccc-gpu

    # Create conda environment from lock file
    conda-lock install --name ccc-gpu conda-lock.yml --conda mamba

    # Activate environment
    conda activate ccc-gpu

    # Install the package in development mode
    pip install -e .

Alternative Setup
~~~~~~~~~~~~~~~~~

If you prefer a simpler approach without conda-lock:

.. code-block:: bash

    # Create basic conda environment
    conda create -n ccc-gpu python=3.9
    conda activate ccc-gpu

    # Install CUDA toolkit and dependencies
    conda install -c conda-forge cudatoolkit-dev cmake ninja
    pip install numpy scipy numba pybind11 scikit-build-core

    # Install the package
    pip install -e .

CUDA Setup
----------

Make sure you have CUDA installed and configured:

.. code-block:: bash

    # Check CUDA installation
    nvcc --version
    nvidia-smi

    # The build system will automatically detect your CUDA installation

Updating Dependencies
---------------------

To update the environment when dependencies change:

.. code-block:: bash

    # Regenerate lock file (for developers)
    conda-lock --file environment/environment-gpu.yml --conda mamba

    # Update existing environment
    conda-lock install --name ccc-gpu conda-lock.yml --conda mamba

Testing
-------

To execute all the test suites, at the root of the repository, run:

.. code-block:: bash

    bash ./scripts/run_tests.sh python

Controlling Debug Logging
--------------------------

By default, CCC-GPU runs silently without debug output. You can enable detailed logging (including CUDA device information, memory usage, and processing details) using the ``CCC_GPU_LOGGING`` environment variable:

.. code-block:: bash

    # Run with default behavior (no debug output)
    python your_script.py

    # Enable debug logging for troubleshooting
    CCC_GPU_LOGGING=1 python your_script.py

    # Or set it for the session
    export CCC_GPU_LOGGING=1
    python your_script.py

This is particularly useful for:

- Debugging GPU memory issues
- Understanding CUDA device utilization  
- Monitoring batch processing performance
- Troubleshooting installation problems