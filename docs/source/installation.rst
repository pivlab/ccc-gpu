Installation
============

Prerequisites
-----------------

Hardware requirements:
- GPU with CUDA Compute Capability 8.6 or higher
Software requirements:
- OS: Linux x86_64

Quick Install with pip
----------------------

The ``cccgpu`` package is now available for installation via pip from test PyPI.

However, note that cccgpu depends on `libstdc++`. For a smooth installation, we recommend using a wrapper conda environment to install it:

.. code-block:: bash

    conda create -n ccc-gpu-toolchain-env -c conda-forge python=3.10 pip pytest libstdcxx-ng && conda activate ccc-gpu-toolchain-env

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

Install from source using the provided conda-lock environment:

1. Clone Repository
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/pivlab/ccc-gpu
    cd ccc-gpu

2. Setup Environment with conda-lock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This process uses a temporary environment to manage the conda-lock installation, keeping your base environment clean:

.. note::
   **Why conda-lock?** We use conda-lock to ensure **reproducible installations** across different systems. Unlike regular ``environment.yml`` files, conda-lock provides exact version pins for all packages and their dependencies, preventing version conflicts and ensuring you get the same environment that was tested during development.

.. code-block:: bash

    # Create temporary environment for conda-lock
    conda create -n ccc-gpu-setup python=3.10 -y  # or: mamba create -n ccc-gpu-setup python=3.10 -y
    conda activate ccc-gpu-setup

    # Install conda-lock in temporary environment
    conda install --channel=conda-forge conda-lock -y  # or: mamba install --channel=conda-forge conda-lock -y

    # Create the main ccc-gpu environment from lock file
    conda-lock install --name ccc-gpu conda-lock.yml  # or: conda-lock install --name ccc-gpu conda-lock.yml --conda mamba

    # Activate the main environment
    conda activate ccc-gpu

    # Install the package from source
    pip install .

3. Optional: Clean up temporary environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once installation is complete, you can optionally remove the temporary setup environment:

.. code-block:: bash

    # Remove temporary environment (optional)
    conda deactivate  # Make sure you're not in ccc-gpu-setup
    conda remove -n ccc-gpu-setup --all -y  # or: mamba remove -n ccc-gpu-setup --all -y

Alternative: Install conda-lock in base environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to install conda-lock directly in your base environment:

.. code-block:: bash

    # Option 1: Using pip
    pip install conda-lock

    # Option 2: Using conda
    conda install --channel=conda-forge conda-lock -y  # or: mamba install --channel=conda-forge conda-lock -y

    # Then create environment directly
    conda-lock install --name ccc-gpu conda-lock.yml  # or: conda-lock install --name ccc-gpu conda-lock.yml --conda mamba
    conda activate ccc-gpu
    pip install .

.. note::
   If you prefer to use Mamba for faster package resolution, you can install MiniForge which includes Mamba:

   .. code-block:: bash

       curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
       bash Miniforge3-$(uname)-$(uname -m).sh -b

   Then replace ``conda`` with ``mamba`` in the commands above.


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
