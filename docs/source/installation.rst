Installation
============

Prerequisites
-----------------

Hardware requirements:

- NVIDIA GPU with CUDA compute capability 7.5 or higher (the wheels ship native
  code for 7.5, 8.0, 8.6, 8.9, and 9.0)

Software requirements:

- OS: Linux x86_64 (glibc 2.28 or later)
- Python 3.10 to 3.14
- NVIDIA driver providing CUDA 12.0 or higher

Quick Install with pip
----------------------

The ``cccgpu`` package is available on PyPI:

.. code-block:: bash

    pip install cccgpu

``cccgpu`` depends on ``libstdc++``. If your system copy is too old, install it
into a conda environment first, for example:

.. code-block:: bash

    conda create -n ccc-gpu -c conda-forge python=3.12 pip pytest libstdcxx-ng
    conda activate ccc-gpu
    pip install cccgpu

Then verify the installation:

.. code-block:: bash

    python -c "from ccc.coef.impl_gpu import ccc as ccc_gpu; import numpy as np; print(ccc_gpu(np.random.rand(100), np.random.rand(100)))"


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

This process uses pipx to install conda-lock in an isolated environment, keeping your base environment clean:

.. note::
   **Why conda-lock?** We use conda-lock to ensure **reproducible installations** across different systems. Unlike regular ``environment.yml`` files, conda-lock provides exact version pins for all packages and their dependencies, preventing version conflicts and ensuring you get the same environment that was tested during development.

.. code-block:: bash

    # Install conda-lock using pipx (installs in isolated environment)
    pipx install conda-lock

    # Create the main ccc-gpu environment from lock file
    conda-lock install --name ccc-gpu conda-lock.yml  # or: conda-lock install --name ccc-gpu conda-lock.yml --conda mamba

    # Activate the main environment
    conda activate ccc-gpu

    # Install the package from source
    pip install .

.. note::
   If you don't have pipx installed, you can install it with ``pip install pipx`` or follow the `pipx installation guide <https://pypa.github.io/pipx/installation/>`_.

3. Optional: Remove conda-lock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you no longer need conda-lock after installation, you can remove it:

.. code-block:: bash

    # Remove conda-lock (optional)
    pipx uninstall conda-lock

Alternative: Install conda-lock in base environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to install conda-lock directly in your base environment instead of using pipx:

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
