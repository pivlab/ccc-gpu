.. _installation:

===============
Installation
===============

This section describes how to set up the development environment and install the package from source.

Prerequisites
------------------------

This project uses Mamba and Conda-Lock for dependency management. You'll need to install these tools first:

1. Installing Mamba
~~~~~~~~~~~~~~~~~~~~

Install Mamba globally using the MiniForge distribution:

.. code-block:: bash

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b

For detailed instructions, see the `MiniForge Repository <https://github.com/conda-forge/miniforge>`_.

2. Installing Conda-Lock
~~~~~~~~~~~~~~~~~~~~~~~~~

Install Conda-Lock globally:

.. code-block:: bash

    pip install conda-lock

Alternative Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer not to install these tools globally, you can create a dedicated conda environment:

.. code-block:: bash

    conda create -n ccc-gpu-dev -c conda-forge mamba conda-lock
    conda activate ccc-gpu-dev

Setting Up the Development Environment
---------------------------------------


Create the Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the environment using the pre-generated lock file:

.. code-block:: bash

    conda-lock install --name ccc-gpu conda-lock.yml --conda mamba

Installing the Package
~~~~~~~~~~~~~~~~~~~~~~~~~

Activate the conda environment and install the package from source:

.. code-block:: bash

    conda activate ccc-gpu
    # (Make sure you are at the root of the repository)
    pip install -e .

To uninstall:

.. code-block:: bash

    pip uninstall ccc-gpu

Updating Dependencies
----------------------

To update the environment when dependencies change:

.. code-block:: bash

    # Regenerate the lock file
    conda-lock --file environment/environment-gpu.yml --conda mamba

    # Update the environment
    conda-lock install --name ccc-gpu conda-lock.yml --conda mamba

.. note::
    Future versions may transition to using `Pixi <https://pixi.sh/>`_ for dependency management.
