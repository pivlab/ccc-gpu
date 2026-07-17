Python/CUDA Bindings
====================

The GPU coefficient computation lives in a compiled extension module,
``ccc_cuda_ext``, built from the CUDA C++ sources in ``libs/ccc_cuda_ext/`` and
exposed to Python with `pybind11 <https://pybind11.readthedocs.io/>`_.

Layout
------

- ``binder.cu`` -- the pybind11 module definition (``PYBIND11_MODULE``). It
  declares the Python-visible functions and converts between NumPy arrays and
  the device buffers used by the kernels.
- ``coef.cu`` / ``coef.cuh`` -- ``compute_coef``, the primary entry point used by
  :func:`ccc.coef.impl_gpu.ccc`. Given the precomputed partitions it runs the
  ARI kernels on the GPU and returns the coefficients, the maximizing partition
  indexes, and (optionally) permutation p-values.
- ``metrics.cu`` / ``metrics.cuh`` -- the Adjusted Rand Index kernels.
- ``math.cuh`` -- small device-side helpers.

How it is built
---------------

The extension is compiled by CMake (see the root ``CMakeLists.txt``) and driven
by ``scikit-build-core`` during ``pip install``. ``pybind11_add_module`` builds
the shared object, and the CUDA architectures the wheel targets are set via
``-DCMAKE_CUDA_ARCHITECTURES`` in ``pyproject.toml`` (``[tool.scikit-build]``).
See :doc:`build_cuda_module` for a step-by-step local build.

Calling convention
------------------

``ccc.coef.impl_gpu`` prepares the per-feature partitions on the host (NumPy) and
hands them to ``ccc_cuda_ext.compute_coef`` together with the feature/cluster/
object counts and the ``return_parts`` / ``pvalue_n_perms`` flags. The extension
returns NumPy arrays, which the Python layer reshapes into the polymorphic return
value documented in :doc:`../api`.
