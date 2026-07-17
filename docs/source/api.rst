API Reference
=============

The public entry point is the ``ccc()`` function. Two implementations share the
same signature and return contract:

- :func:`ccc.coef.impl_gpu.ccc` -- the GPU-accelerated implementation (recommended
  for large inputs; coefficient values are computed in float32).
- :func:`ccc.coef.impl.ccc` -- the reference CPU implementation (float64).

Both accept a ``pvalue_n_perms`` argument to estimate a one-sided permutation
p-value alongside each coefficient. See :doc:`usage` (the "Computing p-values"
section) for the method, interpretation, and cost.

GPU implementation
------------------

.. autofunction:: ccc.coef.impl_gpu.ccc

CPU implementation
------------------

.. autofunction:: ccc.coef.impl.ccc
