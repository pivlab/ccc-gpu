Usage
===============

The main exposed APIs are indentical to the original CCC implementation.

.. code-block:: python

    from ccc.coef.impl_gpu import ccc as ccc_gpu
    from ccc.coef.impl import ccc

    random_feature1 = np.random.rand(1000)
    random_feature2 = np.random.rand(1000)

    ccc_value = ccc(random_feature1, random_feature2)
    ccc_value_gpu = ccc_gpu(random_feature1, random_feature2)
    assert np.allclose(ccc_value, ccc_value_gpu)


Check :ref:`installation` for more details about how to install the package.
