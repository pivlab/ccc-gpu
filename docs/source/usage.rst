Usage
=====

CCC-GPU provides a simple API identical to the original CCC implementation, with GPU acceleration for improved performance on large datasets.

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    # New CCC-GPU implementation import
    from ccc.coef.impl_gpu import ccc
    # Original CCC implementation import (for comparison)
    # from ccc.coef.impl import ccc

    # Generate sample data
    np.random.seed(0)
    x = np.random.randn(1000)
    y = x**2 + np.random.randn(1000) * 0.1  # Non-linear relationship

    # Compute CCC coefficient
    correlation = ccc(x, y)
    print(f"CCC coefficient: {correlation:.3f}")

Comparing CPU and GPU Implementations
-------------------------------------

.. code-block:: python

    import numpy as np
    from ccc.coef.impl_gpu import ccc as ccc_gpu
    from ccc.coef.impl import ccc as ccc_cpu

    # Generate random data
    np.random.seed(42)
    random_feature1 = np.random.rand(1000)
    random_feature2 = np.random.rand(1000)

    # Compute using both implementations
    ccc_value_cpu = ccc_cpu(random_feature1, random_feature2)
    ccc_value_gpu = ccc_gpu(random_feature1, random_feature2)
    
    print(f"CPU result: {ccc_value_cpu:.6f}")
    print(f"GPU result: {ccc_value_gpu:.6f}")
    
    # Results should be nearly identical
    assert np.allclose(ccc_value_cpu, ccc_value_gpu, rtol=1e-5)

Working with Gene Expression Data
---------------------------------

CCC-GPU is particularly useful for genomics applications:

.. code-block:: python

    import pandas as pd
    from ccc.coef.impl_gpu import ccc
    import numpy as np
    from scipy.spatial.distance import squareform

    # Load gene expression data
    # Assume genes are in columns, samples in rows
    gene_expr = pd.read_csv('gene_expression.csv', index_col=0)

    # Compute gene-gene correlations
    gene_correlations = ccc(gene_expr.T)  # Transpose so genes are in rows

    # Convert to square matrix
    corr_matrix = squareform(gene_correlations)
    np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations

    # Find top correlations
    top_indices = np.unravel_index(np.argsort(corr_matrix.ravel())[-10:], corr_matrix.shape)
    gene_names = gene_expr.columns.tolist()

    print("Top 10 gene pairs by CCC:")
    for i, j in zip(top_indices[0], top_indices[1]):
        print(f"{gene_names[i]} - {gene_names[j]}: {corr_matrix[i, j]:.3f}")

Advanced Usage
--------------

Custom Clustering Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ccc.coef.impl_gpu import ccc
    import numpy as np

    # Generate sample data
    data1 = np.random.randn(500)
    data2 = data1**3 + np.random.randn(500) * 0.2

    # Use custom number of clusters (default uses sqrt(n_features))
    ccc_value = ccc(data1, data2, internal_n_clusters=5)
    print(f"CCC with 5 clusters: {ccc_value:.3f}")

    # Use a range of cluster numbers
    ccc_value = ccc(data1, data2, internal_n_clusters=[2, 3, 4, 5, 6])
    print(f"CCC with cluster range [2-6]: {ccc_value:.3f}")

P-value Computation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ccc.coef.impl_gpu import ccc
    import numpy as np

    # Generate sample data
    np.random.seed(123)
    x = np.random.randn(300)
    y = x + np.random.randn(300) * 0.5

    # Compute CCC with p-value using permutations
    ccc_value, p_value = ccc(x, y, pvalue_n_perms=1000)
    print(f"CCC: {ccc_value:.3f}, p-value: {p_value:.3f}")

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ccc.coef.impl_gpu import ccc
    import numpy as np

    # Generate larger dataset
    np.random.seed(456)
    data = np.random.randn(50, 1000)  # 50 features, 1000 samples

    # Use multiple CPU cores for preprocessing
    correlations = ccc(data, n_jobs=4)
    print(f"Computed {len(correlations)} pairwise correlations")

Debug Logging
-------------

Enable detailed logging to monitor GPU performance and troubleshoot issues:

.. code-block:: bash

    # Set environment variable before running Python
    export CCC_GPU_LOGGING=1
    python your_ccc_script.py

Or in Python:

.. code-block:: python

    import os
    os.environ['CCC_GPU_LOGGING'] = '1'
    
    from ccc.coef.impl_gpu import ccc
    # Now CCC will output detailed GPU debug information

Performance Tips
----------------

1. **Large Datasets**: CCC-GPU performs best on datasets with 1000+ features
2. **Memory Usage**: Monitor GPU memory usage for very large datasets
3. **Batch Processing**: For extremely large datasets, consider processing in batches
4. **CUDA Architecture**: Ensure your GPU supports the compiled CUDA architecture (75+)

For more examples, refer to the `original CCC repository <https://github.com/greenelab/ccc>`_.
