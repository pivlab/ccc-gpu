.. ccc-gpu documentation master file, created by
   sphinx-quickstart on Thu Jan  9 15:14:14 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CCC-GPU Documentation
=====================

The **Clustermatch Correlation Coefficient (CCC)** is a highly-efficient, next-generation correlation coefficient that captures not-only-linear relationships and can work on numerical and categorical data types. **CCC-GPU** is a GPU-accelerated implementation that provides significant performance improvements for large-scale datasets using CUDA.

CCC is based on the simple idea of clustering data points and then computing the Adjusted Rand Index (ARI) between the two clusterings. It is a robust and efficient method that can detect linear and non-linear relationships, making it suitable for a wide range of applications in genomics, machine learning, and data science.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   introduction
   usage
   algorithms
   development/index
