Introduction
============

Overview
--------

The **Clustermatch Correlation Coefficient (CCC)** is a highly-efficient, next-generation correlation coefficient that captures not-only-linear relationships and can work on numerical and categorical data types. **CCC-GPU** is a GPU-accelerated implementation that provides significant performance improvements for large-scale datasets using CUDA.

CCC is based on the simple idea of clustering data points and then computing the Adjusted Rand Index (ARI) between the two clusterings. It is a robust and efficient method that can detect linear and non-linear relationships, making it suitable for a wide range of applications in genomics, machine learning, and data science.

Key Features
------------

- **Non-linear Correlation Detection**: Unlike traditional correlation methods, CCC can capture complex non-linear relationships
- **Mixed Data Type Support**: Seamlessly handles numerical and categorical data in the same analysis
- **GPU Acceleration**: CUDA-powered implementation provides 16x-74x speedups over CPU versions
- **Scalable**: Efficiently processes large genomics datasets with 20k+ features
- **Robust**: Handles edge cases and provides stable results across different data distributions
- **Easy Integration**: Simple API compatible with existing scientific Python workflows

Scientific Applications
-----------------------

CCC-GPU is particularly valuable for:

**Genomics and Bioinformatics**
  - Gene expression correlation analysis
  - Multi-omics data integration
  - Biomarker discovery
  - Pathway analysis

**Machine Learning**
  - Feature selection and dimensionality reduction
  - Data preprocessing for complex datasets
  - Non-linear dependency detection
  - Mixed-type data analysis

**General Data Science**
  - Exploratory data analysis
  - Complex pattern recognition
  - Time series analysis with non-linear trends
  - Social network analysis

Performance Benchmarks
-----------------------

CCC-GPU provides significant performance improvements over CPU-only implementations:

.. list-table:: Performance Comparison (CCC-GPU vs. CCC with 12 CPU cores)
   :header-rows: 1
   :align: center

   * - Number of Genes
     - Speedup Factor
   * - 500
     - 16.52×
   * - 1,000
     - 30.65×
   * - 2,000
     - 45.72×
   * - 4,000
     - 59.46×
   * - 6,000
     - 67.46×
   * - 8,000
     - 71.48×
   * - 10,000
     - 72.38×
   * - 16,000
     - 73.83×
   * - 20,000
     - 73.88×

*Benchmarks performed on synthetic gene expression data with 1000 fixed samples. Hardware: AMD Ryzen Threadripper 7960X CPU and NVIDIA RTX 4090 GPU.*

The performance scaling demonstrates that CCC-GPU is particularly effective for large-scale datasets commonly encountered in genomics research.

Why CCC?
--------

Traditional correlation measures like Pearson and Spearman correlation have limitations:

- **Pearson correlation**: Only detects linear relationships
- **Spearman correlation**: Limited to monotonic relationships  
- **Mutual information**: Requires careful binning and parameter tuning

CCC addresses these limitations by:

1. **Adaptive Clustering**: Uses quantile-based clustering that adapts to data distribution
2. **Multiple Scales**: Tests different numbers of clusters to capture relationships at various scales  
3. **Robust Statistics**: Based on the Adjusted Rand Index, which is corrected for chance
4. **Parameter-free**: Requires minimal parameter tuning for most applications

Technical Innovation
--------------------

The GPU implementation leverages several technical innovations:

- **CUDA Kernel Optimization**: Custom kernels for ARI computation and maximum finding
- **Memory Hierarchy Utilization**: Strategic use of GPU memory hierarchy for performance
- **Parallel Algorithm Design**: Efficient parallelization of inherently sequential algorithms
- **Memory Management**: Advanced memory management using Thrust and CUB libraries

Project Structure
-----------------

This repository contains:

- **CCC-GPU Library** (``libs/ccc/``): Main Python package with GPU and CPU implementations
- **CUDA Extension** (``libs/ccc_cuda_ext/``): Low-level CUDA implementation with pybind11 bindings
- **Comprehensive Testing**: Extensive test suites for validation and benchmarking
- **Analysis Notebooks**: Real-world examples and research applications
- **Documentation**: Complete usage guides and algorithm explanations

Getting Started
---------------

To get started with CCC-GPU, follow the :doc:`installation` guide and then explore the :doc:`usage` examples. The :doc:`algorithms` section provides detailed information about the underlying mathematical foundations and implementation details.

Citation
--------

If you use CCC-GPU in your research, please cite:

.. code-block:: bibtex

    @article{zhang2025cccgpu,
      title={CCC-GPU: A graphics processing unit (GPU)-optimized nonlinear correlation coefficient for large transcriptomic analyses},
      author={Zhang, Hang and Fotso, Kenneth and Pividori, Milton},
      journal={bioRxiv},
      year={2025},
      publisher={Cold Spring Harbor Laboratory},
      doi={10.1101/2025.06.03.657735},
      pmid={40502087},
      pmcid={PMC12157546}
    }

The original CCC implementation and methodology can be found at: https://github.com/greenelab/ccc
