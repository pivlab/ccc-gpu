# Clustermatch Correlation Coefficient GPU (CCC-GPU)

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)

The **Clustermatch Correlation Coefficient (CCC)** is a highly-efficient, next-generation correlation coefficient that captures not-only-linear relationships and can work on numerical and categorical data types. **CCC-GPU** is a GPU-accelerated implementation that provides significant performance improvements for large-scale datasets using CUDA.

CCC is based on the simple idea of clustering data points and then computing the Adjusted Rand Index (ARI) between the two clusterings. It is a robust and efficient method that can detect linear and non-linear relationships, making it suitable for a wide range of applications in genomics, machine learning, and data science.

## Code Structure

- **libs/ccc**: Python code for CCC-GPU
- **libs/ccc_cuda_ext**: CUDA C++ code for CCC-GPU
- **tests**: Test suits
- **nbs**: Notebooks for analysis and visualization

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.0 or higher (for GPU acceleration)
- CMake 3.15 or higher
- A C++20 compatible compiler

### Quick Install (Coming Soon!)

**A conda package will be published soon for easy installation:**

```bash
# Coming soon - simplified conda installation
conda install -c conda-forge ccc-gpu
```

### Install from Source

For now, install from source using the provided conda-lock environment:

#### 1. Install Prerequisites

First, install Mamba (recommended) and conda-lock:

```bash
# Install MiniForge (includes Mamba)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b

# Install conda-lock
pip install conda-lock
```

#### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/ccc-gpu.git
cd ccc-gpu

# Create conda environment from lock file
conda-lock install --name ccc-gpu conda-lock.yml --conda mamba

# Activate environment
conda activate ccc-gpu

# Install the package in development mode
pip install -e .
```

#### Alternative Setup

If you prefer a simpler approach without conda-lock:

```bash
# Create basic conda environment
conda create -n ccc-gpu python=3.9
conda activate ccc-gpu

# Install CUDA toolkit and dependencies
conda install -c conda-forge cudatoolkit-dev cmake ninja
pip install numpy scipy numba pybind11 scikit-build-core

# Install the package
pip install -e .
```

### CUDA Setup

Make sure you have CUDA installed and configured:

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# The build system will automatically detect your CUDA installation
```

### Updating Dependencies

To update the environment when dependencies change:

```bash
# Regenerate lock file (for developers)
conda-lock --file environment/environment-gpu.yml --conda mamba

# Update existing environment
conda-lock install --name ccc-gpu conda-lock.yml --conda mamba
```

## Usage

### Basic Usage

CCC-GPU provides a simple API identical to the original CCC implementation:

```python
import numpy as np
# New CCC-GPU implementation import
from ccc.coef.impl_gpu import ccc
# Original CCC implementation import
# from ccc.coef.impl import ccc

# Generate sample data
np.random.seed(0)
x = np.random.randn(1000)
y = x**2 + np.random.randn(1000) * 0.1  # Non-linear relationship

# Compute CCC coefficient
correlation = ccc(x, y)
print(f"CCC coefficient: {correlation:.3f}")
```

### Working with Gene Expression Data

CCC-GPU is particularly useful for genomics applications:

```python
import pandas as pd
from ccc.coef import ccc

# Load gene expression data
# Assume genes are in columns, samples in rows
gene_expr = pd.read_csv('gene_expression.csv', index_col=0)

# Compute gene-gene correlations
gene_correlations = ccc(gene_expr.T)  # Transpose so genes are in rows

# Find highly correlated gene pairs
import numpy as np
from scipy.spatial.distance import squareform

# Convert to square matrix
corr_matrix = squareform(gene_correlations)
np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations

# Find top correlations
top_indices = np.unravel_index(np.argsort(corr_matrix.ravel())[-10:], corr_matrix.shape)
gene_names = gene_expr.columns.tolist()

print("Top 10 gene pairs by CCC:")
for i, j in zip(top_indices[0], top_indices[1]):
    print(f"{gene_names[i]} - {gene_names[j]}: {corr_matrix[i, j]:.3f}")
```

Refer to the original CCC Repository for more usage examples: [https://github.com/greenelab/ccc](https://github.com/greenelab/ccc)

## Performance Benchmarks

CCC-GPU provides significant performance improvements over CPU-only implementations:

| Number of genes | CCC-GPU vs. CCC (12 cores) |
|---|---|
| 500 | 16.52 |
| 1000 | 30.65 |
| 2000 | 45.72 |
| 4000 | 59.46 |
| 6000 | 67.46 |
| 8000 | 71.48 |
| 10000 | 72.38 |
| 16000 | 73.83 |
| 20000 | 73.88 |

*Benchmarks performed on synthetic gene expression data with 1000 fixed samples. Hardware: AMD Ryzen Threadripper 7960X CPU and an NVIDIA RTX 4090 GPU*

## Documentation

Build and view documentation locally:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in your browser.

If using VS Code, the `Live Preview` extension provides convenient in-editor viewing.

## Citation

If you use CCC-GPU in your research, please cite the original CCC paper:

```bibtex
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
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original CCC implementation: [https://github.com/greenelab/ccc](https://github.com/greenelab/ccc)
- CUDA development team for the excellent CUDA toolkit
- pybind11 for seamless Python-C++ integration
