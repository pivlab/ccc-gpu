# Clustermatch Correlation Coefficient GPU (CCC-GPU)

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://ccc-gpu.readthedocs.io/en/latest/)

The **Clustermatch Correlation Coefficient (CCC)** is a highly-efficient, next-generation correlation coefficient that captures not-only-linear relationships and can work on numerical and categorical data types. **CCC-GPU** is a GPU-accelerated implementation that provides significant performance improvements for large-scale datasets using CUDA.

📖 **Full documentation available at: [https://ccc-gpu.readthedocs.io/en/latest/](https://ccc-gpu.readthedocs.io/en/latest/)**

CCC is based on the simple idea of clustering data points and then computing the Adjusted Rand Index (ARI) between the two clusterings. It is a robust and efficient method that can detect linear and non-linear relationships, making it suitable for a wide range of applications in genomics, machine learning, and data science.

## Code Structure

- **libs/ccc**: Python code for CCC-GPU
- **libs/ccc_cuda_ext**: CUDA C++ code for CCC-GPU
- **tests**: Test suits
- **nbs**: Notebooks for analysis and visualization

## Installation

### Requirements

**Hardware:**
- GPU with CUDA Compute Capability 8.6 or higher

**Software:**
- OS: Linux x86_64, Windows x86_64 (not tested yet)
- Python 3.10 (support for more versions coming soon)
- CUDA 12.0 or higher (for GPU acceleration)

### Quick Install with pip

The `cccgpu` package is available via pip from test PyPI. However, note that cccgpu depends on `libstdc++`. For a smooth installation without compatibility issues with your local system, we recommend using a wrapper conda environment to install it:

```bash
# Create conda environment with required dependencies
conda create -n ccc-gpu-env -c conda-forge python=3.10 pip pytest libstdcxx-ng
conda activate ccc-gpu-env

# Install cccgpu from test PyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            --only-binary=cccgpu cccgpu

# Verify installation
python -c "from ccc.coef.impl_gpu import ccc as ccc_gpu; import numpy as np; print(ccc_gpu(np.random.rand(100), np.random.rand(100)))"
```

Support for more Python versions and architectures requires extra effort, and will be added soon.

**Note:** This installs from test PyPI while the package is in testing phase. Once stable, it will be available from the main PyPI repository with `pip install cccgpu`.

**Command options explained:**

- ``--index-url https://test.pypi.org/simple/``: Specifies test PyPI as the primary package index to search for ``cccgpu``
- ``--extra-index-url https://pypi.org/simple/``: Adds the main PyPI repository as a fallback to install dependencies (numpy, scipy, numba, etc.) that may not be available on test PyPI
- ``--only-binary=cccgpu``: Ensures that only binary wheels are installed for ``cccgpu`` package, so you don't need to compile it from source
- ``cccgpu``: The package name to install

### Install from Source

Install from source using the provided conda-lock environment:

#### 1. Clone Repository

```bash
# Clone the repository
git clone https://github.com/pivlab/ccc-gpu
cd ccc-gpu
```

#### 2. Setup Environment with conda-lock

This process uses a temporary environment to manage the conda-lock installation, keeping your base environment clean:

> **Why conda-lock?** We use conda-lock to ensure **reproducible installations** across different systems. Unlike regular `environment.yml` files, conda-lock provides exact version pins for all packages and their dependencies, preventing version conflicts and ensuring you get the same environment that was tested during development.

```bash
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
```

#### 3. Optional: Clean up temporary environment

Once installation is complete, you can optionally remove the temporary setup environment:

```bash
# Remove temporary environment (optional)
conda deactivate  # Make sure you're not in ccc-gpu-setup
conda remove -n ccc-gpu-setup --all -y  # or: mamba remove -n ccc-gpu-setup --all -y
```

#### Alternative: Install conda-lock in base environment

If you prefer to install conda-lock directly in your base environment:

```bash
# Option 1: Using pip
pip install conda-lock

# Option 2: Using conda
conda install --channel=conda-forge conda-lock -y  # or: mamba install --channel=conda-forge conda-lock -y

# Then create environment directly
conda-lock install --name ccc-gpu conda-lock.yml  # or: conda-lock install --name ccc-gpu conda-lock.yml --conda mamba
conda activate ccc-gpu
pip install .
```

> **Note**: If you prefer to use Mamba for faster package resolution, you can install MiniForge which includes Mamba:
> ```bash
> curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
> bash Miniforge3-$(uname)-$(uname -m).sh -b
> ```
> Then replace `conda` with `mamba` in the commands above.


## Testing
To execute all the test suites, at the root of the repository, run:

```bash
conda activate ccc-gpu
bash ./scripts/run_tests.sh python
```

## Usage
### End-to-End Tutorial

You can find a tutorial showing simplified analysis steps for those we used in our paper in this [notebook](nbs/99-tutorials/05-walkthrough-with-gtex-data.ipynb) using the GTEx v8 data.


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
# New CCC-GPU implementation import
from ccc.coef.impl_gpu import ccc
# Original CCC implementation import
# from ccc.coef.impl import ccc

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

### Controlling Debug Logging

By default, CCC-GPU runs silently without debug output. You can enable detailed logging (including CUDA device information, memory usage, and processing details) using the `CCC_GPU_LOGGING` environment variable:

```bash
# Run with default behavior (no debug output)
python your_script.py

# Enable debug logging for troubleshooting
CCC_GPU_LOGGING=1 python your_script.py

# Or set it for the session
export CCC_GPU_LOGGING=1
python your_script.py
```

This is particularly useful for:
- Debugging GPU memory issues
- Understanding CUDA device utilization
- Monitoring batch processing performance
- Troubleshooting installation problems

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

*Benchmarks performed on synthetic gene expression data with 1000 fixed samples. Hardware: AMD Ryzen Threadripper 7960X CPU and an NVIDIA RTX 4090 GPU. Git commit on which the benchmark results were collected: 05f129dfa47ad801eff963b4189484c7c64bd28e*

## Documentation

Build and view the full documentation locally:

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
  author={Zhang, Haoyu and Fotso, Kevin and Pividori, Milton},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.06.03.657735},
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
