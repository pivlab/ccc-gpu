# CCC-GPU PyPI Build Scripts

This directory contains scripts for building and uploading Python wheels for the `cccgpu` package across multiple Python versions (3.10-3.15).

## 📋 Table of Contents

- [Overview](#overview)
- [Scripts](#scripts)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Build Methods](#build-methods)
- [Manual Testing Instructions](#manual-testing-instructions)
- [Troubleshooting](#troubleshooting)
- [Implementation Progress](#implementation-progress)
- [Technical Notes](#technical-notes)

---

## Overview

The `cccgpu` package is a GPU-accelerated Python extension built with:
- **PyBind11** (Python-C++ bindings)
- **CUDA** (GPU acceleration)
- **CMake** + **scikit-build-core** (build system)

Building wheels for multiple Python versions requires special handling because:
1. Each Python version has a different ABI (Application Binary Interface)
2. CUDA dependencies must be available during compilation
3. Wheels must be `manylinux` compatible for broad Linux distribution support

---

## Scripts

| Script | Purpose | Speed | Output |
|--------|---------|-------|--------|
| `00-build_package.sh` | Legacy single-version builder | Fast | 1 wheel (Python 3.10) |
| `01-build-multi-python-conda.sh` | Local multi-version builds | **Fast** | 6 wheels (3.10-3.15) |
| `02-build-with-cibuildwheel.sh` | Production manylinux builds | Slow | 6 manylinux wheels |
| `03-list-built-wheels.sh` | Display built wheels | Instant | Summary table |
| `10-upload_to_test_pypi.sh` | Upload to test PyPI | Medium | Published packages |

### Recommended Workflow

**For development/testing:**
```bash
./01-build-multi-python-conda.sh  # Fast, uses existing CUDA
./03-list-built-wheels.sh         # Verify builds
```

**For PyPI release:**
```bash
./02-build-with-cibuildwheel.sh   # Production-grade manylinux wheels
./03-list-built-wheels.sh         # Verify builds
./10-upload_to_test_pypi.sh       # Upload to test PyPI
```

---

## Requirements

### For Method 1 (Local Conda Builds - Recommended)

- **Conda or Mamba** (any version)
- **CUDA 12.5** from nvidia channel (installed automatically per environment)
- **Existing `ccc-gpu` conda environment** (for reference, not required)
- **Disk space**: ~5GB per Python environment (includes CUDA)
- **Time**: ~20-30 minutes for all 6 versions (first time, includes CUDA download)

### For Method 2 (cibuildwheel - Production)

- **Docker** installed and running
- **Python 3.10+** (to run cibuildwheel)
- **Internet connection** (downloads CUDA in containers)
- **Disk space**: ~15GB (Docker images + builds)
- **Time**: ~1-2 hours (first run with CUDA download)

### For Upload

- **Test PyPI account** (https://test.pypi.org/)
- **API token** or credentials in `~/.pypirc` or `.pypirc`

---

## Quick Start

### 1. Build wheels for all Python versions (3.10-3.15)

**Option A: Fast local builds**
```bash
cd /path/to/ccc-gpu
./scripts/pypi/01-build-multi-python-conda.sh
```

**Option B: Production builds**
```bash
cd /path/to/ccc-gpu
./scripts/pypi/02-build-with-cibuildwheel.sh
```

### 2. Verify built wheels
```bash
./scripts/pypi/03-list-built-wheels.sh
```

Expected output:
```
Built wheels in dist/:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Filename                                          Size
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cccgpu-0.2.1-cp310-cp310-linux_x86_64.whl         5.2 MB
cccgpu-0.2.1-cp311-cp311-linux_x86_64.whl         5.2 MB
cccgpu-0.2.1-cp312-cp312-linux_x86_64.whl         5.2 MB
cccgpu-0.2.1-cp313-cp313-linux_x86_64.whl         5.2 MB
cccgpu-0.2.1-cp314-cp314-linux_x86_64.whl         5.2 MB
cccgpu-0.2.1-cp315-cp315-linux_x86_64.whl         5.2 MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 6 wheels
```

### 3. Upload to test PyPI
```bash
./scripts/pypi/10-upload_to_test_pypi.sh
```

---

## Build Methods

### Method 1: Local Conda Builds (Fast, Recommended)

**Script:** `01-build-multi-python-conda.sh`

**How it works:**
1. Creates separate conda environments for Python 3.10, 3.11, 3.12, 3.13, 3.14, 3.15
2. Installs CUDA 12.5 from nvidia channel in each environment
3. Installs cmake, ninja, pybind11, and other build tools
4. Sets CUDA environment variables pointing to conda's CUDA installation
5. Builds wheel using `python -m build --no-isolation` (allows access to conda packages)
6. Optionally repairs wheels with `auditwheel` (if installed)
7. Collects all wheels in `dist/` directory

**Advantages:**
- ✅ No Docker required
- ✅ Easy to debug
- ✅ Iterative development
- ✅ Self-contained conda environments

**Disadvantages:**
- ❌ Wheels may not be portable to other Linux distros (depends on system libraries)
- ❌ Large disk space (CUDA installed per environment)
- ❌ First build slower (downloads CUDA for each env)

**Usage:**
```bash
# Clean previous builds (optional)
rm -rf dist/

# Build all versions
./scripts/pypi/01-build-multi-python-conda.sh

# Build specific versions only (modify script if needed)
# Edit the PYTHON_VERSIONS array in the script
```

**Environment names created:**
- `ccc-gpu-py310`
- `ccc-gpu-py311`
- `ccc-gpu-py312`
- `ccc-gpu-py313`
- `ccc-gpu-py314`
- `ccc-gpu-py315`

---

### Method 2: cibuildwheel (Production, manylinux)

**Script:** `02-build-with-cibuildwheel.sh`

**How it works:**
1. Runs Docker containers with `manylinux_2_28` base image
2. Downloads and installs CUDA 12.6 inside each container
3. Builds wheels for Python 3.10-3.15
4. Repairs wheels with `auditwheel` (bundles dependencies)
5. Produces `manylinux_2_28_x86_64` wheels (PyPI-ready)

**Advantages:**
- ✅ Produces portable manylinux wheels
- ✅ Compatible with most Linux distributions
- ✅ PyPI-ready
- ✅ Reproducible builds

**Disadvantages:**
- ❌ Slower (downloads CUDA each time unless cached)
- ❌ Requires Docker
- ❌ Large disk space usage

**Usage:**
```bash
# Ensure Docker is running
docker ps

# Clean previous builds (optional)
rm -rf dist/

# Build all versions
./scripts/pypi/02-build-with-cibuildwheel.sh

# Advanced: Build specific Python versions
CIBW_BUILD="cp310-*" cibuildwheel --platform linux --output-dir dist
```

**Configuration:**
All cibuildwheel settings are in `pyproject.toml` under `[tool.cibuildwheel]`.

---

## Manual Testing Instructions

After building wheels, test each Python version manually to ensure compatibility.

### Prerequisites
- Clean test environments (no development dependencies)
- GPU with CUDA Compute Capability 8.6+ available
- CUDA 12.0+ runtime libraries installed

### Test Procedure

#### Step 1: Create test environment

```bash
# For Python 3.10
conda create -n test-cccgpu-310 python=3.10 -y
conda activate test-cccgpu-310

# Install minimal dependencies (no development packages)
pip install pytest numpy scipy pandas scikit-learn numba
```

#### Step 2: Install the wheel

```bash
# Install the wheel for Python 3.10
pip install dist/cccgpu-0.2.1-cp310-cp310-*.whl

# Verify installation
pip show cccgpu
```

#### Step 3: Run basic import test

```bash
python -c "
import numpy as np
from ccc.coef.impl_gpu import ccc

# Test with small arrays
x = np.random.rand(100)
y = x**2 + np.random.rand(100) * 0.1

result = ccc(x, y)
print(f'CCC result: {result}')
assert 0 <= result <= 1, 'CCC should be between 0 and 1'
print('✓ Basic test passed')
"
```

#### Step 4: Run test suite (optional)

```bash
# Clone the repository if testing in a clean environment
cd /path/to/ccc-gpu
pytest tests/ -v
```

#### Step 5: Test with GPU logging enabled

```bash
CCC_GPU_LOGGING=1 python -c "
import numpy as np
from ccc.coef.impl_gpu import ccc

x = np.random.rand(1000)
y = np.random.rand(1000)
result = ccc(x, y)
print(f'Result: {result}')
"
```

Expected output should include CUDA device information:
```
[debug] CUDA Device Info:
[debug] Device 0: "NVIDIA RTX 4090"
[debug] Free memory: 23.X GB, Total memory: 24.0 GB
...
```

#### Step 6: Repeat for all Python versions

```bash
# Python 3.11
conda create -n test-cccgpu-311 python=3.11 -y
conda activate test-cccgpu-311
pip install dist/cccgpu-0.2.1-cp311-cp311-*.whl
# ... repeat tests ...

# Python 3.12
conda create -n test-cccgpu-312 python=3.12 -y
conda activate test-cccgpu-312
pip install dist/cccgpu-0.2.1-cp312-cp312-*.whl
# ... repeat tests ...

# Python 3.13
conda create -n test-cccgpu-313 python=3.13 -y
conda activate test-cccgpu-313
pip install dist/cccgpu-0.2.1-cp313-cp313-*.whl
# ... repeat tests ...

# Python 3.14 (if available)
conda create -n test-cccgpu-314 python=3.14 -y
conda activate test-cccgpu-314
pip install dist/cccgpu-0.2.1-cp314-cp314-*.whl
# ... repeat tests ...

# Python 3.15 (if available)
conda create -n test-cccgpu-315 python=3.15 -y
conda activate test-cccgpu-315
pip install dist/cccgpu-0.2.1-cp315-cp315-*.whl
# ... repeat tests ...
```

### Test Checklist

Create a test results table:

| Python Version | Import Test | Basic CCC Test | GPU Logging | Test Suite | Status |
|----------------|-------------|----------------|-------------|------------|--------|
| 3.10 | ☐ | ☐ | ☐ | ☐ | ⬜ |
| 3.11 | ☐ | ☐ | ☐ | ☐ | ⬜ |
| 3.12 | ☐ | ☐ | ☐ | ☐ | ⬜ |
| 3.13 | ☐ | ☐ | ☐ | ☐ | ⬜ |
| 3.14 | ☐ | ☐ | ☐ | ☐ | ⬜ |
| 3.15 | ☐ | ☐ | ☐ | ☐ | ⬜ |

---

## Troubleshooting

### Common Issues

#### 1. "Command not found: conda" or environment activation fails

**Cause:** Conda not properly initialized in shell

**Solution:**
```bash
# Initialize conda for bash
eval "$(conda shell.bash hook)"
# or for mamba
eval "$(mamba shell.bash hook)"

# Then retry
conda activate ccc-gpu
```

#### 2. "Failed to find nvcc. Compiler requires the CUDA toolkit"

**Cause:** Either CUDA not installed in conda environment, OR build isolation preventing access to conda CUDA

**Solution A - Automated (script handles this):**
The script now:
1. Installs CUDA 12.5 in each environment
2. Sets CUDA environment variables
3. Uses `--no-isolation` flag to access conda packages

**Solution B - Manual fix if script fails:**
```bash
# Activate the problematic environment
conda activate ccc-gpu-py310  # or other version

# Ensure CUDA is installed
conda install -c nvidia -c conda-forge cuda=12.5 cmake ninja pybind11 -y

# Verify CUDA is accessible
which nvcc
nvcc --version

# Set environment variables
export CUDA_HOME="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=${CONDA_PREFIX}/bin/nvcc"

# Build manually with no-isolation
python -m build --wheel --no-isolation --outdir dist/
```

**Why `--no-isolation` is needed:**
By default, `python -m build` creates an isolated environment that doesn't have access to conda packages. The `--no-isolation` flag allows the build to access conda's CUDA installation.

**Important:** When using `--no-isolation`, ALL dependencies from `pyproject.toml` `[build-system] requires` must be installed in the environment:
- `scikit-build-core>=0.10`
- `pybind11>=2.11.0`
- `cmake>=3.15`
- `ninja`
- `setuptools>=42`
- `wheel`

The script installs all of these automatically.

#### 3. "CUDA not found" during build (system-wide)

**Cause:** CUDA not in PATH or CUDA_HOME not set (rare with conda CUDA)

**Solution:**
```bash
# Check CUDA installation
which nvcc
echo $CUDA_HOME

# If using system CUDA instead of conda, add to your ~/.bashrc:
export CUDA_HOME=/usr/local/cuda-12.6  # Adjust version
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Then reload
source ~/.bashrc
```

#### 4. "Docker daemon not running"

**Cause:** Docker service not started

**Solution:**
```bash
# On Ubuntu/Debian
sudo systemctl start docker
sudo systemctl enable docker

# Verify
docker ps
```

#### 5. "Permission denied" on Docker

**Cause:** User not in docker group

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, then verify
docker ps
```

#### 6. "Python 3.14 or 3.15 not available"

**Cause:** These versions may not be released yet or not in conda-forge

**Solution:**
```bash
# Check available versions
conda search python=3.14 -c conda-forge
conda search python=3.15 -c conda-forge

# If not available, the script will skip them automatically
# You can also use pyenv to install pre-release versions:
pyenv install 3.14.0a1  # Example for alpha release
```

#### 7. "Wheel is not compatible" when installing

**Cause:** Wrong platform or Python version

**Solution:**
```bash
# Check wheel filename
ls -la dist/

# Filename format: cccgpu-{version}-{python}-{abi}-{platform}.whl
# Example: cccgpu-0.2.1-cp310-cp310-linux_x86_64.whl
#          Package version: 0.2.1
#          Python: cp310 (CPython 3.10)
#          ABI: cp310
#          Platform: linux_x86_64

# Ensure you're installing the correct wheel for your Python version
python --version  # Should match cpXXX in wheel name
```

#### 8. "ImportError: libcudart.so.12 not found"

**Cause:** CUDA runtime libraries not in system path

**Solution:**
```bash
# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Or install CUDA runtime from conda
conda install -c nvidia cuda-runtime
```

#### 9. Build fails with "compute capability 7.5 not supported"

**Cause:** Your GPU is older than the target compute capability

**Solution:**
Edit `pyproject.toml` and change:
```toml
cmake.args = [
    "-DCMAKE_CUDA_ARCHITECTURES=52",  # Change from 75 to 52 (or your GPU's arch)
]
```

GPU Compute Capabilities:
- 5.2: Maxwell (GTX 900 series)
- 6.0: Pascal (GTX 1000 series)
- 7.0: Volta (V100)
- 7.5: Turing (RTX 2000 series)
- 8.0: Ampere (A100)
- 8.6: Ampere (RTX 3000 series)
- 8.9: Ada Lovelace (RTX 4000 series)

#### 10. "No space left on device"

**Cause:** Insufficient disk space

**Solution:**
```bash
# Clean conda caches
conda clean --all -y

# Clean Docker images
docker system prune -a

# Clean old builds
rm -rf dist/ build/ *.egg-info
```

---

## Implementation Progress

### Development Tasks

- [x] Updated `pyproject.toml` (removed `wheel.py-api` constraint)
- [x] Added cibuildwheel configuration to `pyproject.toml`
- [x] Created `scripts/pypi/README.md` (this file)
- [ ] Created `scripts/pypi/01-build-multi-python-conda.sh`
- [ ] Created `scripts/pypi/02-build-with-cibuildwheel.sh`
- [ ] Created `scripts/pypi/03-list-built-wheels.sh`
- [ ] Updated `scripts/pypi/00-build_package.sh` with documentation

### Build Testing

- [ ] Built wheels for Python 3.10
- [ ] Built wheels for Python 3.11
- [ ] Built wheels for Python 3.12
- [ ] Built wheels for Python 3.13
- [ ] Built wheels for Python 3.14 (if available)
- [ ] Built wheels for Python 3.15 (if available)

### Manual Testing

- [ ] Python 3.10: Import test
- [ ] Python 3.10: Basic CCC test
- [ ] Python 3.10: GPU logging test
- [ ] Python 3.10: Full test suite
- [ ] Python 3.11: Import test
- [ ] Python 3.11: Basic CCC test
- [ ] Python 3.11: GPU logging test
- [ ] Python 3.11: Full test suite
- [ ] Python 3.12: Import test
- [ ] Python 3.12: Basic CCC test
- [ ] Python 3.12: GPU logging test
- [ ] Python 3.12: Full test suite
- [ ] Python 3.13: Import test
- [ ] Python 3.13: Basic CCC test
- [ ] Python 3.13: GPU logging test
- [ ] Python 3.13: Full test suite
- [ ] Python 3.14: Import test (if available)
- [ ] Python 3.14: Basic CCC test (if available)
- [ ] Python 3.14: GPU logging test (if available)
- [ ] Python 3.14: Full test suite (if available)
- [ ] Python 3.15: Import test (if available)
- [ ] Python 3.15: Basic CCC test (if available)
- [ ] Python 3.15: GPU logging test (if available)
- [ ] Python 3.15: Full test suite (if available)

### Upload and Distribution

- [ ] Uploaded to test PyPI
- [ ] Verified installation from test PyPI (Python 3.10)
- [ ] Verified installation from test PyPI (Python 3.11)
- [ ] Verified installation from test PyPI (Python 3.12)
- [ ] Verified installation from test PyPI (Python 3.13)
- [ ] Verified installation from test PyPI (Python 3.14)
- [ ] Verified installation from test PyPI (Python 3.15)
- [ ] Ready for production PyPI release

---

## Technical Notes

### Python Version Support

- **Python 3.10-3.13:** Fully supported and stable
- **Python 3.14:** Beta/RC phase (as of 2025)
- **Python 3.15:** Alpha phase (as of 2025)

**Note:** Python 3.14 and 3.15 wheels are experimental. They will be built if the Python versions are available in conda-forge, but may have compatibility issues.

### CUDA Compatibility

- **Host builds (Method 1):** Uses CUDA installed on your system (12.0+ recommended)
- **cibuildwheel builds (Method 2):** Downloads and installs CUDA 12.6 in containers
- **Runtime requirement:** CUDA 12.0+ runtime libraries on target system

### Compute Capability

Currently targeting **CUDA compute capability 7.5** (RTX 2000 series and newer).

To support older GPUs, modify `pyproject.toml`:
```toml
cmake.args = ["-DCMAKE_CUDA_ARCHITECTURES=52"]  # For GTX 900 series
```

### manylinux Compatibility

**manylinux_2_28** (used in Method 2) is compatible with:
- Ubuntu 20.04+
- Debian 11+
- RHEL 8+
- Fedora 32+
- Most modern Linux distributions

### Wheel Size

Expected wheel sizes:
- **Local builds (Method 1):** ~5-10 MB (depends on system dependencies)
- **manylinux builds (Method 2):** ~10-15 MB (includes bundled dependencies)

### Build Time

Approximate build times:
- **Single wheel (Method 1):** ~2-3 minutes
- **All 6 wheels (Method 1):** ~10-15 minutes
- **Single wheel (Method 2):** ~15-20 minutes (first time with CUDA download)
- **All 6 wheels (Method 2):** ~1-2 hours (first time)

Subsequent builds are faster due to Docker layer caching.

---

## Additional Resources

### Documentation
- [cibuildwheel documentation](https://cibuildwheel.pypa.io/)
- [PyPA packaging guide](https://packaging.python.org/)
- [manylinux specification](https://github.com/pypa/manylinux)
- [CUDA downloads](https://developer.nvidia.com/cuda-downloads)

### CCC-GPU Project
- [Main repository](https://github.com/pivlab/ccc-gpu)
- [Original CCC implementation](https://github.com/greenelab/ccc)
- [CCC paper](https://www.biorxiv.org/content/10.1101/2025.06.03.657735v1)

---

## Support

For issues related to:
- **Building wheels:** Check this README, troubleshooting section
- **CUDA errors:** Verify CUDA installation and compute capability
- **Package functionality:** See main repository README.md
- **PyPI upload:** Check test PyPI credentials and `.pypirc` configuration

---

**Last Updated:** 2025-01-29
**Maintained by:** CCC-GPU Development Team