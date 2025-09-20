# Building Python Wheels for CCC-GPU

This document describes how to build Python wheels for the CCC-GPU project with CUDA support.

## Overview

The project uses `cibuildwheel` to automatically build wheels for multiple Python versions (3.10-3.14) and platforms (Linux and Windows) with CUDA GPU acceleration support.

## GitHub Actions Workflows

### Main Workflow: `build_wheels.yml`

This is the primary workflow that builds wheels for all supported configurations:

- **Triggers**: Push to main/build-multi-os branches, pull requests, tags starting with 'v*'
- **Platforms**: Ubuntu 20.04 (Linux) and Windows 2019
- **Python versions**: 3.10, 3.11, 3.12, 3.13, 3.14
- **CUDA version**: 12.8 (for latest GPU support and features)

#### Key Features:
1. Builds wheels for all Python versions in parallel
2. Automatically installs CUDA toolkit in build environment
3. Bundles CUDA libraries with wheels for standalone distribution
4. Runs basic import tests after building
5. Uploads wheels as GitHub Actions artifacts
6. Optionally publishes to PyPI on version tags

### Simplified Workflow: `build_wheels_simple.yml`

A manual workflow for testing single configurations:

- **Trigger**: Manual dispatch with parameters
- **Configurable**: Choose specific Python version and OS
- **Use case**: Quick testing of build configuration changes

## Configuration Files

### `pyproject.toml`

Contains cibuildwheel configuration:
- Build targets and skip patterns
- Linux-specific CUDA installation commands
- Windows-specific DLL bundling configuration
- Test commands to verify wheel integrity

### `scripts/repair_windows_wheel.py`

Helper script for Windows wheels:
- Bundles CUDA runtime DLLs with the wheel
- Ensures standalone distribution without CUDA installation requirement
- Can be used manually if automatic repair fails

## Building Locally

### Prerequisites

1. Install cibuildwheel:
```bash
pip install cibuildwheel
```

2. Install CUDA Toolkit 12.8 (or compatible version)

### Build Commands

Build all wheels for current platform:
```bash
cibuildwheel --output-dir wheelhouse
```

Build specific Python version:
```bash
CIBW_BUILD="cp311-*" cibuildwheel --output-dir wheelhouse
```

Build with custom CUDA path (Linux):
```bash
export CUDA_HOME=/usr/local/cuda-12.8
cibuildwheel --output-dir wheelhouse
```

## Platform-Specific Notes

### Linux

- Uses manylinux2014 base image for compatibility
- CUDA is installed inside the container via yum
- Wheels are repaired with `auditwheel` to bundle shared libraries
- Compatible with most Linux distributions

### Windows

- Requires Visual Studio C++ build tools
- CUDA is installed on the runner before building
- Uses `delvewheel` to bundle CUDA DLLs
- May need manual DLL path adjustment for non-standard CUDA installations

## CUDA Architecture Support

The project uses CUDA 12.8 and targets CUDA compute capability 7.5 (as specified in CMakeLists.txt). 

### CUDA 12.8 Benefits:
- **Latest Features**: Support for newest CUDA APIs and libraries
- **Performance**: Enhanced compiler optimizations and runtime performance
- **GPU Architecture**: Full support for Ada Lovelace (RTX 40) and Hopper architectures
- **Memory Management**: Improved unified memory and memory pool APIs
- **Driver Compatibility**: Works with the latest NVIDIA drivers (550+)
- **Developer Tools**: Enhanced debugging and profiling capabilities
- **Stability**: Latest bug fixes and security updates

### Supported GPU Architectures:
- GeForce RTX 40 series (Ada Lovelace) - compute capability 8.9
- GeForce RTX 30 series (Ampere) - compute capability 8.6  
- GeForce RTX 20 series (Turing) - compute capability 7.5
- GeForce GTX 16 series - compute capability 7.5
- Quadro RTX series
- Tesla T4 and newer data center GPUs

To support additional architectures, modify the `CMAKE_CUDA_ARCHITECTURES` setting in:
- `CMakeLists.txt`
- `pyproject.toml` (scikit-build configuration)

## Troubleshooting

### Common Issues

1. **CUDA not found during build**
   - Ensure CUDA_HOME/CUDA_PATH environment variable is set
   - Verify CUDA toolkit version compatibility

2. **Wheel import fails with missing DLL**
   - Windows: Run the repair script manually
   - Linux: Check auditwheel output for excluded libraries

3. **Build fails with compiler errors**
   - Verify C++ standard compatibility (requires C++20)
   - Check CUDA toolkit version matches nvcc requirements

### Debug Build

Enable verbose output:
```bash
CIBW_BUILD_VERBOSITY=3 cibuildwheel --output-dir wheelhouse
```

## Publishing to PyPI

The workflow automatically publishes to PyPI when:
1. A tag starting with 'v' is pushed (e.g., v0.2.2)
2. The PYPI_API_TOKEN secret is configured in repository settings

Manual upload:
```bash
twine upload dist/*.whl dist/*.tar.gz
```

## Testing Wheels

After building, test the wheels:

```bash
# Create clean virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install wheel
pip install wheelhouse/cccgpu-*.whl

# Test import
python -c "import ccc_cuda_ext; print('Success!')"

# Run tests
pytest tests/
```

## Maintenance

### Updating CUDA Version

To update the CUDA version:

1. Update `.github/workflows/build_wheels.yml`:
   - Change `cuda-version` in the matrix
   - Update CUDA paths in environment variables

2. Update `pyproject.toml`:
   - Modify CUDA installation commands in `[tool.cibuildwheel.linux]`
   - Update paths in `[tool.cibuildwheel.windows]`

3. Update documentation with new version requirements

### Adding Python Versions

To add support for new Python versions:

1. Update `CIBW_BUILD` environment variable in workflows
2. Add version to `build` list in `pyproject.toml`
3. Update test matrix in workflow

## Resources

- [cibuildwheel documentation](https://cibuildwheel.pypa.io/)
- [scikit-build-core documentation](https://scikit-build-core.readthedocs.io/)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [pybind11 documentation](https://pybind11.readthedocs.io/)