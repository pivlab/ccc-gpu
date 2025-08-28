# Publishing CCC-GPU to PyPI

This document describes how to build and publish the cccgpu package to PyPI and test PyPI.

## Prerequisites

1. **Conda environment**: Ensure you have the `ccc-gpu` conda environment set up
2. **PyPI account**: Register at [PyPI](https://pypi.org/) and [test PyPI](https://test.pypi.org/)
3. **API tokens**: Generate API tokens for authentication (recommended over passwords)

## Setup

### 1. Configure PyPI credentials

Copy the example configuration and update with your credentials:

```bash
cp .pypirc.example ~/.pypirc
chmod 600 ~/.pypirc  # Protect your credentials
```

Edit `~/.pypirc` and add your API tokens:
- Get test PyPI token from: https://test.pypi.org/manage/account/token/
- Get PyPI token from: https://pypi.org/manage/account/token/

### 2. Make scripts executable

```bash
chmod +x scripts/pypi/00-build_package.sh 
chmod +x scripts/pypi/10-upload_to_test_pypi.sh
```

## Building the Package

The package uses `scikit-build-core` to build C++/CUDA extensions along with the Python package.

### Build command:

```bash
./scripts/pypi/00-build_package.sh 
```

This script will:
1. Activate the `ccc-gpu` conda environment
2. Clean previous builds
3. Install/update build dependencies (`build`, `twine`, `setuptools`, `wheel`, `auditwheel`)
4. Build both source distribution (`.tar.gz`) and wheel (`.whl`)
5. Fix wheel platform tags for PyPI compatibility (convert `linux_x86_64` to `manylinux_2_17_x86_64`)
6. Place built packages in `dist/` directory

### Manual build (if needed):

```bash
mamba activate ccc-gpu
python -m pip install --upgrade build auditwheel
python -m build
# Fix wheel tags if needed
auditwheel repair dist/*.whl --plat-tag manylinux_2_17_x86_64 --wheel-dir dist/
```

## Publishing to Test PyPI

Test PyPI is a separate instance for testing package uploads without affecting the main index.

### Upload to test PyPI:

```bash
./scripts/pypi/10-upload_to_test_pypi.sh
```

This script will:
1. Activate the conda environment
2. Check package integrity with `twine check`
3. Upload to test PyPI
4. Provide installation instructions

### Manual upload (if needed):

```bash
mamba activate ccc-gpu
python -m twine upload --repository testpypi dist/*
```

### Installing from test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            cccgpu
```

Note: The `--extra-index-url` is needed to install dependencies from the main PyPI.

## Publishing to Production PyPI

Once tested, publish to the main PyPI:

```bash
mamba activate ccc-gpu
python -m twine upload dist/*
```

### Installing from PyPI:

```bash
pip install cccgpu
```

## Version Management

Before building a new release:

1. Update version in `pyproject.toml`:
   ```toml
   [project]
   version = "0.2.1"  # Increment as needed
   ```

2. Tag the release:
   ```bash
   git tag -a v0.2.1 -m "Release version 0.2.1"
   git push origin v0.2.1
   ```

## Troubleshooting

### CUDA/GPU Dependencies

The package requires CUDA toolkit for building. Users installing from PyPI need:
- CUDA toolkit installed
- Compatible GPU
- Appropriate CUDA version matching the build

### Build Errors

If build fails with CUDA errors:
1. Ensure CUDA toolkit is installed and accessible
2. Check `CUDAToolkit_ROOT` environment variable
3. Verify GPU and CUDA compatibility

### Authentication Issues

If upload fails with authentication errors:
1. Verify API tokens in `~/.pypirc`
2. Use `__token__` as username with API tokens
3. Ensure tokens have upload permissions

### Package Already Exists

If version already exists:
1. Increment version in `pyproject.toml`
2. Rebuild the package
3. Upload the new version

## CI/CD Integration

For automated publishing, set these secrets in your CI/CD system:
- `TWINE_USERNAME`: Set to `__token__`
- `TWINE_PASSWORD`: Your PyPI API token
- `TWINE_REPOSITORY_URL`: https://test.pypi.org/legacy/ (for test)

Example GitHub Actions workflow:

```yaml
- name: Build and publish
  env:
    TWINE_USERNAME: __token__
    TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
  run: |
    python -m build
    python -m twine upload dist/*
```

## Best Practices

1. **Always test on test PyPI first** before publishing to production
2. **Use API tokens** instead of passwords for security
3. **Semantic versioning**: Follow MAJOR.MINOR.PATCH convention
4. **Check package**: Run `twine check dist/*` before uploading
5. **Clean builds**: Remove old builds before creating new ones
6. **Document changes**: Update CHANGELOG for each release
7. **Test installation**: Verify package installs correctly after publishing