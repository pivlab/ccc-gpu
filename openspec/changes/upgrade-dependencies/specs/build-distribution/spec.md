# Spec Delta: build-distribution

## ADDED Requirements

### Requirement: Tested environment matches shipped constraints

The locked development environment SHALL use dependency versions consistent with what the published package's constraints resolve for users (numpy 2.x era, numba supporting it, a non-EOL Python), and the full test suite SHALL pass in that environment.

#### Scenario: Lock regeneration

- **WHEN** the conda-lock file is regenerated from its declared source environment file
- **THEN** the command succeeds (the source file exists) and the resulting environment passes the full CPU + GPU test suite

### Requirement: Wheel architectures match documented hardware support

Published wheels SHALL contain native SASS for each GPU architecture the documentation claims to support, and the documented compute-capability floor SHALL equal the lowest compiled architecture.

#### Scenario: Ampere user

- **WHEN** a user with a compute-capability 8.6 GPU installs the wheel
- **THEN** the extension loads native 8.6 code (no JIT-from-PTX penalty) and the docs' stated floor matches the wheel's lowest arch

### Requirement: Pinned, consistent build toolchain

Build-system requirements SHALL pin tested ranges (pybind11 with an upper major cap, a scikit-build-core minimum, one CMake floor consistent across pyproject and CMakeLists) and SHALL NOT include unused requirements.

#### Scenario: Isolated build reproducibility

- **WHEN** `pip install .` runs in build isolation at two different times
- **THEN** resolved build tools stay within the tested major versions and the CMake floor is the same one CMakeLists enforces

### Requirement: Declared Python support is built and tested

Every Python version claimed in README/classifiers SHALL have a wheel built by the cibuildwheel matrix and pass the CPU test suite.

#### Scenario: Claim audit

- **WHEN** the support matrix builds
- **THEN** the set of built wheels equals the documented Python range
