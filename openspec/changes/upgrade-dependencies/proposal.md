# Upgrade Dependencies and Build Configuration

## Why

The tested environment (numpy 1.26, numba 0.60, Python 3.10) diverges from what pip users actually resolve (unpinned numpy → 2.x) and blocks Python 3.13/3.14 support the README promises; Python 3.10 hits EOL October 2026. The build has internal contradictions: wheels compile for compute capability 7.5 only while the README requires 8.6+, three conflicting CMake floors coexist, pybind11's open `>=2.11` floor now resolves to the untested 3.x major, and `conda-lock.yml` references a source file that doesn't exist so the lock cannot be regenerated.

## What Changes

- **Python deps**: dev/lock environment to numpy 2.x + numba ≥0.61 + Python 3.12; pin pybind11 to a tested range (`>=2.13,<4` after validating against 3.x); add a `scikit-build-core` minimum; drop unnecessary `setuptools`/`wheel` from build requires; align pytest and scipy pins.
- **CUDA architectures**: build fat wheels for `75;80;86;89;90` (real SASS for Ampere/Ada/Hopper instead of PTX-JIT from 7.5) and reconcile the documented hardware floor accordingly.
- **CMake**: one consistent version story (single floor across pyproject and CMakeLists; refresh the stale `...3.26` policy ceiling).
- **CUDA toolkit**: bump build toolkit 12.5 → 12.8/12.9 (stays within the 12.x driver-compat window; enables newer GCC and Blackwell targets when arch list grows); fix the "CUDA 12.9" comment that installs 12.5.
- **Lock/env consolidation**: restore a real source env file for conda-lock and regenerate; fold or delete the drift-prone extra env files (`environment-benchmark.yaml` re-pins, py3.11 toolchain file); declare extras for research modules that remain in the repo (coordinated with `cleanup-dead-code` wheel scoping).
- **cibuildwheel**: verify the cp310–cp314 matrix actually builds post-upgrade (cp313/cp314 need the numba/pybind11 bumps); align the Python-range claims (README, pyproject comment, classifiers).

## Capabilities

### New Capabilities
- `build-distribution`: consistent, current, regenerable build/dependency configuration producing wheels that match documented hardware and Python support.

### Modified Capabilities

(none — no existing specs)

## Impact

- `pyproject.toml` (build-system, cibuildwheel, deps), root `CMakeLists.txt`, `conda-lock.yml` + `environment/*.yml`, README/docs requirement claims (coordinated with `improve-docs`).
- Risk: numba/numpy coupling — full CPU+GPU suite gates the lock regeneration; wheel size grows with the fat arch list.
- Should land after `restructure-tests` (a trustworthy suite is the upgrade safety net); before or with the first post-cleanup release.
