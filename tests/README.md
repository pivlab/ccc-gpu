# Unit tests

The suite is partitioned by pytest markers (registered in `pyproject.toml` with
`--strict-markers`):

- `gpu` — requires a CUDA device and the compiled `ccc_cuda_ext` extension.
  Auto-applied to everything under `tests/gpu/`. On a machine without
  `cupy`/`ccc_cuda_ext` these tests are skipped (not collected) so a plain
  `pytest tests/` still passes.
- `slow` — heavy CPU cases (e.g. 1M-element inputs); excluded from the default
  CI subset.
- `network` — downloads data over the network (`test_giant.py`, the Titanic
  dataset test); excluded from the default CI subset.

Correctness tests contain **no** benchmarking: no wall-clock timing, no speedup
assertions, no log files. Performance measurement lives in the `ccc-gpu-bench`
CLI (`python -m ccc.bench`).

## Run

```bash
# CPU subset — what CI runs (no GPU, no network, no slow cases):
pytest tests/ -m "not gpu and not slow and not network"

# GPU tests (needs a CUDA device + the built extension):
pytest tests/ -m gpu

# Slow CPU tests:
pytest tests/ -m slow

# Everything (local, GPU box):
pytest tests/

# Convenience wrapper (installs the package, then runs a suite):
bash ./scripts/run_tests.sh cpu     # or: gpu | slow | cpp | all
```

Note: the research-only modules (`test_giant.py`, `test_methods.py`,
`test_plots.py`, `test_corr.py`) import undeclared heavy dependencies
(`requests`, `minepy`, `IPython`) and are excluded from the wheel; ignore them
with `--ignore=...` when running a full local suite without those deps installed.

## CUDA C++ tests (gtest / ctest)

The native CUDA tests under `tests/cuda_ext/` are built behind an opt-in CMake
option (`CCC_BUILD_TESTS`, default OFF), so the default wheel build never fetches
googletest:

```bash
cmake -S . -B build-tests -DCCC_BUILD_TESTS=ON -GNinja
cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
```
