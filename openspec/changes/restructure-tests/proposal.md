# Restructure the Test Suites

## Why

`pytest tests/` — the documented command — fails collection on any CPU-only machine because GPU tests import `cupy` unconditionally; there are no pytest markers, no `conftest.py`, and the `excluded/` dir is only skipped by a shell script. ~140 lines of benchmark/logging infra are copy-pasted verbatim between two GPU test files, tolerances are inconsistent (1e-6 to 1e-2), and the CUDA C++ gtests exist as source but are never compiled. Coverage gaps exist exactly where the GPU implementation diverges from CPU (categorical `return_parts`, constant features, error-path parity).

## What Changes

- Register pytest markers (`gpu`, `slow`, `network`) in `pyproject.toml` with `--strict-markers`; add a root `conftest.py` that skips GPU tests gracefully (marker-based auto-skip when `cupy`/`ccc_cuda_ext` is unavailable) so plain `pytest tests/` passes everywhere.
- Strip timing/speedup/log-file machinery out of GPU parity tests (moves to the benchmark CLI track); shrink their parameter grids to small, fast cases — tests become pure correctness checks.
- Deduplicate shared helpers into `conftest.py` fixtures (data generators, `clean_gpu_memory` as a fixture, common ARI reference cases); standardize float tolerances and document the contract (parity: `atol/rtol=1e-6`; kernel-level: justified per-case).
- Wire the CUDA gtests into CMake behind `option(CCC_BUILD_TESTS OFF)` + ctest; fix or delete the known-broken ones; gate googletest FetchContent behind the same option (no more fetching gtest on every wheel build); raise the 10s ctest timeout.
- Close GPU coverage gaps: categorical `return_parts` (currently known-broken and commented out — reproduce, fix or xfail with tracked issue), constant-feature matrices, too-few-objects error parity, mixed numerical+categorical DataFrame.
- Make `scripts/run_tests.sh` and `tests/README.md` match the new marker-based commands.

## Capabilities

### New Capabilities
- `test-suite`: marker-partitioned, environment-aware pytest suites plus compiled-and-run CUDA gtests.

### Modified Capabilities

(none — no existing specs)

## Impact

- `pyproject.toml` (markers), new `tests/conftest.py` (+ `tests/gpu/conftest.py`), all `tests/gpu/test_*.py`, four timing tests in `tests/test_coef.py` + one in `tests/test_coef_pval.py` (timing assertions removed here, covered by benchmark CLI), root `CMakeLists.txt`, `tests/cuda_ext/*.cu`, `scripts/run_tests.sh`, `tests/README.md`.
- Enables the CI `test-cpu` job (add-lint-tooling track) to switch to `-m "not gpu and not slow and not network"`.
- Depends on `cleanup-dead-code` (dead test files already removed) and feeds `extract-benchmarks` (the removed timing infra defines what the CLI must measure).
