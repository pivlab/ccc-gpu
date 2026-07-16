# Tasks — restructure-tests

## 1. Markers and conftest

- [ ] 1.1 Register `gpu`, `slow`, `network` markers + `--strict-markers` in `pyproject.toml`
- [ ] 1.2 Add `tests/gpu/conftest.py`: auto-mark dir as `gpu`, skip cleanly when cupy/ccc_cuda_ext missing; add root `tests/conftest.py` for shared fixtures
- [ ] 1.3 Mark 1M-element CPU tests `slow`; mark `test_giant.py` and Titanic tests `network`
- [ ] 1.4 Verify: `pytest tests/` green on CPU-only env; `pytest -m gpu` green on GPU box

## 2. De-benchmark the parity tests

- [ ] 2.1 Delete the duplicated timing/logging blocks from `tests/gpu/test_ccc_gpu.py` and `test_ccc_gpu_return_parts.py`; keep pure correctness assertions
- [ ] 2.2 Remove wall-clock speedup assertions from `tests/test_coef.py` (4 tests) and `tests/test_coef_pval.py:247` (keep any residual correctness value; note grids for the benchmark CLI)
- [ ] 2.3 Remove commented-out mega parameter grids; add one `slow`+`gpu` large-input parity safety-net test
- [ ] 2.4 Delete `test_ari_gpu.py::test_pairwise_ari_benchmark_features` (superseded by benchmark CLI)

## 3. Fixtures and tolerances

- [ ] 3.1 Convert `tests/gpu/utils.py` helpers (incl. `clean_gpu_memory`) into conftest fixtures; dedupe shared ARI reference cases and data generators
- [ ] 3.2 Standardize parity tolerances to 1e-6; tighten kernel-level ARI tolerances or justify inline; fix seeds + explanatory comment in the p-value statistical tests
- [ ] 3.3 Delete dead diagnostic code (e.g., `analyze_differences` with commented-out assertion)

## 4. CUDA gtests

- [ ] 4.1 Add `option(CCC_BUILD_TESTS OFF)`; move googletest FetchContent inside it; restore/rewrite `add_tests_from_directory` with `gtest_discover_tests`; raise ctest timeout
- [ ] 4.2 Get `tests/cuda_ext/test_ari.cu`, `test_ari_random.cu`, `test_coef.cu` compiling and passing; convert known-bad disabled cases to `GTEST_SKIP` with tracking notes (re-check against fix-cuda-correctness — they may now pass)
- [ ] 4.3 Wire `scripts/run_tests.sh cpp` to configure with the option and run ctest

## 5. Coverage gaps

- [ ] 5.1 Reproduce categorical `return_parts` on GPU; fix or `xfail(strict=True)` + issue
- [ ] 5.2 Add GPU tests: constant-feature matrix, too-few-objects error parity, mixed-type DataFrame

## 6. Docs and runner

- [ ] 6.1 Update `scripts/run_tests.sh` and `tests/README.md` to the marker-based commands; switch CI test job to `-m "not gpu and not slow and not network"`
