# Tasks — cleanup-dead-code

## 1. Pre-deletion verification

- [ ] 1.1 Grep sweep over `analysis/`, `scripts/`, `docs/`, `tests/` for every symbol/module slated for deletion; record hits and resolve (keep or update caller)

## 2. Dead Python

- [ ] 2.1 Delete unreachable compute stack + dead imports + commented example/debug blocks in `impl_gpu.py`; delete `sklearn/metrics_gpu.py` and empty `numpy/` subpackage
- [ ] 2.2 Delete stray `libs/ccc/pyproject.toml` scaffold (coordinate: ruff config already moved to root in add-lint-tooling, or move it now if this lands first)
- [ ] 2.3 Run full CPU + GPU test suites

## 3. Dead CUDA/C++

- [ ] 3.1 Delete `cub.cu`, `example_return_optional_vectors` (coef.cu/coef.cuh/binder.cu + its test), `ari_reduced`, dead `s_mem_size` lines, unused kernel params and includes
- [ ] 3.2 Delete orphaned `libs/ccc_cuda_ext/CMakeLists.txt` and dead test stubs (`test_binder.py`, `hello_test.cc`, `test_kernel.cpp`, `test_partition_pairing.cpp`)
- [ ] 3.3 Document `return_parts` as Python-side-only in the binder docstring
- [ ] 3.4 Rebuild extension; run GPU suite

## 4. Test-tree and repo artifacts

- [ ] 4.1 Delete committed `.log` files, nested `tests/gpu/tests/` dir, `tests/gpu/test_cpu_behavior.py`, `misc_memory_consumption_calculation.py`, `tests/gpu/excluded/`, unused `tests/data/ccc-example-*.pkl`; fix `tests/data/README.md` naming
- [ ] 4.2 Add `.gitignore` entries (`*.log`, `tests/**/logs/`, build dirs, `compile_commands.json`)

## 5. Docker / environment leftovers

- [ ] 5.1 Replace Dockerfile with minimal CUDA-runtime + conda-lock image; delete `entrypoint.sh`, upstream `environment/README.md`, `environment/scripts/`
- [ ] 5.2 Verify `docker build` succeeds and README docker instructions match

## 6. Wheel scoping

- [ ] 6.1 Narrow wheel contents via `[tool.scikit-build.wheel]` excludes (plots/methods/giant/corr/log+yaml as decided)
- [ ] 6.2 Build wheel; install in a clean venv; import-walk every shipped module (scriptable check)
- [ ] 6.3 Add BREAKING note to changelog for removed wheel modules
