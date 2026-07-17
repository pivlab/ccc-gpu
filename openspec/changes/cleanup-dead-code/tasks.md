# Tasks — cleanup-dead-code

## 1. Pre-deletion verification

- [x] 1.1 Grep sweep over `analysis/`, `scripts/`, `docs/`, `tests/` for every symbol/module slated for deletion; record hits and resolve (keep or update caller)

## 2. Dead Python

- [x] 2.1 Delete unreachable compute stack + dead imports + commented example/debug blocks in `impl_gpu.py`; delete `sklearn/metrics_gpu.py` and empty `numpy/` subpackage
- [x] 2.2 Delete stray `libs/ccc/pyproject.toml` scaffold (coordinate: ruff config already moved to root in add-lint-tooling, or move it now if this lands first)
- [x] 2.3 Run full CPU + GPU test suites

## 3. Dead CUDA/C++

- [x] 3.1 Delete `cub.cu`, `example_return_optional_vectors` (coef.cu/coef.cuh/binder.cu + its test), `ari_reduced`, dead `s_mem_size` lines, unused kernel params and includes
- [x] 3.2 Delete orphaned `libs/ccc_cuda_ext/CMakeLists.txt` and dead test stubs (`test_binder.py`, `hello_test.cc`, `test_kernel.cpp`, `test_partition_pairing.cpp`)
- [x] 3.3 Document `return_parts` as Python-side-only in the binder docstring
- [x] 3.4 Rebuild extension; run GPU suite

## 4. Test-tree and repo artifacts

- [x] 4.1 Delete committed `.log` files, nested `tests/gpu/tests/` dir, `tests/gpu/test_cpu_behavior.py`, `misc_memory_consumption_calculation.py`, `tests/gpu/excluded/`, unused `tests/data/ccc-example-*.pkl`; fix `tests/data/README.md` naming
- [x] 4.2 Add `.gitignore` entries (`*.log`, `tests/**/logs/`, build dirs, `compile_commands.json`)

## 5. Docker / environment leftovers

- [x] 5.1 Replace Dockerfile with minimal CUDA-runtime + conda-lock image; delete `entrypoint.sh`, upstream `environment/README.md`, `environment/scripts/`
- [~] 5.2 Verify `docker build` succeeds and README docker instructions match — Dockerfile made correct-by-inspection (references only existing files: `conda-lock.yml` + repo root); `docker build` NOT runnable in this environment (no docker daemon). README has no docker section to reconcile.

## 6. Wheel scoping

- [x] 6.1 Narrow wheel contents via `[tool.scikit-build.wheel]` excludes (plots/methods/giant/corr excluded; `log.py`+`log_config.yaml` KEPT — required by `ccc.utils.curl`+`test_log.py` — with `pyyaml` added to declared deps so `ccc.log` imports cleanly)
- [x] 6.2 Build wheel; import-walk every shipped module (scriptable check) — all 15 shipped modules import cleanly
- [x] 6.3 Add BREAKING note to changelog for removed wheel modules
