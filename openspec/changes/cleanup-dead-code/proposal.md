# Clean Up Dead Code and Repo Hygiene

## Why

Roughly 15% of the CUDA extension surface and ~250 lines of the Python GPU module are dead code left over from the upstream-CCC fork and the GPU cut-over; the wheel ships research modules whose imports fail (undeclared deps); 29 benchmark log files are committed; and a broken upstream Dockerfile misleads users. Decision (2026-07-16): delete provably-dead code outright, keep research modules in-repo for `analysis/` but exclude them from the published wheel.

## What Changes

- **Delete dead Python**: unreachable compute stack in `impl_gpu.py` (`cdist_parts_basic/parallel`, `compute_ccc`, `compute_ccc_perms`, `compute_coef`, `get_coords_from_index`, ~lines 248–568), its now-unused imports (`DummyExecutor`, `ari`, `unravel_index_2d`), dead local `X_has_cat_features`, large commented-out example/debug blocks; delete `libs/ccc/sklearn/metrics_gpu.py` (unfinished cupy prototype ending in `NotImplementedError`) and the empty `libs/ccc/numpy/` subpackage.
- **Delete dead CUDA/C++**: `cub.cu` (tutorial demo), `example_return_optional_vectors` (+ binding + header decl + its test), unimplemented `ari_reduced`, dead `s_mem_size` computation (`metrics.cu:599-601`), unused kernel params/includes; delete the orphaned `libs/ccc_cuda_ext/CMakeLists.txt` and broken test stubs (`test_binder.py` importing nonexistent `cuda_ccc`, `hello_test.cc`, empty `test_kernel.cpp`, non-compiling `test_partition_pairing.cpp`). Decide `return_parts` in the C++ signature: keep-and-document as Python-handled, or remove the dead parameter.
- **Delete the scaffold** `libs/ccc/pyproject.toml` (placeholder metadata; ruff config moves to root in `add-lint-tooling`).
- **Repo hygiene**: remove 29 committed `.log` files (`tests/gpu/tests/logs/`, `tests/logs/`), the stray nested `tests/gpu/tests/` dir, print-only pseudo-tests (`tests/gpu/test_cpu_behavior.py` with its hardcoded dead path, `misc_memory_consumption_calculation.py`), unused `tests/data/ccc-example-*.pkl`; add `.gitignore` entries.
- **Fix or remove upstream leftovers**: Dockerfile (copies nonexistent `environment/environment.yml`, non-CUDA base) + `entrypoint.sh` + `environment/README.md` + `environment/scripts/` (python 3.9 / R stack) — replace with a minimal CUDA-based Dockerfile or delete with a README note.
- **Slim the wheel**: narrow `wheel.packages`/`wheel.exclude` so the published package ships only supported modules (`coef/`, `utils/`, `scipy/`, `pytorch/` as needed by imports); research modules (`plots.py`, `methods.py`, `giant.py`, `corr.py`, `log.py`) stay in-repo for `analysis/` but out of the wheel — **BREAKING** for anyone importing them from the PyPI package.

## Capabilities

### New Capabilities
- `wheel-packaging`: the published wheel contains only supported modules, and every shipped module imports cleanly with declared dependencies.
- `repo-hygiene`: no dead code, generated artifacts, or broken build/docker files in the tree.

### Modified Capabilities

(none — no existing specs)

## Impact

- `libs/ccc/**`, `libs/ccc_cuda_ext/**`, root `pyproject.toml` (wheel config), `Dockerfile`, `entrypoint.sh`, `environment/`, `tests/` (file deletions only — behavior-affecting test restructuring is the `restructure-tests` track), `.gitignore`.
- Depends on `fix-cuda-correctness` (don't delete around code being fixed); should land before `add-lint-tooling`'s mass format (less to reformat) — final ordering: fix → cleanup → lint.
- Verify with grep + full test suite that nothing in `analysis/` or `tests/` imports deleted symbols.
