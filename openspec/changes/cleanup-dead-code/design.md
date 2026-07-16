# Design — cleanup-dead-code

## Context

The repo is a fork of upstream CCC (greenelab) plus a GPU port. The port left the CPU compute stack duplicated inside `impl_gpu.py` (now unreachable from `ccc()`), an abandoned cupy prototype, demo/pybind example code in the production extension, an orphaned second CMake project, and upstream Docker/environment files that reference files that don't exist here. Review agents verified reachability by tracing imports/call graphs.

## Goals / Non-Goals

**Goals:**
- Every line in `libs/` is reachable from a supported entry point or a test.
- `pip install cccgpu` yields a package where every importable module imports successfully.
- The tree contains no generated artifacts or broken top-level files.

**Non-Goals:**
- Refactoring live code (kernel dedup, function splitting) — later tracks.
- Deleting `analysis/` research modules from the repo (kept per user decision; only excluded from the wheel).
- Test restructuring beyond deleting dead files (`restructure-tests` track).

## Decisions

1. **Delete, don't comment out or attic.** Git history is the attic. Each deletion category is one commit for easy revert.
2. **Wheel scoping via explicit exclude list** in `[tool.scikit-build.wheel]`: exclude `plots.py`, `methods.py`, `giant.py`, `corr.py` and anything else importing undeclared deps, rather than restructuring the package layout (a `src/` re-layout is more churn than value right now). `log.py`+`log_config.yaml`: keep in wheel only if `coef` path uses it — review says it doesn't; exclude and keep `utils`' existing defensive try/except.
3. **`return_parts` in the C++ API**: keep the parameter (the Python layer composes parts output), but document it as unused in the extension and stop advertising it in the binder docstring — removing it would churn the binding signature for no user benefit. Revisit in a future ABI-breaking release.
4. **Dockerfile: replace, minimal.** A ~20-line `nvidia/cuda:12.x-runtime-ubuntu22.04` + conda-lock Dockerfile preserving the documented install path; delete `entrypoint.sh`, upstream `environment/README.md` and `environment/scripts/`. If maintaining Docker isn't wanted, fallback is delete + README note — but a working minimal image is low-cost and the current one actively misleads.
5. **Order within the initiative**: after `fix-cuda-correctness`, before `add-lint-tooling`'s format-all commit.

## Risks / Trade-offs

- [Someone imports research modules from the wheel today] → BREAKING note in changelog; modules remain available from a source checkout; version bump signals it.
- [A "dead" symbol is used by an analysis notebook] → pre-deletion grep sweep over `analysis/`, `scripts/`, `docs/`; anything hit either stays or the notebook is updated in the same PR.
- [Deleting `tests/gpu/excluded/` loses reference material for future GPU subroutine work] → it's fully commented-out code preserved in git history; delete.

## Migration Plan

One PR, commits grouped: (1) dead Python, (2) dead CUDA/C++ + CMake orphan, (3) test-tree artifacts + gitignore, (4) Docker/environment leftovers, (5) wheel scoping. Full CPU+GPU suite after each group.

## Open Questions

- None.
