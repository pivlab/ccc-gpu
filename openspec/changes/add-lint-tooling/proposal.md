# Add Linting, Formatting, and CI

## Why

Tooling exists only on paper: pre-commit is configured (with ruff *and* black, redundantly) but has never been run on the tree; the only ruff config lives in a stray placeholder file (`libs/ccc/pyproject.toml`, a pybind11 example scaffold); the CUDA/C++ code has no clang-format/clang-tidy at all (57 trailing-whitespace lines in `coef.cu`, mixed brace and naming styles); and the repo has no CI whatsoever, so none of this is enforced.

## What Changes

- Move the ruff configuration into the **root** `pyproject.toml` as the single source of truth; tune the rule set for this codebase; delete the black hooks (keep `ruff-format`) and update hook pins.
- Add `.clang-format` (Allman, 4-space, 120 cols — matching the codebase majority) and `.clang-tidy` (bugprone/performance/modernize checks; would have caught the p-value integer-narrowing bugs) with pre-commit integration for `.cu/.cuh/.cpp/.cc` files; export `compile_commands.json` from CMake.
- Run all formatters once, landing the result as a single isolated formatting commit (no logic changes mixed in).
- Add GitHub Actions CI: (a) lint job — `ruff check`, `ruff format --check`, clang-format check; (b) CPU-only pytest job using markers from the `restructure-tests` track; (c) wheel-build smoke job. GPU tests remain a documented local command.

## Capabilities

### New Capabilities
- `code-linting`: enforced, single-source lint/format configuration covering Python and CUDA/C++.
- `continuous-integration`: automated lint, CPU test, and build checks on every push/PR.

### Modified Capabilities

(none — no existing specs)

## Impact

- Root `pyproject.toml`, `.pre-commit-config.yaml`, new `.clang-format`, `.clang-tidy`, `.github/workflows/ci.yml`, root `CMakeLists.txt` (compile-commands export).
- One large mechanical formatting commit touching most files (isolated, no behavior change).
- Depends on: `fix-cuda-correctness` landing first (avoid reformatting code that's about to be fixed); the CI test job depends on `restructure-tests` markers (lint + build jobs can enable immediately).
- The stray `libs/ccc/pyproject.toml` deletion is coordinated with `cleanup-dead-code` (that track owns the deletion; this track owns where the config lands).
