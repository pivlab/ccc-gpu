# Design — add-lint-tooling

## Context

Current state: `.pre-commit-config.yaml` runs ruff (v0.8.5, stale pin) + ruff-format + black + black-jupyter; ruff rules live only in `libs/ccc/pyproject.toml` (a placeholder scaffold not referenced by the build), so files under `libs/ccc/` and everything else resolve *different* lint configs. No C++ tooling. No CI (`.github/` absent).

## Goals / Non-Goals

**Goals:**
- One lint/format config per language, at the repo root, enforced locally (pre-commit) and remotely (CI).
- CUDA/C++ static analysis that catches the bug classes actually found in review (integer narrowing, unchecked returns).

**Non-Goals:**
- mypy/type-checking gate — numba-heavy code fights static typing; existing hints are partly wrong. Revisit after cleanup.
- Fixing all clang-tidy findings immediately — the tool lands with a curated check list; fixes beyond formatting are follow-up work.
- CI GPU runners (user decision: GPU tests stay local).

## Decisions

1. **Ruff only (lint + format), drop black.** They overlap; ruff-format is a black-compatible superset for this need. Migrate the rule set from the scaffold file (B, I, ARG, C4, EM, PL, PT, PTH, RET, SIM, UP, NPY, PD) but **drop `isort.required-imports = ["from __future__ import annotations"]`** initially to avoid a semantic-adjacent mass edit; per-file-ignores for `analysis/` notebooks/scripts (looser) and `tests/` (`PLR2004` magic values etc.).
2. **`.clang-format` based on `Microsoft` style** (Allman default) with `IndentWidth: 4`, `ColumnLimit: 120`, `PointerAlignment: Right`, `SortIncludes: CaseSensitive` — matches the majority style so the mechanical diff is minimized.
3. **`.clang-tidy` with a curated check list**: `clang-analyzer-*, bugprone-*, performance-*, modernize-*, readability-braces-around-statements, readability-else-after-return, cppcoreguidelines-init-variables, misc-unused-parameters`, minus `modernize-use-trailing-return-type`, `modernize-avoid-c-arrays`, `bugprone-easily-swappable-parameters`. Run via `CMAKE_EXPORT_COMPILE_COMMANDS=ON` with `--extra-arg=--cuda-host-only`. CI runs it non-blocking (report-only) at first; promoted to blocking once the baseline is clean.
4. **Formatting lands as one isolated commit** with a `.git-blame-ignore-revs` entry so blame stays useful.
5. **CI = three jobs in one workflow**: `lint` (ruff check/format-check, clang-format check via `pre-commit run --all-files` for consistency), `test-cpu` (pytest with `-m "not gpu and not slow and not network"` — enabled once restructure-tests lands markers; until then it may run the top-level CPU tests with an explicit ignore of `tests/gpu`), `build` (build the wheel with `CMAKE_ARGS` for a no-GPU compile check — nvcc compiles without a device present; use the cibuildwheel container image or install cuda-toolkit via apt in the runner).

## Risks / Trade-offs

- [Mass formatting commit churns history] → single commit + `.git-blame-ignore-revs`; land immediately after fix-cuda-correctness merges.
- [clang-tidy on CUDA is imperfect (host-only pass)] → accepted; it still covers host orchestration code where most review findings live. Device-only diagnostics come from `compute-sanitizer` in the test track.
- [Wheel-build CI job needs the CUDA toolkit (large download)] → cache the toolkit layer / use a container image; a compile-only smoke is still far better than nothing.

## Migration Plan

1. Land configs + pre-commit changes. 2. Run `pre-commit run --all-files`; commit as formatting-only. 3. Enable lint + build CI jobs. 4. Enable test-cpu job (after restructure-tests). Rollback: configs are additive; reverting the workflow file disables CI.

## Open Questions

- None.
