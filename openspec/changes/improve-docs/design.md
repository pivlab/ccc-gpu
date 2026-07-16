# Design — improve-docs

## Context

Docs stack: Sphinx + napoleon (Google-style docstrings) on ReadTheDocs; zero autodoc directives today, so docstring quality is invisible; three API-doc formats coexist (Google docstrings, README markdown bullets, hand-written RST prose in usage.rst). The GPU implementation returns float32 while CPU returns float64 — worth stating where results are documented.

## Goals / Non-Goals

**Goals:**
- A user can fully understand and correctly use `pvalue_n_perms` from any of: docstring, docs site, README.
- Every version/license/citation/requirement claim is consistent and single-sourced where possible.
- Docstrings render on the site (autodoc), so improving them pays off once.

**Non-Goals:**
- Rewriting tutorials/notebooks in `analysis/99-tutorials/` beyond fixing wrong claims.
- New theme/redesign of the docs site (fix the mismatch, don't redesign).
- Documenting internals of the optimization backlog (that track self-documents).

## Decisions

1. **Docstring style: Google (napoleon)** — already the dominant style and configured; README examples reference the rendered API page instead of duplicating full arg tables (shrinks drift surface).
2. **Version single-sourcing**: pyproject `[project].version` is authoritative; `ccc.__version__` uses `importlib.metadata.version("cccgpu")` with a try/except fallback; Sphinx `conf.py` reads the same. CITATION.cff is updated by hand at release (documented in publishing docs).
3. **License**: standardize on the LICENSE file (BSD-2-Clause-Patent) across CITATION.cff, README badge/text, and pyproject classifiers — task includes an explicit maintainer confirmation step since CITATION.cff currently claims MIT.
4. **P-value docs live in three layers, one source of depth**: docstring = complete reference; docs-site section = method explanation + cost guidance + example; README = short example linking to the site. The docs-site section also notes the GPU p-value fixes from `fix-cuda-correctness` (values may differ from pre-fix releases).
5. **API reference via autodoc**: new `docs/source/api.rst` with `autofunction` for `ccc.coef.impl_gpu.ccc` and `ccc.coef.impl.ccc` (mock imports already configured for RTD). Theme: switch `conf.py` to `sphinx_rtd_theme` (it's what requirements installs; haiku is a config fossil).
6. **Kernel doc convention**: every `__global__` kernel gets a header comment stating its launch contract — grid mapping, required blockDim, dynamic shared-memory size formula, and output/NaN conventions. Wrong comments are corrected in place.

## Risks / Trade-offs

- [License change in CITATION.cff might contradict an intentional MIT decision] → explicit confirmation task; the LICENSE file is treated as ground truth absent contrary instruction.
- [autodoc on RTD can fail on GPU imports] → `autodoc_mock_imports` already covers this; verify with a local `READTHEDOCS=True` build in CI-less fashion.
- [Docs drift recurring] → version single-sourcing + README linking (not duplicating) reduce the standing drift surface; consistency items get a release-checklist entry in publishing docs.

## Migration Plan

Single PR after fix-cuda-correctness and cleanup land. Verify: local `make html` clean (no warnings on touched pages), rendered API page shows corrected docstrings, `grep`-audit for the old wrong claims (0.2.0, MIT, 12.0+, test PyPI, cm_vlaues).

## Open Questions

- License intent (BSD-2-Clause-Patent assumed) — confirm with maintainer during implementation.
