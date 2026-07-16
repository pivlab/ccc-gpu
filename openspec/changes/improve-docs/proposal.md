# Improve Documentation

## Why

The p-value argument (`pvalue_n_perms`) — the flagship user-facing gap — is documented by one vague docstring line with typos (`cm_vlaues`, nonexistent `pvalue_n_permutations`), and no doc explains that it's a one-sided permutation test or warns about its per-pair cost. Beyond that: four-way version disagreement (pyproject 0.2.4 / `__version__` 0.2.2 / docs 0.2.0 / CITATION v1.0.0), a license conflict (CITATION.cff says MIT; LICENSE is BSD-2-Clause-Patent), the Sphinx site never renders any docstring (autodoc configured but unused), the docs-site citation has wrong author names and a stale bioRxiv DOI, and install/CUDA-requirement claims contradict each other across README and docs.

## What Changes

- **P-value documentation (P0)**: rewrite the `pvalue_n_perms` docstring sections in both `impl_gpu.py` and `impl.py` (method: permutation test; estimator `(count+1)/(n_perms+1)`; one-sided interpretation; min resolvable p ≈ `1/(n_perms+1)`; return shapes; cost warning for 2D inputs — `n*(n-1)/2` pairs × n_perms extra evaluations); fix typos; add a dedicated "Computing p-values" page/section on the docs site and expand the README section.
- **Correct wrong docs**: return-type annotations on `ccc()` (fixed 4-tuple annotated, polymorphic returned); stale docs-site citation (authors, Bioinformatics 2026 btag068); `compute_coef` "seven elements" (are eight); misc typos (`maximimized`, `coordiates`, `returns_parts`).
- **Single-source the version**: one authoritative version (pyproject) read by `ccc.__version__` (importlib.metadata) and Sphinx `conf.py`; fix CITATION.cff version.
- **Resolve the license conflict**: align CITATION.cff (currently MIT) and README badge/text to the LICENSE file (BSD-2-Clause-Patent) — flagging for maintainer confirmation in the tasks.
- **Docs site**: wire up autodoc (API reference page rendering the real docstrings), fix theme mismatch (`haiku` configured vs `sphinx-rtd-theme` installed), fill or remove the `bindings.rst` "TBD" stub, de-duplicate `docs/PUBLISHING.md` vs `development/package_publishing.rst`, add Marc Subirana-Granés to `conf.py` authors.
- **Reconcile claims**: install instructions (README real PyPI vs docs test PyPI), CUDA floor (12.0 badge vs 12.5 text), compute capability (8.6 vs "75+" — align with the arch decision in `upgrade-dependencies`), Python version range.
- **CUDA inline docs**: add the missing doc comment on `compute_coef` (return tuple structure, NaN convention, clamp semantics, `n_partitions ≤ 255` limit); document each kernel's launch contract (grid/block shape, dynamic shared-memory formula); fix wrong/rotten comments (serial code claimed parallel at `metrics.cu:243`, "Now only" relic, bounds-check-theater comments).

## Capabilities

### New Capabilities
- `user-documentation`: accurate, consistent user-facing docs with a rendered API reference and complete p-value documentation.
- `code-documentation`: inline docs (docstrings, kernel doc comments) that match the code's actual behavior and contracts.

### Modified Capabilities

(none — no existing specs)

## Impact

- `libs/ccc/coef/impl_gpu.py`, `impl.py` (docstrings/annotations), `libs/ccc_cuda_ext/*.cu*` (comments), `README.md`, `docs/source/**`, `docs/requirements.txt`, `CITATION.cff`, `libs/ccc/__init__.py`, `docs/source/conf.py`, `.readthedocs.yaml` if needed.
- Best after `fix-cuda-correctness` (documents corrected p-value behavior) and `cleanup-dead-code` (no docs for deleted code); kernel launch-contract docs ideally after the lint format lands to avoid churn.
