# Tasks — improve-docs

## 1. P-value documentation (P0)

- [ ] 1.1 Rewrite `pvalue_n_perms` arg + Returns docs in `impl_gpu.py` and `impl.py` (method, estimator, one-sided interpretation, min resolvable p, shapes, >0 threshold rule); fix `cm_vlaues`/`pvalue_n_permutations`/`returns_parts` typos
- [ ] 1.2 Add "Computing p-values" section to `docs/source/usage.rst` (or new page): method explanation, cost warning for 2D inputs, note that GPU p-values changed post fix-cuda-correctness
- [ ] 1.3 Expand README p-value bullets; link to the docs section

## 2. Fix wrong docs

- [ ] 2.1 Fix `ccc()` return-type annotations (polymorphic) in both impls; fix `compute_coef` "seven elements" count; fix misc typos (`maximimized`, `coordiates`)
- [ ] 2.2 Fix docs-site citation (`introduction.rst`): correct authors, Bioinformatics 2026 btag068 DOI; add Marc Subirana-Granés to `conf.py` authors/copyright

## 3. Metadata single-sourcing

- [ ] 3.1 `ccc.__version__` via `importlib.metadata`; Sphinx version from package metadata; fix CITATION.cff version; remove stale "change setup.py" comment
- [ ] 3.2 **Confirm license intent with maintainer**, then align CITATION.cff, README badge/text, and pyproject classifier to LICENSE (BSD-2-Clause-Patent)
- [ ] 3.3 Reconcile install instructions (real PyPI everywhere), CUDA floor, compute-capability claim (match upgrade-dependencies arch list), Python range claim

## 4. Docs site

- [ ] 4.1 Add `api.rst` with autodoc for CPU/GPU `ccc()`; add to toctree; verify mocked build (`READTHEDOCS=True make html`) renders it
- [ ] 4.2 Switch theme to `sphinx_rtd_theme` in `conf.py`; fill or drop `development/bindings.rst` stub; merge `docs/PUBLISHING.md` into `development/package_publishing.rst`
- [ ] 4.3 Add release checklist (version/citation touchpoints) to publishing docs

## 5. CUDA inline docs

- [ ] 5.1 Add `compute_coef` doc comment (return structure, NaN convention, clamp semantics, n_partitions ≤ 255) in coef.cu/coef.cuh with matching param names
- [ ] 5.2 Add launch-contract comments to each kernel (grid mapping, blockDim requirements, shared-mem formula)
- [ ] 5.3 Fix wrong/rotten comments: `metrics.cu:243` warp claim, "Now only" relic + unused-param doc, bounds-check-theater comments in coef.cu, math.cuh/metrics.cu typos

## 6. Verification

- [ ] 6.1 Clean `make html` (no new warnings); grep-audit for retired claims (0.2.0, MIT, `cm_vlaues`, test PyPI URL, "12.0+")
