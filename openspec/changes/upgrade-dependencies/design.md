# Design — upgrade-dependencies

## Context

Two dependency realities coexist: pip metadata (unpinned numpy/scipy/numba/pandas/scikit-learn) and the conda-lock dev env (numpy 1.26.4, numba 0.60, Python 3.10.16, CUDA 12.5, pybind11 2.13.6). Code greps clean for NumPy-2-removed APIs, so the upgrade is expected to be low-friction; numba is the pacing dependency for new Python versions. The lock's declared source (`environment/environment-gpu-new.yml`) doesn't exist.

## Goals / Non-Goals

**Goals:**
- Tested env ≈ shipped constraints (no more "tested on 1.26, users get 2.x").
- Regenerable lock from a real source file; fewer env definitions.
- Wheels whose compiled arch list matches documented hardware support.

**Non-Goals:**
- CUDA 13 major bump (raises driver floor; defer until user base is ready).
- Conda-forge packaging or new distribution channels.
- mypy/typing additions (out of scope per lint-track decision).

## Decisions

1. **Upgrade order: env first, constraints second.** Regenerate the lock with numpy 2.x/numba 0.61+/py3.12, run the full suite on GPU hardware, then encode floors in pyproject (`numpy>=1.26`, keep permissive unless breakage found — the library is a consumer, not a framework).
2. **pybind11: validate 3.x, then pin `>=2.13,<4`.** If 3.x passes (build + tests, cp310–cp314), allow it; the cap guards against the next major. cp314 wheels likely require it.
3. **Arch list `75;80;86;89;90` explicit** (not `all-major`): predictable wheel size, covers Turing→Hopper natively; PTX from 90 forward-compats newer GPUs. Blackwell SASS (`100;120`) added when the toolkit bump lands and hardware to validate exists. Documented floor becomes "CC 7.5+" — resolving the README contradiction in the *permissive* direction since 7.5 SASS exists today.
4. **Toolkit 12.8/12.9 in cibuildwheel + dev env**: newest 12.x at implementation time; keeps r555+ driver compatibility story intact.
5. **Env consolidation to two files + lock**: `environment/environment-gpu.yml` (dev, source of the lock) and `docs/requirements.txt` (RTD). Benchmark/toolchain env files fold into the dev env or optional dependency groups; upstream-leftover files were deleted in cleanup.
6. **Extras** in pyproject: `[project.optional-dependencies] plots`, `research` for the in-repo research modules (installable from source checkout), `test` (pytest etc.), keeping the core install lean.

## Risks / Trade-offs

- [numba 0.61+/numpy 2.x behavior shifts in `@njit` partitioning] → CPU suite has strong coverage of partitioning; GPU parity tests double-check end numbers; lock regeneration is one commit, easy to revert.
- [Fat wheel size (5 archs)] → measure; if PyPI limits threaten, trim to `80;86;89;90` + PTX and document, or request a size-limit increase.
- [pybind11 3.x latent incompatibility] → validated explicitly; cap prevents surprise majors either way.
- [cp314 build failures late in the matrix] → cibuildwheel smoke in CI (build track) catches early; README claims updated only after wheels actually build.

## Migration Plan

1. New source env → conda-lock regenerate → local full suite (GPU box). 2. pyproject floors/caps + build-system cleanup + arch list + toolkit bump. 3. cibuildwheel matrix validation. 4. Update claims (with improve-docs). Rollback: lock and pyproject changes are independent commits.

## Open Questions

- Exact newest 12.x toolkit at implementation time — pick then (12.8 minimum).
