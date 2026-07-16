# Design — restructure-tests

## Context

Test layout today: mature upstream CPU suite (`tests/test_*.py`), GPU suite (`tests/gpu/`) that hard-imports cupy at collection, source-only CUDA gtests (never built: root CMake hook commented out), and benchmark code entangled in parity tests writing `.log` files via a cwd-relative path. `tests/__init__.py` exists while `tests/gpu/` has none — GPU tests rely on pytest's sys.path insertion to import `utils`.

## Goals / Non-Goals

**Goals:**
- `pytest tests/` passes on a CPU-only machine (GPU tests visibly skipped); `pytest -m gpu` runs the GPU suite on a GPU box; ctest runs compiled CUDA tests when enabled.
- Parity tests are fast (< ~2 min GPU suite) and deterministic.

**Non-Goals:**
- New benchmark tooling (extract-benchmarks track) — this track only removes benchmark code from tests.
- Achieving exhaustive GPU/CPU coverage parity in one pass — the named gaps are closed; the rest is backlog.

## Decisions

1. **Marker-based auto-skip via conftest, not import guards in every file.** `tests/gpu/conftest.py` applies `pytest.mark.gpu` to the whole directory (`collect_ignore`/`pytest_collection_modifyitems`) and skips with a clear reason when `importlib.util.find_spec("cupy")` or `ccc_cuda_ext` is missing. Rationale: zero per-file boilerplate, correct behavior for bare `pytest`.
2. **Package the test helpers.** Move `tests/gpu/utils.py` content into fixtures in `tests/gpu/conftest.py`; delete the copy-pasted logging/timing block entirely (its measurement role moves to the benchmark CLI). Keep import style consistent (no reliance on sys.path insertion).
3. **Tolerance contract.** End-to-end GPU-vs-CPU parity: `atol=1e-6, rtol=1e-6` (already proven achievable). Kernel-level ARI vs sklearn: tighten from `1e-2` to `1e-6` unless a documented fp32 reason exists per case. The p-value statistical comparison keeps its adaptive thresholds but gains fixed seeds and a comment explaining why exact equality is impossible (different RNGs).
4. **CUDA gtests behind `CCC_BUILD_TESTS` (default OFF)** with `enable_testing()` + `gtest_discover_tests`; googletest FetchContent moves inside the option. Broken stubs were deleted in cleanup; `test_ari.cu`/`test_ari_random.cu`/`test_coef.cu` are compiled and fixed; the "large inputs, wrong results" disabled cases in `test_ari_random.cu` become explicit `GTEST_SKIP` with a tracking note or get fixed by the fix-cuda-correctness work (verify — they may have been symptoms).
5. **Slow/heavy CPU tests** (1M-element overflow tests) get `@pytest.mark.slow`; network tests (`test_giant.py`, Titanic download) get `@pytest.mark.network`. Default local run = everything; CI = `-m "not gpu and not slow and not network"`.

## Risks / Trade-offs

- [Shrinking parity grids reduces incidental large-input coverage] → the benchmark CLI runs the large grids on demand; add one `slow`+`gpu` marked large-input parity test as a safety net.
- [Categorical `return_parts` may be an actual GPU bug, not a test gap] → timebox a reproduction; if real, file into fix backlog and mark `xfail(strict=True)` with issue link rather than leaving it invisible.
- [gtest fixes could balloon] → scope is compile + pass or explicit skip with tracking; deep kernel-test authoring is future work.

## Migration Plan

Land as one PR after cleanup-dead-code: markers/conftest first (mechanical), then per-file test edits, then CMake/ctest. Flip the CI test job to marker expression in the same PR (workflow file already exists).

## Open Questions

- None.
