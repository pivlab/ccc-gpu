# Design — fix-cuda-correctness

## Context

The CUDA extension has two ARI implementations: the main batched kernel path (`metrics.cu`, well-tested, correct semantics) and a separate, thread-serial permutation path for p-values (`coef.cu:166-420`) that regressed on integer widths, cluster capacity, and invalid-partition semantics. Error handling is split across three styles: manual `cudaGetLastError` + throw (`coef.cu:622-633`), `CUDA_CHECK_MANDATORY` → `exit()` (`utils.cuh:53,74-84`), and nothing at all (`metrics.cu:625,650`).

## Goals / Non-Goals

**Goals:**
- Every kernel launch is checked; every CUDA failure raises a Python exception with a useful message.
- P-values are numerically correct for any input the main coefficient path can handle.
- Regression tests pin each fixed behavior.

**Non-Goals:**
- Performance work on the p-value path (per-comparison launch/sync loop, thread-serial kernel) — that is the `optimization-backlog` track. Fixes here must not preclude it, but batching for *memory-boundedness* is in scope because unbounded allocation is a correctness failure (OOM crash).
- Kernel deduplication / restructuring (`add-lint-tooling` + later refactor tracks).

## Decisions

1. **Single error-handling macro, throwing.** Replace `gpu_assert`'s `exit()` with `throw std::runtime_error` (pybind11 auto-translates). Introduce one `CUDA_CHECK(expr)` macro (properly `do { } while(0)`-wrapped) used after every launch: check `cudaGetLastError()` immediately and `cudaDeviceSynchronize()` where results are consumed. Rationale: `exit()` in a library is never acceptable; two macros with different failure modes invite misuse.
2. **64-bit indexing at every layer boundary.** All comparison/permutation counts use `uint64_t`/`size_t` end-to-end; kernels receive 64-bit params. Cheapest fix for C2; consistent with `n_feature_comp` already being `uint64_t` at `coef.cu:522`.
3. **Reuse main-kernel semantics for the null distribution.** Extract the invalid-partition rules (`-1` → 0.0, `-2` → NaN, clamp-to-zero on reduction) into a shared `__device__` helper used by both `ari_kernel` and `computePermutedARI`, rather than re-encoding rules twice. This is a minimal extraction, not the full dedup.
4. **Remove `MAX_CLUSTERS` cap via dynamic shared memory.** The permutation kernel builds its contingency matrix in shared memory sized `k*k` at launch (as the main kernel does) instead of a fixed 16×16 thread-local array. If `k*k` exceeds shared-memory limits, raise a clear error (same guard the main path uses via `check_shared_memory_size`).
5. **Batch p-value permutation storage.** Cap `d_perm_ccc_values` to a fixed budget (reuse the existing `batch_n_features` batching notion) and loop. Simple chunking is enough; overlap/streams are out of scope.
6. **Shape validation at the pybind boundary.** `process_input_array` checks `buffer.shape == {n_features, n_parts, n_objs}` and raises `py::value_error` on mismatch.

## Risks / Trade-offs

- [P-values change for affected datasets] → Document in CHANGELOG/release notes: prior values were biased/wrong; add a note to the p-value docs (improve-docs track cross-references this).
- [Throwing from code previously calling `exit()` may surface latent error paths] → New tests exercise a forced failure (e.g., invalid shape) and assert a Python exception, not a crash.
- [Dynamic shared memory for permutation kernel changes occupancy] → Acceptable; correctness first. Perf is re-examined in optimization track with the benchmark CLI baseline.
- [64-bit index params slightly increase register pressure] → Negligible at these kernel sizes.

## Migration Plan

Single PR (or small stack) on a feature branch; each bug fix is a separate commit with its regression test. GPU tests must pass before/after; CPU parity tests (`tests/gpu/test_ccc_gpu*.py`) act as the golden reference.

## Open Questions

- None blocking. If supporting arbitrary k in the permutation kernel proves large, fallback is an explicit error for k>16 (still strictly better than silent zeros) with full support deferred to the optimization track's kernel rewrite.
