# Fix CUDA Correctness Bugs

## Why

A deep review (2026-07-16) found real correctness bugs in the CUDA extension — concentrated in the p-value permutation path — that produce silently wrong results (all-NaN outputs, ARI=0 for >16 clusters, biased p-values) or kill the Python interpreter outright. These must be fixed **before** the planned refactoring tracks so that refactors can be verified against already-correct behavior.

## What Changes

- Add launch + sync error checks after every kernel launch; convert `exit()`-based error handling (`gpu_assert` in `utils.cuh`) to thrown `std::runtime_error` so pybind11 raises Python exceptions instead of killing the interpreter.
- Fix 32-bit integer overflow in the p-value path: `comp_idx * n_perms` indexing (`coef.cu:406`), `uint32_t n_comparisons` truncation of a `uint64_t` (`coef.cu:400`, call site `coef.cu:756-761`), and 32-bit `get_coords_from_index` truncation (`coef.cu:707-708`).
- Fix `int` sums in `computePermutedARI` that overflow with ~46k+ objects per cluster (`coef.cu:264,284-289`) — use `long long` as the main kernel already does.
- Remove the `MAX_CLUSTERS=16` silent-zero cliff in `computePermutedARI` (`coef.cu:225,239`): either support arbitrary k (matching the main kernel) or raise a clear error.
- Make the permutation null distribution use the **same invalid-partition semantics** as the observed statistic (`-1` → ARI 0.0, `-2` → NaN, clamping rules) so p-values are unbiased (`coef.cu:365-368,376-380` vs `metrics.cu:466-481`, `coef.cu:148`).
- Batch the p-value path's `d_perm_ccc_values` allocation (`coef.cu:676`) so large inputs don't OOM where the main coefficient path would succeed.
- Validate input shape in `process_input_array` against the scalar dims (`metrics.cu:530-542`); guard the `n_aris - batch_start` unsigned underflow ordering (`metrics.cu:573-574`); handle the negative-key/`uint8_t` truncation fallout in `findMaxAriKernel` (`coef.cu:151-156`).
- Fix the inverted `-1`/`-2` singleton/categorical legend in the `ccc()` docstrings (`impl_gpu.py:659-661`/`673-675`, `impl.py:659-661`) — docstring bug with correctness-level impact for consumers of `max_parts`.

## Capabilities

### New Capabilities
- `cuda-error-reporting`: CUDA errors surface as Python exceptions; no silent-NaN results; no process exit from library code.
- `pvalue-computation`: permutation-test p-values are correct at scale (64-bit indexing, no cluster-count cliff, null distribution consistent with the observed statistic, memory-bounded batching).

### Modified Capabilities

(none — no existing specs)

## Impact

- `libs/ccc_cuda_ext/coef.cu`, `metrics.cu`, `utils.cuh` (error handling, p-value kernels, bounds checks).
- `libs/ccc/coef/impl_gpu.py`, `impl.py` (docstring legend fix only).
- Behavior change: previously-silent failures now raise; p-values change for datasets with categorical/singleton partitions or >16 clusters (they were wrong before).
- Tests: new regression tests for each bug class (GPU-marked; large-scale overflow cases may need a big-memory GPU and should be marked `slow`).
