# Tasks — fix-cuda-correctness

## 1. Error handling foundation

- [x] 1.1 Rewrite `gpu_assert`/`CUDA_CHECK`/`CUDA_CHECK_MANDATORY` in `utils.cuh` as a single `do{}while(0)`-wrapped throwing macro; remove all `exit()` calls
- [x] 1.2 Add launch-error + sync checks after `ari_kernel`/`ari_kernel_global` (`metrics.cu:625,650`) and audit every other launch site (`coef.cu`) for coverage (added `CUDA_CHECK_KERNEL`)
- [x] 1.3 Add shape validation in `process_input_array` (`metrics.cu:530-542`) and fix its misleading error message; validate `n_partitions > 0` in `compute_coef`; reorder the `n_aris - batch_start` underflow check (`metrics.cu:573`)
- [x] 1.4 Python-side regression tests: invalid shape raises `ValueError`; interpreter survives (forced CUDA-runtime `RuntimeError` covered by inspection — see HANDOFF)

## 2. P-value path integer correctness

- [x] 2.1 Convert `n_comparisons`, `comp_idx * n_perms` indexing, and `get_coords_from_index` call-sites to 64-bit (`coef.cu`; added 64-bit `get_coords_from_index` overload in `math.cuh`)
- [x] 2.2 Convert `sum_squares`/`sum_comb_c`/`sum_comb_k` to `long long` in `computePermutedARI`
- [x] 2.3 Regression test (GPU): large-cluster p-value vs CPU reference (true >2^32 indexing covered by construction — see HANDOFF)

## 3. Cluster capacity and null-distribution semantics

- [x] 3.1 Replace the fixed 16×16 thread-local contingency in `computePermutedARI` with per-thread global-memory scratch sized by k; removed the silent-zero return
- [x] 3.2 Extract shared `__device__` invalid-partition helper (`classify_partition_pair` in `metrics.cuh`: categorical -1 → 0.0, singleton -2 → NaN, clamp rules) used by both `ari_kernel`/`ari_kernel_global` and the permutation kernel
- [x] 3.3 Regression tests: k > 16 p-values; categorical/singleton feature p-values consistent with CPU permutation scheme
- [x] 3.4 Handle `findMaxAriKernel` negative-key fallout and validate the `n_partitions ≤ 255` (`uint8_t max_parts`) limit

## 4. Memory-bounded p-values

- [x] 4.1 Batch `d_perm_ccc_values` allocation with a memory-derived chunk size; loop the permutation kernels per chunk
- [x] 4.2 Batching is exercised by the existing/new pvalue tests (single-chunk); large multi-chunk case covered by construction — see HANDOFF

## 5. Docstring legend fix

- [x] 5.1 Fix the inverted -1/-2 singleton/categorical legend in `ccc()` docstrings (`impl_gpu.py`, `impl.py`) to match `get_parts` and the code

## 6. Verification

- [x] 6.1 Full focused test suite passes (218 passed; CPU/GPU parity tests unchanged)
- [x] 6.2 Ran `compute-sanitizer --tool memcheck` on the changed p-value kernels (0 errors)
- [x] 6.3 Behavior changes (p-value corrections) noted in HANDOFF (no CHANGELOG file exists; docs track owns release notes)
