# Tasks — fix-cuda-correctness

## 1. Error handling foundation

- [ ] 1.1 Rewrite `gpu_assert`/`CUDA_CHECK`/`CUDA_CHECK_MANDATORY` in `utils.cuh` as a single `do{}while(0)`-wrapped throwing macro; remove all `exit()` calls
- [ ] 1.2 Add launch-error + sync checks after `ari_kernel`/`ari_kernel_global` (`metrics.cu:625,650`) and audit every other launch site (`coef.cu`) for coverage
- [ ] 1.3 Add shape validation in `process_input_array` (`metrics.cu:530-542`) and fix its misleading error message; validate `n_partitions > 0` in `compute_coef`; reorder the `n_aris - batch_start` underflow check (`metrics.cu:573`)
- [ ] 1.4 Python-side regression tests: invalid shape raises `ValueError`; forced launch failure raises `RuntimeError` (not crash, not NaN)

## 2. P-value path integer correctness

- [ ] 2.1 Convert `n_comparisons`, `comp_idx * n_perms` indexing, and `get_coords_from_index` call-sites to 64-bit (`coef.cu:400,406,707-708,756-761`)
- [ ] 2.2 Convert `sum_squares`/`sum_comb_c`/`sum_comb_k` to `long long` in `computePermutedARI` (`coef.cu:264,284-289`)
- [ ] 2.3 Regression test (GPU, marked slow): large-n_objects p-value vs CPU reference

## 3. Cluster capacity and null-distribution semantics

- [ ] 3.1 Replace the fixed 16×16 thread-local contingency in `computePermutedARI` with dynamic shared memory sized by k (or raise a clear error if truly capped); remove the silent-zero return (`coef.cu:225,239`)
- [ ] 3.2 Extract shared `__device__` invalid-partition helper (categorical -1 → 0.0, singleton -2 → NaN, clamp rules) used by both `ari_kernel` and the permutation kernel (`metrics.cu:466-481`, `coef.cu:365-380,148`)
- [ ] 3.3 Regression tests: k > 16 p-values; categorical/singleton feature p-values consistent with CPU permutation scheme
- [ ] 3.4 Handle `findMaxAriKernel` negative-key fallout and document/validate the `n_partitions ≤ 255` (`uint8_t max_parts`) limit (`coef.cu:151-156`)

## 4. Memory-bounded p-values

- [ ] 4.1 Batch `d_perm_ccc_values` allocation (`coef.cu:676`) with a memory-derived chunk size; loop the permutation kernels per chunk
- [ ] 4.2 Test: p-values complete on an input sized beyond the single-allocation limit (marked slow/gpu)

## 5. Docstring legend fix

- [ ] 5.1 Fix the inverted -1/-2 singleton/categorical legend in `ccc()` docstrings (`impl_gpu.py:673-675`, `impl.py:659-661`) to match `get_parts` and the code

## 6. Verification

- [ ] 6.1 Full GPU test suite passes; CPU/GPU parity tests unchanged
- [ ] 6.2 Run `compute-sanitizer --tool memcheck` on the gtest/pytest GPU subset touching changed kernels
- [ ] 6.3 Note behavior changes (p-value corrections) in changelog/release notes
