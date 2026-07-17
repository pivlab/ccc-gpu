# CCC-GPU Optimization Results

A first implementation pass against the ranked
[optimization backlog](source/development/optimization_backlog.rst). Two rounds were
implemented and verified; the rest were assessed and deferred with data. The headline
result is a **~70–78× speedup of the p-value permutation path** — the marquee bottleneck
identified in the backlog — with correctness preserved bit-for-bit against the CPU
reference.

## Environment & methodology

- **Hardware/stack:** NVIDIA RTX 4090 (sm_89, 24 GB), driver 580, CUDA toolkit 12.5,
  cupy 14.1, Python 3.12 (conda env `ccc-gpu-dev`).
- **Measurement:** the `ccc-gpu-bench` CLI (shipped by the `extract-benchmarks` change),
  GPU-only, `n_perms=100`, `--warmup 1`, min-of-repeats.
- **Correctness gate (every round):** `pytest -m gpu` — the GPU-vs-CPU parity suite,
  including the p-value parity tests — plus `compute-sanitizer memcheck` on the changed
  kernels. Every reported speedup was gated on **zero** correctness or memory regressions.
- **Commit range:** on branch `pr-f-optimization-backlog`, on top of the refactoring
  initiative.

## Baseline

The p-value path scaled as **O(number of feature comparisons)** — it launched the
permutation kernel once per comparison (one thread per permutation, ~100 threads, leaving
the GPU's ~16k lanes idle) with a full device synchronize after each launch.

| Features | Comparisons | GPU p-value (baseline) |
|---:|---:|---:|
| 30  | 435    | 7.64 s |
| 50  | 1,225  | 18.96 s |
| 80  | 3,160  | 46.82 s |
| 200 | 19,900 | **> 294 s** (timed out) |

For reference, the coefficient-only path was already compute-bound and much faster:
0.67 s @ 1k, 7.85 s @ 5k, 28.1 s @ 10k features.

## Round 1 — H2: parallelize permutations across comparisons

**Target.** The per-comparison launch/sync loop with an under-utilized kernel: wall time
was linear in the number of comparisons and the single biggest bottleneck in the codebase
(200 features ≈ 294 s).

**Method.** A new `computePermutationCCCBatched` kernel maps **one thread to each
(comparison, permutation) pair**, so a single launch covers `n_comps × n_perms` threads and
all comparisons run concurrently. Each thread derives its feature pair from the condensed
comparison index and selects the permuted feature from a **precomputed per-feature
valid-partition count** (matching the CPU rule: permute the feature with more valid
partitions). The host now steps over comparison sub-batches sized to a scratch budget,
collapsing the launch count from `n_feature_comp` to a handful. Invalid-partition semantics
(categorical → 0, singleton → NaN, clamp ≥ 0) are unchanged.

**Outcome — ✅ ~31–34×**, and inputs that previously timed out now complete.

| Features | Baseline | Round 1 (H2) | Speedup |
|---:|---:|---:|---:|
| 30  | 7.64 s  | 0.225 s | 34× |
| 50  | 18.96 s | 0.613 s | 31× |
| 80  | 46.82 s | 1.528 s | 31× |
| 200 | > 294 s | 9.175 s | ~32× |

The gain is GPU-utilization-driven (thousands of concurrent threads vs. ~100), so the path
still scales with the comparison count but at ~0.46 ms/comparison instead of ~14.8 ms.

## Round 2 — H2b: coalesced per-thread local contingency

**Target.** After Round 1 each thread's k×k contingency lived in global scratch at
`scratch + gid × scratch_stride`; consecutive warp lanes were `scratch_stride` ints apart,
so **every contingency access was uncoalesced** (32 separate cache lines per warp), and the
buffer cost up to ~788 MB.

**Method.** For the common small-k case (k ≤ 16, i.e. `k*k + 2k ≤ 288`), build the
contingency in a **per-thread local array**. CUDA interleaves local memory across a warp, so
the accesses become coalesced and L1-cached; the global scratch buffer collapses to a single
element and the per-launch comparison bound disappears. Larger k still uses the
global-scratch fallback. Same math; parity-guarded.

**Outcome — ✅ another ~2.3–2.4×** on top of Round 1, and ~788 MB of scratch reclaimed.

| Features | Round 1 (H2) | Round 2 (H2b) | R1→R2 | **Total vs baseline** |
|---:|---:|---:|---:|---:|
| 30  | 0.225 s | 0.099 s | 2.3× | **77×** |
| 50  | 0.613 s | 0.269 s | 2.3× | **70×** |
| 80  | 1.528 s | 0.640 s | 2.4× | **73×** |
| 200 | 9.175 s | 3.760 s | 2.4× | **~78×** |

## Combined result

The p-value permutation path is **~70–78× faster** than baseline and no longer times out on
inputs that previously did, while producing the same coefficients and p-values as before
(bit-for-bit against the CPU reference). Both changes are pure GPU-side; the public API,
outputs, and CPU path are untouched.

## Verification

For every round: `pytest -m gpu` (full GPU parity suite) — **79 passed**; the focused
p-value + cuda-correctness parity tests — **30 passed**; the k > 16 global-scratch fallback
parity test — **passed**; and `compute-sanitizer memcheck` on the new kernel —
**0 errors**.

## Assessed but deferred (not pursued — with reasons)

- **H1 / H3 — per-batch parts upload, k-reduce, and buffer re-alloc in the coefficient
  path.** *Data-driven decision to skip.* The coefficient path is compute-bound: 5k features
  (1 batch) take 7.85 s and 10k features (4 batches) take 28.1 s, which is **below** the
  pure-quadratic extrapolation (4 × 7.85 = 31.4 s). A multi-batch run therefore carries no
  per-batch penalty — the redundant upload (~tens of ms), reduce (~ms), and re-allocation are
  in the noise. Implementing them would add risk to a working path for < 1% gain.
- **A3 — memory-aware batch sizing.** A robustness item (avoid OOM on smaller GPUs / grow
  batches on larger ones), not a speedup on the reference hardware. Deferred.
- **K-level kernel restructures (K1/K2/K3 — fuse the max-reduction, cache partition rows,
  privatize the contingency histogram).** These target the ARI kernel that now dominates the
  coefficient path. They are medium-to-high effort, restructure the floating-point reduction
  order (correctness risk), and — per the backlog — need Nsight Compute counter data to rank
  and validate. `ncu` is blocked in this environment (`RmProfilingAdminOnly=1`) and `nsys` is
  absent, so pursuing them here would be optimizing blind. Deferred to a profiling-enabled
  session.

**Stopping rationale.** Two rounds delivered a large, verified win on the one path that was
pathologically slow. The remaining tranche-1 items are measurably negligible and the
tranche-2 items require profiling this machine cannot provide — so this is a natural, honest
place to stop the first pass. Each deferred item remains scoped in the backlog for a
follow-up.

## Reproducing the measurements

```bash
# p-value path, GPU-only, matching this report
ccc-gpu-bench coef --features 30 50 80 200 --samples 1000 \
    --pvalue-n-perms 100 --gpu-only --repeats 2 --warmup 1 --format jsonl -o pval.jsonl

# correctness gate
pytest tests/ -m gpu -o addopts=""
```
