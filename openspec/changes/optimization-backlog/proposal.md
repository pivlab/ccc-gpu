# Establish the Performance Optimization Backlog (Exploration Only)

## Why

The 2026-07-16 review identified substantial optimization headroom (per-comparison kernel launch+sync loops, redundant H2D uploads every batch, a ~4 GB eliminable intermediate buffer, thread-serial permutation kernels), but the repo has no GPU profiles (existing `analysis/00-benchmark/` profiles the CPU implementation only) and several opportunities need measurement to rank. Decision: cleanup and refactoring land first; this change only produces the **documented, evidence-ranked backlog and profiling baseline** — no optimization implementation.

## What Changes

- Add `docs/source/development/optimization_backlog.rst` (or md) recording the ranked opportunity table with code references, estimated impact/effort/risk, and which items require profiling validation first:
  - **H2** p-value host loop: one launch + `cudaDeviceSynchronize` per feature comparison; monolithic `n_feature_comp × n_perms` buffer (partially addressed for correctness in `fix-cuda-correctness`; perf headroom remains)
  - **H1** hoist per-batch `parts` H2D upload + per-batch `thrust::reduce` for `k` (pass `k` from Python — it's already known there)
  - **K1+K2** block-per-feature-pair fused ARI+argmax: drop the `d_aris` intermediate (~8 GB traffic/batch) and `findMaxAriKernel`; cache partition rows in shared memory
  - **K6** block-parallel rewrite of `computePermutationCCC` (removes local-memory contingency spills and the duplicate ARI implementation)
  - **H3** allocation reuse/pooling; **H4** CPU partitioning: hoist argsort/rank across k values, parallel default; **K3** privatized contingency histograms; **K8** pinned memory + streams; **A3** memory-aware batch sizing; **A2** skip invalid partition pairs; **K7** vectorized int16 loads; misc (scatter-loop memcpy, per-batch memGetInfo, warp-sized reductions)
- Capture the profiling baseline using the benchmark CLI: one nsys timeline + ncu section on a representative workload (e.g. 20k features × 1k objects, n_parts=9, with and without p-values), stored/summarized in the backlog doc to convert "estimated" ranks into measured ones.
- Re-rank the table on that evidence and mark the recommended first implementation tranche.

## Capabilities

### New Capabilities
- `performance-profiling`: a documented GPU profiling procedure and recorded baseline that ranks the optimization backlog by evidence.

### Modified Capabilities

(none — no existing specs)

## Impact

- Documentation + profiling artifacts only; **zero production-code changes**.
- Depends on: `extract-benchmarks` (the CLI provides the workload runner), `fix-cuda-correctness` (baseline must measure correct code).
- Output feeds future per-optimization openspec changes, each gated on the baseline numbers.
