# Tasks — optimization-backlog

## 1. Backlog document

- [x] 1.1 Write `docs/source/development/optimization_backlog` page from the review's ranked table (all K/H/A items with file:line refs, impact/effort/risk, needs-profiling flags); add to toctree
- [x] 1.2 Cross-reference items partially addressed by fix-cuda-correctness (memory-bounded p-values landed; perf headroom remaining)

## 2. Profiling baseline

- [~] 2.1 Document the procedure: nsys/ncu commands + the wall-clock bench workloads and metrics ARE documented. PARTIAL: the categorical-heavy / fixed-`n_parts=9` workload cannot be produced by the shipped `ccc-gpu-bench coef` (continuous random data only, no `--n-parts`/categorical-fraction knob); measuring A2's invalid-partition fraction needs a small bench enhancement first (noted in the doc's "What could NOT be captured" section).
- [~] 2.2 Capture baseline on reference GPU. DONE: GPU coefficient-scaling, GPU-vs-CPU speedups, CPU category split, and the p-value blow-up captured (with commit hash + hardware). DEFERRED: nsys timeline + ari_kernel ncu counters (atomics/throughput/occupancy) — NOT capturable in this env (nsys absent; ncu blocked by `RmProfilingAdminOnly=1` / `ERR_NVGPUCTRPERM`). Exact commands documented for a machine with the tools + admin.
- [~] 2.3 Tranche assignment DONE from inspection (tranche 1 = H1/H2/H3/A3; tranche 2 = K1+K2/K6). DEFERRED: evidence-based re-ranking of the counter-dependent items (K3/K7/K8/K1-sizing/A2) — cannot be done without the nsys/ncu data from 2.2; those stay in the "needs profiling" group.

## 3. Handoff

- [x] 3.1 Confirm zero production-code diffs in this change; note in the doc that each tranche item becomes its own openspec change when implementation begins
