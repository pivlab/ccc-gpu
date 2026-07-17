# Tasks — optimization-backlog

## 1. Backlog document

- [x] 1.1 Write `docs/source/development/optimization_backlog` page from the review's ranked table (all K/H/A items with file:line refs, impact/effort/risk, needs-profiling flags); add to toctree
- [x] 1.2 Cross-reference items partially addressed by fix-cuda-correctness (memory-bounded p-values landed; perf headroom remaining)

## 2. Profiling baseline

- [x] 2.1 Document the procedure: bench-CLI workloads (20k×1k n_parts=9, ± pvalue, categorical-heavy), nsys/ncu commands, metrics to record
- [x] 2.2 Capture baseline on reference GPU; summarize timeline shares (transfer/kernel/gap), ari_kernel ncu metrics (atomics serialization, memory throughput, occupancy) into the doc with commit hash + hardware — GPU coefficient/p-value/CPU-split baselines captured; nsys/ncu kernel-counter capture NOT possible in this env (nsys not installed; ncu blocked by `RmProfilingAdminOnly=1` / `ERR_NVGPUCTRPERM`), documented with exact commands to run later
- [x] 2.3 Re-rank profile-dependent items (K3, K7, K8, K1 sizing) on the evidence; mark tranche 1 and tranche 2 recommendations — profile-dependent items kept in a "needs profiling" group (cannot be re-ranked without counters this env cannot read); tranche 1 (H1/H2/H3/A3) and tranche 2 (K1+K2/K6) assigned from inspection

## 3. Handoff

- [x] 3.1 Confirm zero production-code diffs in this change; note in the doc that each tranche item becomes its own openspec change when implementation begins
