# Tasks — optimization-backlog

## 1. Backlog document

- [ ] 1.1 Write `docs/source/development/optimization_backlog` page from the review's ranked table (all K/H/A items with file:line refs, impact/effort/risk, needs-profiling flags); add to toctree
- [ ] 1.2 Cross-reference items partially addressed by fix-cuda-correctness (memory-bounded p-values landed; perf headroom remaining)

## 2. Profiling baseline

- [ ] 2.1 Document the procedure: bench-CLI workloads (20k×1k n_parts=9, ± pvalue, categorical-heavy), nsys/ncu commands, metrics to record
- [ ] 2.2 Capture baseline on reference GPU; summarize timeline shares (transfer/kernel/gap), ari_kernel ncu metrics (atomics serialization, memory throughput, occupancy) into the doc with commit hash + hardware
- [ ] 2.3 Re-rank profile-dependent items (K3, K7, K8, K1 sizing) on the evidence; mark tranche 1 and tranche 2 recommendations

## 3. Handoff

- [ ] 3.1 Confirm zero production-code diffs in this change; note in the doc that each tranche item becomes its own openspec change when implementation begins
