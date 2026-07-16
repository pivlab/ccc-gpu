# Design — optimization-backlog

## Context

Full opportunity analysis exists from the review (kernel-level, host-side, algorithmic, each with file:line references and impact/effort/risk estimates). What's missing is measurement: no nsys/ncu data exists, so compute-vs-transfer-vs-launch-latency shares are estimates. Some items are safe to schedule without profiling (pathological by inspection: per-comparison sync loop, per-batch re-upload); others (atomics contention, vectorized loads, streams overlap) need profiles to justify.

## Goals / Non-Goals

**Goals:**
- A single authoritative backlog document a future contributor can execute from, with measured (not guessed) rankings for the profile-dependent items.
- A repeatable profiling procedure (exact commands, workload, metrics to record).

**Non-Goals:**
- Implementing any optimization (explicit user decision: after cleanup/refactor).
- CI-automated profiling.

## Decisions

1. **Backlog lives in the docs site** (`development/optimization_backlog`) not an issue tracker: it carries tables and code references that benefit from versioned review alongside the code they describe. Individual implementation efforts become their own openspec changes later.
2. **Baseline workloads**: (a) 20k features × 1k objects, n_parts=9, coefficient-only; (b) same with `pvalue_n_perms=100`; (c) one categorical-heavy case for A2 sizing. Driven via `ccc-gpu-bench` presets so numbers are reproducible.
3. **Metrics recorded**: nsys — timeline share of H2D/D2H vs kernels vs gaps (launch latency), per-kernel time; ncu on `ari_kernel` — shared-memory atomics serialization (K3), memory throughput vs roofline (K7), occupancy (K1/K2 sizing). Summarized as a table in the doc; raw reports archived outside git (path + hash noted) to keep the repo artifact-free.
4. **Tranche recommendation rule**: items ranked High impact + Low/Medium effort + evidence-confirmed form tranche 1 (expected: H1, H2-perf, H3, A3); K1/K6 restructures form tranche 2 gated on golden parity tests.

## Risks / Trade-offs

- [Baseline hardware ≠ user hardware] → record full environment metadata (the bench CLI embeds it); rankings note where hardware could reorder priorities.
- [Backlog rots as code changes] → each future optimization change updates the doc's status column; the doc records the commit hash it was measured at.

## Migration Plan

Docs-only PR after extract-benchmarks lands. No rollback concerns.

## Open Questions

- None.
