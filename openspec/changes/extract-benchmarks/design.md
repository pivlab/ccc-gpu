# Design — extract-benchmarks

## Context

Reference measurements to preserve (from the code being removed and committed logs): GPU vs CPU end-to-end times and speedups over grids f∈{500…56,200}, n∈{50…1000}, cores∈{6,12,24}, e.g. 10,000×1,000 → 21.7s GPU vs 802.6s CPU (37×); ARI-kernel micro-bench; CPU n_jobs scaling. A disconnected CPU-only profiling harness exists at `analysis/00-benchmark/` (cProfile category breakdown, papermill notebook).

## Goals / Non-Goals

**Goals:**
- One command reproduces any historical measurement with structured output.
- Runs degrade gracefully: CPU-only machines can run CPU-side benchmarks; GPU modes error clearly without a GPU.
- Output schema is stable enough to diff across commits (perf regression tracking, optimization-track baselines).

**Non-Goals:**
- CI perf gating (hosted runners are too noisy/GPU-less).
- Kernel-level profiling (nsys/ncu) automation — the CLI prints the suggested wrapper command; running profilers stays manual.
- Rewriting `analysis/00-benchmark/` notebooks; they may later call the CLI, not vice versa.

## Decisions

1. **Location: `libs/ccc/bench/` inside the wheel** with `[project.scripts] ccc-gpu-bench = "ccc.bench.cli:main"` and `python -m ccc.bench`. In-wheel (vs `scripts/`) so installed users can benchmark their own hardware — this is a GPU library; "how fast on my GPU?" is a user-facing question. Stdlib `argparse` — no new runtime deps.
2. **Output: JSON Lines by default, `--format csv` optional.** One record per case: config (mode, f, n, k, n_jobs, pvalue_n_perms, seed), environment (GPU name, driver, CUDA runtime, CPU model, package version), measurements (warmup-excluded wall times, repeats, mean/min, speedup, n_coefficients). Records written incrementally (long sweeps survive interruption).
3. **Methodology defaults**: fixed seed (42, overridable), 1 warmup + N=3 timed repeats (`--repeats`), report min and mean; GPU sync before/after timing; memory cleanup between cases (reuse the `clean_gpu_memory` logic).
4. **Presets are data, not comments**: a small dict of named grids (`smoke`, `paper`, `ari-smoke`, …); `--preset paper` reproduces the poster/README table. Custom grids via repeatable `--features/--samples/--n-jobs` args.
5. **`--profile` integrates the existing category profiler** for CPU runs (import the categorization from `analysis/00-benchmark/run_profiling.py` into `ccc.bench.profiling`); for GPU runs it emits the recommended `nsys profile …` invocation string.

## Risks / Trade-offs

- [Shipping bench code in the wheel adds surface] → small, stdlib-only, no import cost unless invoked.
- [CPU reference for big grids is very slow (800s+ per case)] → presets separate `smoke` from `paper`; `--gpu-only` flag skips CPU reference where speedup isn't needed.
- [Speedup numbers depend on hardware; README table could mislead] → output embeds full environment metadata; docs state the reference hardware.

## Migration Plan

Land after restructure-tests. Add CLI + docs; regenerate one `paper`-preset run on the reference machine to sanity-check parity with historical logs; delete nothing further (test-side removal already done).

## Open Questions

- None.
