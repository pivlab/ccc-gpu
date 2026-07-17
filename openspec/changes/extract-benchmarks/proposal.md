# Extract Benchmarking into a Standalone CLI

## Why

Benchmarking is currently entangled with pytest: GPU parity tests wrap every call in timing code, write ad-hoc `.log` files via a fragile cwd-relative path, and carry commented-out "poster benchmark" grids up to 56,200 features; four CPU tests assert wall-clock speedup ratios (flaky by nature). Tests should be tiny and fast; sweeps belong in a dedicated command. Decision (2026-07-16): a standalone CLI, which also becomes the profiling baseline for the future optimization track.

## What Changes

- New `ccc.bench` package with a CLI (`python -m ccc.bench`, plus console script `ccc-gpu-bench`), shipped in the wheel.
- Subcommands/modes covering everything the pytest-embedded benchmarks measured:
  - `coef`: end-to-end GPU vs CPU wall-clock + speedup over an `(n_features, n_samples, n_jobs)` grid, with `--pvalue-n-perms` and `--return-parts` variants;
  - `ari`: ARI-kernel-only GPU vs CPU throughput over `(n_features, n_parts, n_objs, k)`;
  - `scaling`: CPU threading/parallelism scaling (replaces the removed n_jobs speedup tests).
- Structured output: CSV or JSON lines (`--output`), one row per case with full config, timings, speedup, coefficient count, GPU/driver metadata, seed — replacing the ad-hoc log format (the committed logs' fields are the reference schema).
- `--profile` flag hooking the existing `analysis/00-benchmark/run_profiling.py` category logic (cProfile for CPU; optional nsys wrapper hint for GPU) so the optimization track gets its baseline from one tool.
- Presets: `--preset smoke` (seconds, sanity), `--preset paper` (the poster/README grids) instead of commented-out parameter lists.
- Documentation page (usage, output schema, how to reproduce README speedup table).

## Capabilities

### New Capabilities
- `benchmark-cli`: reproducible, structured performance measurement decoupled from the test suite.

### Modified Capabilities

(none — no existing specs)

## Impact

- New `libs/ccc/bench/` package; `pyproject.toml` (console script); docs page; README performance section points at the CLI.
- Depends on `restructure-tests` (which removes the in-test timing code this replaces); the measurement targets are defined by that removed code.
- No behavior change for the library itself.
