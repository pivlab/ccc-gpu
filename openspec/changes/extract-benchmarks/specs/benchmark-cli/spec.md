# Spec Delta: benchmark-cli

## ADDED Requirements

### Requirement: Standalone benchmark command

The package SHALL provide a benchmark CLI (`ccc-gpu-bench` / `python -m ccc.bench`) with modes covering end-to-end coefficient benchmarks (GPU vs CPU, with p-value and return-parts variants), ARI-kernel micro-benchmarks, and CPU parallelism scaling.

#### Scenario: End-to-end sweep

- **WHEN** `ccc-gpu-bench coef --features 1000 --samples 500` runs on a GPU machine
- **THEN** it reports GPU time, CPU reference time, and speedup for that case without involving pytest

#### Scenario: Smoke preset

- **WHEN** `ccc-gpu-bench coef --preset smoke` runs
- **THEN** the full run completes in well under a minute on reference hardware

### Requirement: Structured, reproducible output

Every benchmark record SHALL include the full case configuration, seed, environment metadata (GPU/driver/CUDA/CPU/package version), and warmup-excluded repeated timings, emitted as JSON Lines (default) or CSV to a chosen output path, written incrementally.

#### Scenario: Resumable structured output

- **WHEN** a long sweep is interrupted midway
- **THEN** all completed cases are present and parseable in the output file

### Requirement: Graceful degradation without a GPU

CPU-only modes SHALL run on machines without a GPU; GPU modes SHALL fail fast with a clear message rather than a traceback from a missing import.

#### Scenario: No GPU present

- **WHEN** `ccc-gpu-bench coef` runs where the CUDA extension is unavailable
- **THEN** the tool exits with an explanatory error (or runs CPU-only when `--cpu-only` is passed)

### Requirement: Profiling baseline support

The CLI SHALL provide a `--profile` option producing a category-level CPU profile (partitioning vs ARI vs overhead), and SHALL print the recommended external profiler invocation for GPU runs.

#### Scenario: CPU category profile

- **WHEN** `ccc-gpu-bench coef --cpu-only --profile` runs
- **THEN** output includes time attributed to partitioning, coefficient computation, and other categories
