# Spec Delta: test-suite

## ADDED Requirements

### Requirement: Environment-aware collection

The test suite SHALL be partitioned by registered markers (`gpu`, `slow`, `network`) with strict-marker enforcement, and `pytest tests/` SHALL pass on a machine without a GPU by skipping (not erroring on) GPU tests.

#### Scenario: CPU-only machine

- **WHEN** `pytest tests/` runs where cupy or the CUDA extension is unavailable
- **THEN** GPU tests are reported as skipped with a reason, and the run exits green

#### Scenario: GPU machine

- **WHEN** `pytest -m gpu tests/` runs on a machine with a working GPU + extension
- **THEN** the GPU suite runs and passes in a small-input, timing-free configuration

### Requirement: Correctness tests contain no benchmarking

Correctness tests SHALL NOT measure wall-clock time, assert speedups, or write log files. Performance measurement lives exclusively in the benchmark CLI.

#### Scenario: No timing side effects

- **WHEN** the full pytest suite runs
- **THEN** no `.log` files are produced and no assertion depends on elapsed time

### Requirement: Numerical tolerance contract

GPU-vs-CPU parity assertions SHALL use `atol=1e-6, rtol=1e-6`; any looser tolerance MUST carry an inline justification comment.

#### Scenario: Parity assertion

- **WHEN** a GPU result is compared against the CPU reference in a parity test
- **THEN** the comparison uses the contract tolerance or documents why it cannot

### Requirement: Compiled CUDA tests run under ctest

The build SHALL provide a `CCC_BUILD_TESTS` CMake option (default OFF) that compiles the CUDA gtests and registers them with ctest; googletest SHALL only be fetched when the option is ON.

#### Scenario: Native test run

- **WHEN** the project is configured with `-DCCC_BUILD_TESTS=ON` on a GPU machine and `ctest` runs
- **THEN** the CUDA test binaries build and pass (or skip with an explicit tracked reason)

#### Scenario: Wheel build stays offline

- **WHEN** a wheel is built with default options
- **THEN** googletest is not downloaded

### Requirement: GPU coverage for known divergence points

The GPU suite SHALL include tests for categorical inputs with `return_parts=True`, constant-feature matrices, too-few-objects error parity, and mixed numerical+categorical DataFrames (each passing, or `xfail(strict=True)` with a tracked issue).

#### Scenario: Categorical return_parts

- **WHEN** `ccc(df, return_parts=True)` runs on GPU with a categorical feature
- **THEN** results match the CPU implementation, or the test is an explicit strict xfail referencing an open issue
