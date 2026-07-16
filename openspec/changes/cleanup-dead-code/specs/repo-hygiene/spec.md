# Spec Delta: repo-hygiene

## ADDED Requirements

### Requirement: No dead code in the library tree

All code under `libs/` SHALL be reachable from a supported public entry point, a build target, or an executed test. Demo, prototype, and fully-commented-out files SHALL be removed.

#### Scenario: Dead symbol sweep

- **WHEN** the library is searched for the removed symbols (`cdist_parts_basic`, `compute_ccc_perms` in impl_gpu, `metrics_gpu`, `example_return_optional_vectors`, `ari_reduced`, `streamProcessingKernel`)
- **THEN** no definitions remain, and the full test suite plus `analysis/` imports still pass

### Requirement: No generated artifacts in version control

Benchmark/test log files and other generated artifacts SHALL NOT be committed; ignore rules SHALL prevent reintroduction.

#### Scenario: Log files removed and ignored

- **WHEN** the cleanup lands and a benchmark/test writes a log
- **THEN** no `.log` files are tracked by git and `git status` shows new logs as ignored

### Requirement: Top-level build/docker files work or don't exist

Every top-level operational file (Dockerfile, entrypoint, environment definitions) SHALL either build/run successfully as documented or be removed.

#### Scenario: Docker build succeeds

- **WHEN** `docker build .` is run on the repo (if a Dockerfile is present)
- **THEN** the build completes and the image can run the documented verification one-liner (CPU import path at minimum)
