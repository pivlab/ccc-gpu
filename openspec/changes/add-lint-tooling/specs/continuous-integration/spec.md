# Spec Delta: continuous-integration

## ADDED Requirements

### Requirement: CI runs on pushes and pull requests

A GitHub Actions workflow SHALL run on every push to main and every pull request, with three jobs: lint (Python + C++ format/lint checks), CPU-only tests, and a wheel-build smoke check.

#### Scenario: Lint gate

- **WHEN** a PR contains a file violating ruff or clang-format rules
- **THEN** the lint job fails and reports the offending files

#### Scenario: CPU test gate

- **WHEN** a PR breaks a CPU-path test
- **THEN** the test job fails; GPU-marked tests are not collected on the hosted runner

#### Scenario: Build smoke

- **WHEN** a PR breaks compilation of the CUDA extension
- **THEN** the build job fails even though the runner has no GPU device
