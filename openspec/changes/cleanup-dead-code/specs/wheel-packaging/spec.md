# Spec Delta: wheel-packaging

## ADDED Requirements

### Requirement: Every shipped module imports cleanly

The published wheel SHALL contain only modules whose imports are satisfied by the declared runtime dependencies. Research/analysis modules with undeclared dependencies SHALL be excluded from the wheel.

#### Scenario: Clean import surface

- **WHEN** the wheel is installed into a fresh environment with only declared dependencies
- **THEN** importing every module shipped in the wheel succeeds without `ModuleNotFoundError`

#### Scenario: Research modules excluded

- **WHEN** the wheel is built
- **THEN** `ccc/plots.py`, `ccc/methods.py`, `ccc/giant.py`, and `ccc/corr.py` are not present in the archive, while they remain in the repository for `analysis/` use
