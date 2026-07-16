# Spec Delta: code-linting

## ADDED Requirements

### Requirement: Single-source lint and format configuration

The repository SHALL define exactly one lint/format configuration per language — ruff (lint + format) for Python in the root `pyproject.toml`, and `.clang-format`/`.clang-tidy` at the repo root for C++/CUDA. No secondary or shadowing config files SHALL exist.

#### Scenario: Consistent config resolution

- **WHEN** ruff or clang-format runs on any file in `libs/`, `tests/`, or `scripts/`
- **THEN** the root configuration applies (no per-directory divergence)

#### Scenario: Redundant formatters removed

- **WHEN** pre-commit hooks run
- **THEN** exactly one Python formatter (ruff-format) executes; black is not configured

### Requirement: Clean-tree enforcement

`pre-commit run --all-files` SHALL pass on the committed tree, and formatting-only changes SHALL be isolated in dedicated commits listed in `.git-blame-ignore-revs`.

#### Scenario: Fresh clone lints clean

- **WHEN** a contributor runs `pre-commit run --all-files` on a fresh clone
- **THEN** all hooks pass without modifying any file

### Requirement: CUDA/C++ static analysis available

The build SHALL export `compile_commands.json`, and a documented clang-tidy invocation SHALL run the curated check list over the CUDA/C++ sources.

#### Scenario: clang-tidy runs on extension sources

- **WHEN** the documented clang-tidy command is run after a CMake configure
- **THEN** it analyzes `libs/ccc_cuda_ext/*.cu*` host code and reports findings using the repo `.clang-tidy` configuration
