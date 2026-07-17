# Spec Delta: user-documentation

## ADDED Requirements

### Requirement: Complete p-value documentation

User documentation SHALL explain `pvalue_n_perms` as a one-sided permutation test — including the estimator `(count + 1) / (n_perms + 1)`, minimum resolvable p-value, returned shapes for 1D and 2D inputs, and a computational-cost warning for 2D inputs — in the docstring, on a dedicated docs-site section, and summarized in the README.

#### Scenario: User discovers p-value semantics from any entry point

- **WHEN** a user reads the `ccc()` docstring, the docs-site p-value section, or the README p-value example
- **THEN** each states or links to the permutation-test method, interpretation, and cost characteristics, with argument names matching the actual signature

### Requirement: Rendered API reference

The documentation site SHALL render the public API docstrings via autodoc, and the site SHALL build without warnings on the API pages.

#### Scenario: Docstring reaches the site

- **WHEN** the Sphinx site builds
- **THEN** an API reference page shows the current `ccc()` docstrings for both CPU and GPU implementations

### Requirement: Consistent project metadata

Version, license, citation, install instructions, Python range, and CUDA/compute-capability requirements SHALL each have a single authoritative source and agree everywhere they appear (pyproject, `ccc.__version__`, Sphinx conf, CITATION.cff, README, docs site).

#### Scenario: Version single-sourcing

- **WHEN** the package version is bumped in `pyproject.toml`
- **THEN** `ccc.__version__` and the docs-site version reflect it without further edits

#### Scenario: No contradictory claims

- **WHEN** the repo is audited for version/license/CUDA/Python/install claims
- **THEN** no two locations disagree
