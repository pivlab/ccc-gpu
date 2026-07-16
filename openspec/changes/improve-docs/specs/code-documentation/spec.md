# Spec Delta: code-documentation

## ADDED Requirements

### Requirement: Docstrings match code behavior

Public API docstrings and type annotations SHALL accurately describe actual behavior: polymorphic return types, correct argument names, correct `-1`/`-2` partition marker legend, and correct element counts.

#### Scenario: Return contract accuracy

- **WHEN** a user inspects `ccc()`'s signature and docstring
- **THEN** the described/annotated return types cover the scalar, array, 2-tuple (p-values), and 3-tuple (return_parts) cases actually returned

### Requirement: CUDA kernels document their launch contract

Every `__global__` kernel SHALL carry a doc comment stating its grid/block mapping, required block dimensions, dynamic shared-memory size formula (if any), and output conventions (NaN/clamp semantics). The extension's primary entry point (`compute_coef`) SHALL document its full return-tuple structure and limits.

#### Scenario: Kernel contract present

- **WHEN** a developer reads any kernel definition in `libs/ccc_cuda_ext/`
- **THEN** the launch requirements and output conventions are stated at the definition site, and no comment contradicts the code (e.g., claiming parallelism where execution is serial)
