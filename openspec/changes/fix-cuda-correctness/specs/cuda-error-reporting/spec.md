# Spec Delta: cuda-error-reporting

## ADDED Requirements

### Requirement: CUDA failures raise Python exceptions

The CUDA extension SHALL report every CUDA runtime error (launch failure, sync failure, allocation failure, invalid input) by raising a Python exception with a message identifying the failing operation. Library code MUST NOT call `exit()` or otherwise terminate the host process.

#### Scenario: Kernel launch failure surfaces as an exception

- **WHEN** a kernel launch fails (e.g., grid dimension exceeds device limits)
- **THEN** `ccc_cuda_ext.compute_coef` raises a `RuntimeError` naming the failed kernel, and the Python interpreter continues running

#### Scenario: No silent all-NaN results

- **WHEN** any kernel in the coefficient pipeline fails after launch
- **THEN** the call raises instead of returning the NaN-initialized output buffer

### Requirement: Input shape validation at the binding boundary

The extension SHALL validate that the `parts` array shape matches the `n_features`, `n_parts`, and `n_objs` arguments before any device work, and raise `ValueError` on mismatch.

#### Scenario: Mismatched dims rejected

- **WHEN** `compute_coef` is called with a `parts` array whose shape disagrees with the scalar dims
- **THEN** a `ValueError` describing expected vs actual shape is raised and no kernel is launched
