# Spec Delta: pvalue-computation

## ADDED Requirements

### Requirement: P-value indexing is 64-bit safe

The p-value permutation path SHALL compute all buffer indices and comparison counts in 64-bit arithmetic, producing correct results when `n_feature_comp × n_perms` exceeds 2^32.

#### Scenario: Large feature count with permutations

- **WHEN** p-values are computed for an input where `n_feature_comp × n_perms > 2^32`
- **THEN** every comparison's p-value corresponds to its own permutation block (no wrapped/aliased indexing)

### Requirement: No silent cluster-count cliff

The permutation ARI computation SHALL support the same cluster counts as the main coefficient kernel. If a limit is unavoidable, exceeding it SHALL raise an error rather than silently returning ARI = 0.

#### Scenario: More than 16 clusters

- **WHEN** p-values are computed for partitions with k > 16 clusters
- **THEN** permuted ARIs are computed correctly (matching a CPU reference within tolerance), or a clear error is raised — never a silent 0.0

### Requirement: Permutation sums do not overflow

Pair-confusion sums in the permutation ARI SHALL use 64-bit accumulators, matching the main kernel, so results are correct when a cluster holds more than ~46k objects.

#### Scenario: Large n_objects

- **WHEN** p-values are computed on features with ≥ 100,000 objects concentrated in few clusters
- **THEN** permuted ARI values match the CPU reference implementation within float tolerance

### Requirement: Null distribution uses observed-statistic semantics

Permuted CCC values SHALL be computed under the same invalid-partition rules as observed CCC values: categorical marker (-1) partitions contribute ARI 0.0, singleton marker (-2) partitions yield NaN, and the same max/clamp reduction applies to both.

#### Scenario: Categorical feature p-value parity

- **WHEN** p-values are computed for a feature pair involving a categorical feature
- **THEN** the permutation null distribution applies identical partition-validity rules as the observed coefficient, and the resulting p-value is consistent with the CPU implementation's permutation scheme

### Requirement: P-value memory use is bounded

The p-value path SHALL bound device memory for permutation storage by batching, such that any input size accepted by the coefficient-only path also completes with p-values enabled (given time), instead of failing with an allocation error proportional to `n_feature_comp × n_perms`.

#### Scenario: Input larger than one batch

- **WHEN** p-values are requested for an input whose full permutation buffer would exceed available device memory
- **THEN** computation proceeds in batches and returns complete, correct p-values
