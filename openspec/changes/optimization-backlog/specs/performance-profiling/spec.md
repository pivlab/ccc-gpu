# Spec Delta: performance-profiling

## ADDED Requirements

### Requirement: Documented optimization backlog

The developer documentation SHALL contain a ranked optimization backlog in which every entry cites the code it targets (file:line), states estimated impact/effort/risk, and records whether it is inspection-justified or requires profiling evidence.

#### Scenario: Contributor picks up an item

- **WHEN** a contributor reads the backlog document
- **THEN** each entry gives enough context (code references, rationale, measurement status, tranche) to scope an implementation change without re-deriving the analysis

### Requirement: Reproducible profiling baseline

The repository SHALL document a repeatable GPU profiling procedure (exact benchmark-CLI workloads, nsys/ncu commands, metrics to record) and the backlog SHALL include baseline measurements from that procedure, tagged with the commit hash and hardware they were captured on.

#### Scenario: Baseline captured

- **WHEN** the documented procedure runs on the reference workload
- **THEN** the resulting timeline/kernel metrics are summarized in the backlog and profile-dependent items are re-ranked on that evidence

#### Scenario: No profiling artifacts in git

- **WHEN** profiling runs produce raw report files
- **THEN** only summaries enter the repository; raw reports are referenced by location and hash
