"""Standalone benchmark CLI for CCC (``ccc-gpu-bench`` / ``python -m ccc.bench``).

Reproducible, structured GPU-vs-CPU performance measurement decoupled from the
test suite: end-to-end coefficient benchmarks, ARI-kernel micro-benchmarks and
CPU parallelism scaling, emitted as JSON Lines / CSV with full environment
metadata. Runs degrade gracefully on machines without a GPU (CPU-side modes work;
GPU modes fail fast with a clear message). See ``ccc-gpu-bench --help``.
"""
