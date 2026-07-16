# Tasks — extract-benchmarks

## 1. Package skeleton

- [ ] 1.1 Create `libs/ccc/bench/` (`__init__.py`, `__main__.py`, `cli.py`, `runners.py`, `presets.py`, `output.py`, `env.py`); add `[project.scripts] ccc-gpu-bench`
- [ ] 1.2 Implement environment capture (GPU name/driver/CUDA via cupy or nvidia-smi fallback, CPU model, package version) and JSONL/CSV incremental writers

## 2. Benchmark modes

- [ ] 2.1 `coef` mode: GPU vs CPU end-to-end with warmup + repeats, `--pvalue-n-perms`, `--return-parts`, `--gpu-only/--cpu-only`, grid args
- [ ] 2.2 `ari` mode: kernel-level GPU vs CPU (port measurement intent from deleted `test_pairwise_ari_benchmark_features`)
- [ ] 2.3 `scaling` mode: CPU n_jobs scaling (replaces deleted speedup-assertion tests)
- [ ] 2.4 Presets: `smoke` and `paper` grids (paper = poster/README sweep); GPU memory cleanup between cases

## 3. Profiling integration

- [ ] 3.1 Extract category-profiling logic from `analysis/00-benchmark/run_profiling.py` into `ccc.bench.profiling`; wire `--profile` for CPU runs; print nsys suggestion for GPU runs

## 4. Validation and docs

- [ ] 4.1 Run `--preset smoke` on CPU-only and GPU machines; run one `paper` case and sanity-check against historical committed-log numbers
- [ ] 4.2 Docs page: usage, output schema, reference-hardware note, reproducing the README performance table; link from README
- [ ] 4.3 Add a tiny CPU-only smoke test for the CLI plumbing itself (arg parsing, JSONL output) — marked fast, no timing assertions
