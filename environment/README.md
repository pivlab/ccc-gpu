# Conda environments

| File | Purpose |
|------|---------|
| `environment-gpu.yml` | **Source of truth** for the development / CI environment: Python 3.12, NumPy 2.x, numba ≥0.61, the CUDA 12.5 toolchain, pybind11 3.x, and the test + docs tooling needed to build the extension and run the suite. This is the single source file that `../conda-lock.yml` is generated from. |
| `environment-dev.yaml` | Tiny helper env (`sphinx`, `mamba`, `conda-lock`) for building the docs and regenerating the lock without polluting `base`. |

The reproducible environment above deliberately excludes the research/analysis
dependencies (`matplotlib`, `seaborn`, `upsetplot`, `ipython`, `minepy`,
`requests`). Those back the in-repo `ccc.plots`/`methods`/`giant`/`corr` modules,
which are **not** part of the published wheel; install them separately in a
source checkout when running the `analysis/` notebooks. The former
`environment-benchmark.yaml` and `environment-toolchain.yaml` files were removed
to reduce drift. The only published package extra is `test` (`pip install ".[test]"`).

## Regenerating the lock file

`conda-lock.yml` is generated from `environment-gpu.yml` (its single declared
source). To re-solve the whole environment after changing a pin:

```bash
# needs conda-lock (see environment-dev.yaml, or `pipx install conda-lock`)
conda-lock --file environment/environment-gpu.yml --conda mamba --lockfile conda-lock.yml
```

To bump a single package to the newest version allowed by the source pins:

```bash
conda-lock lock --lockfile conda-lock.yml --update <package>
```

Install the locked environment with:

```bash
conda-lock install --name ccc-gpu conda-lock.yml  # add `--conda mamba` for speed
```

> Note: solving a full CUDA-bundled environment can take several minutes. After
> changing channels or pins, regenerate from scratch (delete `conda-lock.yml`
> first) rather than using `--update`, which refuses to run across channel
> changes.
