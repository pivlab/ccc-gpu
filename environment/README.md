# Conda environments

| File | Purpose |
|------|---------|
| `environment-gpu.yml` | **Source of truth** for the development / CI environment (Python 3.12, NumPy 2.x, numba ≥0.61, CUDA 12.x, the build toolchain, and test/docs/research extras). This is the single source file that `../conda-lock.yml` is generated from. |
| `environment-dev.yaml` | Tiny helper env (`sphinx`, `mamba`, `conda-lock`) for building the docs and regenerating the lock without polluting `base`. |

The `plots`/`research`/`test` package extras in `pyproject.toml`
(`pip install ".[plots,research,test]"`) replace the former
`environment-benchmark.yaml` and `environment-toolchain.yaml` files, which were
removed to reduce drift.

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

> Note: solving a full CUDA-bundled environment can take several minutes. The
> checked-in `conda-lock.yml` may still list the historical two-source command
> in its header comment; a regeneration with the command above rewrites it to
> the single `environment-gpu.yml` source.
