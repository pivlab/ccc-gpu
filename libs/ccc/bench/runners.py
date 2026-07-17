"""Benchmark runners: coef (end-to-end), ari (kernel), scaling (CPU n_jobs).

Methodology: fixed seed, ``warmup`` untimed calls then ``repeats`` timed calls;
report min and mean wall time; GPU is synchronized before/after timing and its
memory pool is freed between cases.
"""

import itertools
import sys
import time
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_gpu() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _gpu_sync() -> None:
    import cupy as cp

    cp.cuda.Device().synchronize()


def _timed(fn, repeats: int, warmup: int, sync=None):
    """Return (min_s, mean_s, last_result) for ``fn`` with warmup + repeats."""
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times), result


def _n_coefficients(n_features: int) -> int:
    return n_features * (n_features - 1) // 2


# --------------------------------------------------------------------------- #
# coef: end-to-end GPU vs CPU
# --------------------------------------------------------------------------- #
def run_coef(
    *,
    writer,
    env: dict,
    features: list[int],
    samples: list[int],
    n_jobs: list[int],
    pvalue_n_perms: int | None,
    return_parts: bool,
    seed: int,
    repeats: int,
    warmup: int,
    gpu_only: bool,
    cpu_only: bool,
    verbose: bool = True,
) -> list[dict]:
    import numpy as np

    ccc_cpu = None
    ccc_gpu = None
    if not gpu_only:
        from ccc.coef.impl import ccc as ccc_cpu
    if not cpu_only:
        from ccc.coef.impl_gpu import ccc as ccc_gpu

    records = []
    for nf, ns, nj in itertools.product(features, samples, n_jobs):
        np.random.seed(seed)
        data = np.random.rand(nf, ns)
        kwargs = {"pvalue_n_perms": pvalue_n_perms, "return_parts": return_parts}

        gpu_min = gpu_mean = None
        cpu_min = cpu_mean = None
        if ccc_gpu is not None:
            gpu_min, gpu_mean, _ = _timed(
                lambda d=data, j=nj, kw=kwargs: ccc_gpu(d, n_jobs=j, **kw),
                repeats,
                warmup,
                sync=_gpu_sync,
            )
            _clean_gpu()
        if ccc_cpu is not None:
            cpu_min, cpu_mean, _ = _timed(
                lambda d=data, j=nj, kw=kwargs: ccc_cpu(d, n_jobs=j, **kw),
                repeats,
                warmup,
            )

        speedup = cpu_min / gpu_min if (gpu_min and cpu_min and gpu_min > 0) else None
        record = {
            "mode": "coef",
            "timestamp": _now_iso(),
            "n_features": nf,
            "n_samples": ns,
            "n_jobs": nj,
            "pvalue_n_perms": pvalue_n_perms,
            "return_parts": return_parts,
            "seed": seed,
            "repeats": repeats,
            "warmup": warmup,
            "n_coefficients": _n_coefficients(nf),
            "gpu_time_min_s": gpu_min,
            "gpu_time_mean_s": gpu_mean,
            "cpu_time_min_s": cpu_min,
            "cpu_time_mean_s": cpu_mean,
            "speedup": speedup,
            **env,
        }
        writer.write(record)
        records.append(record)
        if verbose:
            _print_coef(record)
    return records


def _print_coef(r: dict) -> None:
    parts = [f"coef f={r['n_features']} n={r['n_samples']} jobs={r['n_jobs']}"]
    if r["gpu_time_min_s"] is not None:
        parts.append(f"GPU={r['gpu_time_min_s']:.4f}s")
    if r["cpu_time_min_s"] is not None:
        parts.append(f"CPU={r['cpu_time_min_s']:.4f}s")
    if r["speedup"] is not None:
        parts.append(f"speedup={r['speedup']:.2f}x")
    print("  " + "  ".join(parts), file=sys.stderr)


# --------------------------------------------------------------------------- #
# ari: kernel-level GPU vs CPU
# --------------------------------------------------------------------------- #
def _pairwise_combinations(parts):
    import numpy as np

    pairs = []
    n_slices = parts.shape[0]
    for i in range(n_slices):
        for j in range(i + 1, n_slices):
            for row_i in parts[i]:
                for row_j in parts[j]:
                    pairs.append((row_i, row_j))
    return np.array(pairs)


def run_ari(
    *,
    writer,
    env: dict,
    n_features: list[int],
    n_parts: list[int],
    n_objs: list[int],
    k: list[int],
    seed: int,
    repeats: int,
    warmup: int,
    gpu_only: bool,
    cpu_only: bool,
    verbose: bool = True,
) -> list[dict]:
    import numpy as np

    ari_ref = None
    ari_int32 = None
    if not gpu_only:
        from ccc.sklearn.metrics import adjusted_rand_index as ari_ref
    if not cpu_only:
        import ccc_cuda_ext

        ari_int32 = ccc_cuda_ext.ari_int32

    records = []
    for nf, npart, nobj, kk in itertools.product(n_features, n_parts, n_objs, k):
        np.random.seed(seed)
        parts = np.random.randint(0, kk, size=(nf, npart, nobj), dtype=np.int32)
        n_feature_comp = nf * (nf - 1) // 2
        n_aris = n_feature_comp * npart * npart

        gpu_min = gpu_mean = None
        cpu_min = cpu_mean = None
        if ari_int32 is not None:
            gpu_min, gpu_mean, _ = _timed(
                lambda p=parts, a=nf, b=npart, c=nobj: ari_int32(p, a, b, c),
                repeats,
                warmup,
                sync=_gpu_sync,
            )
            _clean_gpu()
        if ari_ref is not None:
            pairs = _pairwise_combinations(parts)

            def _cpu_ref(pairs=pairs, ref=ari_ref):
                return [ref(p0, p1) for p0, p1 in pairs]

            cpu_min, cpu_mean, _ = _timed(_cpu_ref, repeats, warmup)

        speedup = cpu_min / gpu_min if (gpu_min and cpu_min and gpu_min > 0) else None
        record = {
            "mode": "ari",
            "timestamp": _now_iso(),
            "n_features": nf,
            "n_parts": npart,
            "n_objs": nobj,
            "k": kk,
            "seed": seed,
            "repeats": repeats,
            "warmup": warmup,
            "n_aris": n_aris,
            "gpu_time_min_s": gpu_min,
            "gpu_time_mean_s": gpu_mean,
            "cpu_time_min_s": cpu_min,
            "cpu_time_mean_s": cpu_mean,
            "speedup": speedup,
            **env,
        }
        writer.write(record)
        records.append(record)
        if verbose:
            msg = [f"ari f={nf} parts={npart} objs={nobj} k={kk} n_aris={n_aris}"]
            if gpu_min is not None:
                msg.append(f"GPU={gpu_min:.4f}s")
            if cpu_min is not None:
                msg.append(f"CPU={cpu_min:.4f}s")
            if speedup is not None:
                msg.append(f"speedup={speedup:.2f}x")
            print("  " + "  ".join(msg), file=sys.stderr)
    return records


# --------------------------------------------------------------------------- #
# scaling: CPU n_jobs scaling
# --------------------------------------------------------------------------- #
def run_scaling(
    *,
    writer,
    env: dict,
    features: list[int],
    samples: list[int],
    n_jobs: list[int],
    pvalue_n_perms: int | None,
    seed: int,
    repeats: int,
    warmup: int,
    verbose: bool = True,
) -> list[dict]:
    import numpy as np

    from ccc.coef.impl import ccc as ccc_cpu

    records = []
    for nf, ns in itertools.product(features, samples):
        np.random.seed(seed)
        data = np.random.rand(nf, ns)

        times = {}
        for nj in sorted(n_jobs):
            t_min, t_mean, _ = _timed(
                lambda d=data, j=nj, pv=pvalue_n_perms: ccc_cpu(
                    d, n_jobs=j, pvalue_n_perms=pv
                ),
                repeats,
                warmup,
            )
            times[nj] = (t_min, t_mean)

        baseline = times[min(times)][0]
        for nj in sorted(times):
            t_min, t_mean = times[nj]
            speedup = baseline / t_min if t_min > 0 else None
            record = {
                "mode": "scaling",
                "timestamp": _now_iso(),
                "n_features": nf,
                "n_samples": ns,
                "n_jobs": nj,
                "baseline_n_jobs": min(times),
                "pvalue_n_perms": pvalue_n_perms,
                "seed": seed,
                "repeats": repeats,
                "warmup": warmup,
                "cpu_time_min_s": t_min,
                "cpu_time_mean_s": t_mean,
                "speedup": speedup,
                **env,
            }
            writer.write(record)
            records.append(record)
            if verbose:
                sp = f"{speedup:.2f}x" if speedup is not None else "n/a"
                print(
                    f"  scaling f={nf} n={ns} jobs={nj} CPU={t_min:.4f}s speedup={sp}",
                    file=sys.stderr,
                )
    return records
