"""CPU category profiling + GPU profiler suggestion.

The category map and attribution logic are ported from
``analysis/00-benchmark/run_profiling.py`` so the optimization track gets its
baseline from this one tool. ``--profile`` on a CPU run prints where time goes
(partitioning vs ARI vs coordination vs ...); on a GPU run it prints the
recommended external profiler (nsys) invocation, since kernel profiling stays
manual.
"""

import cProfile
import pstats

# Function -> category attribution (ported from analysis/00-benchmark).
FUNCTION_CATEGORIES = {
    "adjusted_rand_index": "ARI",
    "get_pair_confusion_matrix": "ARI",
    "get_contingency_matrix": "ARI",
    "get_parts": "Partitioning",
    "run_quantile_clustering": "Partitioning",
    "get_feature_parts": "Partitioning",
    "get_range_n_clusters": "Partitioning",
    "get_perc_from_k": "Partitioning",
    "rank": "Ranking",
    "cdist_parts_basic": "Coordination",
    "compute_ccc": "Coordination",
    "compute_coef": "Coordination",
    "ccc": "Coordination",
    "get_chunks": "Coordination",
    "get_coords_from_index": "Coordination",
    "get_feature_type_and_encode": "Coordination",
}

_NUMPY_FUNCS = frozenset(
    [
        "searchsorted",
        "argsort",
        "zeros",
        "unique",
        "full",
        "ravel",
        "dot",
        "sum",
        "max",
        "argmax",
        "floor",
        "sqrt",
        "ceil",
        "round",
        "array",
        "arange",
        "empty",
        "copy",
    ]
)


def categorize_function(func_name: str, filename: str) -> str:
    """Categorize a profiled function by name and source file."""
    if func_name in FUNCTION_CATEGORIES:
        return FUNCTION_CATEGORIES[func_name]
    lower = filename.lower()
    if func_name in _NUMPY_FUNCS or "numpy" in lower or "numba" in lower:
        return "NumPy/Numba"
    if "ccc" in lower:
        return "Other CCC"
    return "Other"


def category_breakdown(stats: pstats.Stats) -> tuple[list[dict], float]:
    """Return per-category (time, calls, pct) totals and the CCC total time."""
    stats_dict = stats.stats

    total_time = 0.0
    for (filename, _line, func_name), value in stats_dict.items():
        if func_name == "ccc" and "impl.py" in filename:
            total_time = value[3]
            break
    if total_time == 0.0:
        total_time = max(v[3] for v in stats_dict.values())

    totals: dict[str, dict] = {}
    for (filename, _line, func_name), value in stats_dict.items():
        ncalls, tottime = value[0], value[2]
        category = categorize_function(func_name, filename)
        bucket = totals.setdefault(category, {"tottime": 0.0, "calls": 0})
        bucket["tottime"] += tottime
        bucket["calls"] += ncalls

    rows = [
        {
            "category": cat,
            "tottime": data["tottime"],
            "calls": data["calls"],
            "pct": (data["tottime"] / total_time * 100.0) if total_time else 0.0,
        }
        for cat, data in totals.items()
    ]
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    return rows, total_time


def profile_cpu(ccc_fn, data, n_jobs: int = 1, **ccc_kwargs) -> dict:
    """cProfile a single CPU ``ccc`` call and return the category breakdown."""
    profiler = cProfile.Profile()
    profiler.enable()
    ccc_fn(data, n_jobs=n_jobs, **ccc_kwargs)
    profiler.disable()

    stats = pstats.Stats(profiler)
    rows, total_time = category_breakdown(stats)
    return {"total_time": total_time, "categories": rows}


def print_cpu_profile(profile: dict) -> None:
    """Human-readable category breakdown to stdout."""
    print("\nCPU category profile:")
    print("-" * 52)
    print(f"  {'Category':<16s} {'Time (s)':>10s} {'%':>7s} {'Calls':>12s}")
    print("-" * 52)
    for row in profile["categories"]:
        print(
            f"  {row['category']:<16s} {row['tottime']:>10.4f} "
            f"{row['pct']:>6.1f}% {int(row['calls']):>12,}"
        )
    print("-" * 52)
    print(f"  {'TOTAL':<16s} {profile['total_time']:>10.4f}")


def nsys_suggestion(n_features: int, n_samples: int, seed: int) -> str:
    """Return the recommended nsys command for GPU-side profiling."""
    return (
        "GPU kernel profiling is manual. Suggested Nsight Systems command:\n"
        f"  nsys profile -o ccc_gpu_f{n_features}_n{n_samples} \\\n"
        f'    python -c "import numpy as np; '
        f"from ccc.coef.impl_gpu import ccc; "
        f"np.random.seed({seed}); "
        f'ccc(np.random.rand({n_features}, {n_samples}))"\n'
        "For kernel-level metrics use: ncu --set full <same command>"
    )
