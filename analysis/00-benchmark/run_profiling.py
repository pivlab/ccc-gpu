#!/usr/bin/env python
"""Quick profiling script to verify ARI computation percentage."""

import sys
import os

# Add libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'libs'))

import cProfile
import pstats
import numpy as np
from ccc.coef.impl import ccc


def profile_workload(n_features, n_samples, name):
    """Profile a single workload."""
    np.random.seed(42)
    data = np.random.rand(n_features, n_samples)
    n_pairs = (n_features * (n_features - 1)) // 2

    profiler = cProfile.Profile()
    profiler.enable()
    result = ccc(data, n_jobs=1)
    profiler.disable()

    stats = pstats.Stats(profiler)

    total_time = 0
    ari_time = 0
    ari_calls = 0
    contingency_time = 0
    pair_conf_time = 0

    for key, value in stats.stats.items():
        filename, line_num, func_name = key
        ncalls, tottime, cumtime = value[0], value[2], value[3]

        if func_name == 'ccc' and 'impl.py' in filename:
            total_time = cumtime
        if func_name == 'adjusted_rand_index':
            ari_time = tottime
            ari_calls = ncalls
        if func_name == 'get_contingency_matrix':
            contingency_time = tottime
        if func_name == 'get_pair_confusion_matrix':
            pair_conf_time = tottime

    total_ari = ari_time + contingency_time + pair_conf_time
    ari_pct = (total_ari/total_time)*100 if total_time > 0 else 0

    print(f'=== {name} ({n_features}x{n_samples}, {n_pairs} pairs) ===')
    print(f'Total time: {total_time:.4f}s')
    print(f'ARI time: {total_ari:.4f}s ({ari_pct:.1f}%)')
    print(f'ARI calls: {ari_calls}')
    print()

    return {
        'name': name,
        'n_features': n_features,
        'n_samples': n_samples,
        'n_pairs': n_pairs,
        'total_time': total_time,
        'ari_time': total_ari,
        'ari_pct': ari_pct,
        'ari_calls': ari_calls
    }


if __name__ == '__main__':
    print("CCC Profiling: ARI Computation Percentage")
    print("=" * 50)
    print()

    results = []

    # Run workloads
    results.append(profile_workload(10, 100, 'Small'))
    results.append(profile_workload(50, 500, 'Medium'))
    results.append(profile_workload(100, 1000, 'Large'))

    # Summary
    avg_ari_pct = sum(r['ari_pct'] for r in results) / len(results)
    print("=" * 50)
    print(f"SUMMARY: Average ARI percentage: {avg_ari_pct:.1f}%")
    print("=" * 50)
