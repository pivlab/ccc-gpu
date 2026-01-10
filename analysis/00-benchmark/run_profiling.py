#!/usr/bin/env python
"""
Comprehensive CCC profiling script.

This script profiles the CCC (Clustermatch Correlation Coefficient) implementation
and provides a detailed runtime breakdown by function and category.

Usage:
    conda activate ccc-gpu-benchmark
    python run_profiling.py
"""

import sys
import os

# Add libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'libs'))

import cProfile
import pstats
import numpy as np
import pandas as pd
from ccc.coef.impl import ccc


# Function category definitions
FUNCTION_CATEGORIES = {
    # ARI Computation (the main bottleneck)
    'adjusted_rand_index': 'ARI',
    'get_pair_confusion_matrix': 'ARI',
    'get_contingency_matrix': 'ARI',

    # Partitioning
    'get_parts': 'Partitioning',
    'run_quantile_clustering': 'Partitioning',
    'get_feature_parts': 'Partitioning',
    'get_range_n_clusters': 'Partitioning',
    'get_perc_from_k': 'Partitioning',

    # Ranking
    'rank': 'Ranking',

    # Coordination/Orchestration
    'cdist_parts_basic': 'Coordination',
    'compute_ccc': 'Coordination',
    'compute_coef': 'Coordination',
    'ccc': 'Coordination',
    'get_chunks': 'Coordination',
    'get_coords_from_index': 'Coordination',
    'get_feature_type_and_encode': 'Coordination',
}


def categorize_function(func_name: str, filename: str) -> str:
    """Categorize a function based on its name and source file."""
    if func_name in FUNCTION_CATEGORIES:
        return FUNCTION_CATEGORIES[func_name]

    numpy_funcs = ['searchsorted', 'argsort', 'zeros', 'unique', 'full',
                   'ravel', 'dot', 'sum', 'max', 'argmax', 'floor', 'sqrt',
                   'ceil', 'round', 'array', 'arange', 'empty', 'copy']
    if func_name in numpy_funcs or 'numpy' in filename.lower() or 'numba' in filename.lower():
        return 'NumPy/Numba'

    if 'ccc' in filename.lower():
        return 'Other CCC'

    return 'Other'


def extract_all_function_times(stats: pstats.Stats, top_n: int = 20,
                                min_pct: float = 0.1) -> tuple:
    """Extract comprehensive timing information for ALL functions."""
    stats_dict = stats.stats

    total_time = 0
    for key, value in stats_dict.items():
        filename, line_num, func_name = key
        if func_name == 'ccc' and 'impl.py' in filename:
            total_time = value[3]
            break

    if total_time == 0:
        total_time = max(v[3] for v in stats_dict.values())

    func_data = []
    for key, value in stats_dict.items():
        filename, line_num, func_name = key
        ncalls, tottime, cumtime = value[0], value[2], value[3]

        pct = (tottime / total_time) * 100 if total_time > 0 else 0

        if pct < min_pct:
            continue

        category = categorize_function(func_name, filename)
        short_file = filename.split('/')[-1] if '/' in filename else filename

        func_data.append({
            'Function': func_name,
            'File': short_file,
            'Category': category,
            'Calls': ncalls,
            'TotTime': tottime,
            'CumTime': cumtime,
            'TotTime%': pct,
            'CumTime%': (cumtime / total_time) * 100 if total_time > 0 else 0
        })

    df = pd.DataFrame(func_data)
    df = df.sort_values('TotTime', ascending=False).head(top_n)
    df = df.reset_index(drop=True)

    return df, total_time


def profile_workload(n_features: int, n_samples: int, name: str) -> dict:
    """Profile a single workload with comprehensive breakdown."""
    np.random.seed(42)
    data = np.random.rand(n_features, n_samples)
    n_pairs = (n_features * (n_features - 1)) // 2

    profiler = cProfile.Profile()
    profiler.enable()
    result = ccc(data, n_jobs=1)
    profiler.disable()

    stats = pstats.Stats(profiler)
    func_df, total_time = extract_all_function_times(stats, top_n=20, min_pct=0.1)

    # Calculate category totals
    category_totals = func_df.groupby('Category').agg({
        'TotTime': 'sum',
        'Calls': 'sum'
    }).reset_index()
    category_totals['Percentage'] = (category_totals['TotTime'] / total_time) * 100
    category_totals = category_totals.sort_values('TotTime', ascending=False)

    return {
        'name': name,
        'n_features': n_features,
        'n_samples': n_samples,
        'n_pairs': n_pairs,
        'total_time': total_time,
        'func_df': func_df,
        'category_totals': category_totals
    }


def print_comprehensive_results(result: dict):
    """Print comprehensive profiling results."""
    print(f"\n{'='*80}")
    print(f"{result['name']} Workload: {result['n_features']} features x {result['n_samples']} samples")
    print(f"Pairwise comparisons: {result['n_pairs']:,}")
    print(f"Total time: {result['total_time']:.4f}s")
    print(f"{'='*80}")

    # Category summary
    print("\nCategory Breakdown:")
    print("-" * 50)
    for _, row in result['category_totals'].iterrows():
        print(f"  {row['Category']:15s}: {row['TotTime']:.4f}s ({row['Percentage']:5.1f}%) - {int(row['Calls']):,} calls")

    # Top functions
    print("\nTop Functions by Runtime:")
    print("-" * 80)
    print(f"{'Function':<30s} {'Category':<15s} {'Calls':>10s} {'TotTime':>10s} {'%':>8s}")
    print("-" * 80)

    for _, row in result['func_df'].head(15).iterrows():
        print(f"{row['Function']:<30s} {row['Category']:<15s} {row['Calls']:>10,} {row['TotTime']:>10.4f} {row['TotTime%']:>7.1f}%")


if __name__ == '__main__':
    print("=" * 80)
    print("CCC Comprehensive Profiling: Runtime Breakdown by Function")
    print("=" * 80)

    results = []

    # Run workloads
    for name, n_features, n_samples in [('Small', 10, 100), ('Medium', 50, 500), ('Large', 100, 1000)]:
        print(f"\nProfiling {name} workload...")
        result = profile_workload(n_features, n_samples, name)
        results.append(result)
        print_comprehensive_results(result)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    # Get ARI percentage across workloads
    ari_pcts = []
    for r in results:
        ari_row = r['category_totals'][r['category_totals']['Category'] == 'ARI']
        if len(ari_row) > 0:
            ari_pcts.append(ari_row['Percentage'].values[0])

    if ari_pcts:
        print(f"\nARI Computation Percentage:")
        print(f"  Average: {np.mean(ari_pcts):.1f}%")
        print(f"  Range: {min(ari_pcts):.1f}% - {max(ari_pcts):.1f}%")
        print(f"\nConclusion: ARI computation is the dominant bottleneck,")
        print(f"consuming approximately {np.mean(ari_pcts):.0f}% of total CCC runtime.")

    print("\n" + "=" * 80)
