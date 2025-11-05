# GPU vs CPU Benchmark Analysis

This directory contains an interactive Jupyter notebook for benchmarking GPU-accelerated CCC implementation against CPU versions.

- `./notebook` contains the notebook file to explore the benchmark results interactively.
- `./manuscript_results` contains the benchmark results and script used for the manuscript, colleted on commit 05f129dfa47ad801eff963b4189484c7c64bd28e

## Files

- `benchmark_analysis.ipynb` - Main benchmark notebook (input)
- `benchmark_analysis_output.ipynb` - Results notebook with outputs (created after execution)
- `run_benchmark.sh` - Script to run notebook in background using Papermill
- `check_status.sh` - Check execution status and view progress

## Quick Start

### Option 1: Interactive Mode (Jupyter)

Open and run the notebook interactively:

```bash
jupyter notebook benchmark_analysis.ipynb
```

### Option 2: Background Execution with Papermill (Recommended)

Execute the notebook in background and preserve all outputs:

```bash
cd analysis/03-manuscript/90_benchmarking/ccc-vs-cccgpu
./run_benchmark.sh
```

**Monitoring:**

```bash
# Quick status check (recommended)
./check_status.sh

# View live execution output
tail -f benchmark_execution_*.log

# Check if still running
ps aux | grep papermill

# View final results
cat benchmark_execution_*.log | tail -50
```

**After completion:**
- Open `benchmark_analysis_output.ipynb` to view all results and plots
- Original `benchmark_analysis.ipynb` remains unmodified
- Log file contains complete execution details

## Configuration

Edit the `CONFIG` dictionary in the notebook (Cell 4) to customize:

- `test_cases`: List of (n_features, n_samples) to test
- `n_cpu_cores`: List of CPU core counts to compare [6, 12, 24]
- `seed`: Random seed for reproducibility
- `contain_singletons`: Test with constant features

Example configuration:
```python
CONFIG = {
    'test_cases': [
        (10, 100),      # Quick validation
        (50, 500),
        (500, 1000),    # Standard benchmarks
        (1000, 1000),
        (2000, 1000),
        (4000, 1000),
        (6000, 1000),
        (8000, 1000),
        (10000, 1000),
        # Uncomment for larger tests:
        # (16000, 1000),
        # (20000, 1000),
    ],
    'n_cpu_cores': [6, 12, 24],  # Test different parallelization levels
    'seed': 42,
    'contain_singletons': False,
}
```

## Requirements

- Python 3.8+
- Jupyter notebook
- **Papermill** (`pip install papermill`)
- NumPy, Pandas, Matplotlib, Seaborn
- CuPy (for GPU support)
- CCC library (both CPU and GPU implementations)

Install papermill if needed:
```bash
pip install papermill
```
