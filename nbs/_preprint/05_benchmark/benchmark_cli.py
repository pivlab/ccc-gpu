"""
Script to benchmark different correlation methods (CCC, Pearson, Spearman) using simulated data.
This script generates random data of different shapes and measures the performance of each method.
"""

import logging
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from typing import Callable, Tuple, List, Dict
from datetime import datetime
import sys
import argparse
from ccc.corr import ccc_gpu, pearson, spearman

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_DIR = SCRIPT_DIR / "logs"  # Directory for log files

# Benchmark configurations
BENCHMARK_SHAPES = [
    # Benchmark cases
    (500, 1000),
    (1000, 1000),
    (2000, 1000),
    (4000, 1000),
    (6000, 1000),
    (8000, 1000),
    (10000, 1000),
    (16000, 1000),
    (20000, 1000),
]

# Configuration constants
N_CPU_CORES = 24


def setup_logging(log_dir: Path = None) -> None:
    """Configure logging to write to both file and stdout.

    Args:
        log_dir: Directory to store log files. If None, logs will only go to stdout.
    """
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers = []

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_dir is provided
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"correlation_benchmark_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Log file created at: {log_file}")


def get_correlation_method(method_name: str) -> Callable:
    """Get the correlation method function based on the method name.

    Args:
        method_name: Name of the correlation method ('ccc_gpu', 'pearson', or 'spearman')

    Returns:
        Callable: The correlation method function

    Raises:
        ValueError: If method_name is not one of the supported methods
    """
    method_map = {
        "ccc_gpu": lambda x: ccc_gpu(x, n_jobs=N_CPU_CORES),
        "pearson": pearson,
        "spearman": spearman,
    }

    if method_name not in method_map:
        raise ValueError(
            f"Unsupported correlation method: {method_name}. "
            f"Supported methods are: {', '.join(method_map.keys())}"
        )

    method = method_map[method_name]
    method.__name__ = method_name
    return method


def generate_random_data(shape: Tuple[int, int]) -> pd.DataFrame:
    """Generate random data for benchmarking.

    Args:
        shape: Tuple of (n_features, n_objects) for the data shape

    Returns:
        DataFrame containing random data
    """
    return pd.DataFrame(np.random.rand(*shape))


def benchmark_correlation(
    data: pd.DataFrame, method: Callable, method_name: str
) -> dict:
    """Benchmark a single correlation method.

    Args:
        data: Input DataFrame
        method: Correlation method function
        method_name: Name of the correlation method

    Returns:
        Dictionary containing benchmark results
    """
    start_time = time()
    result = method(data)
    elapsed_time = time() - start_time

    # Calculate memory usage of the result
    result_memory = result.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB

    return {
        "method": method_name,
        "shape": data.shape,
        "time_seconds": elapsed_time,
        "result_memory_mb": result_memory,
        "result_shape": result.shape,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Benchmark correlation methods using simulated data"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["ccc_gpu", "pearson", "spearman"],
        choices=["ccc_gpu", "pearson", "spearman"],
        help="Correlation methods to benchmark",
    )
    return parser.parse_args()


def run_benchmark_suite(
    shapes: List[Tuple[int, int]], methods: List[str]
) -> List[Dict]:
    """Run the benchmark suite for all combinations of shapes and methods.

    Args:
        shapes: List of (n_features, n_objects) tuples to benchmark
        methods: List of correlation methods to benchmark

    Returns:
        List of benchmark results
    """
    all_results = []

    for shape in shapes:
        logging.info(f"\n{'='*50}")
        logging.info(f"Benchmarking shape: {shape}")
        logging.info(f"{'='*50}")

        # Generate data for this shape
        data = generate_random_data(shape)
        logging.info(f"Generated random data with shape: {shape}")

        # Benchmark each method
        for method_name in methods:
            try:
                method = get_correlation_method(method_name)
                result = benchmark_correlation(data, method, method_name)
                all_results.append(result)

                logging.info(
                    f"Method: {method_name}\n"
                    f"  Data shape: {result['shape']}\n"
                    f"  Time taken: {result['time_seconds']:.2f} seconds\n"
                    f"  Result memory: {result['result_memory_mb']:.2f} MB\n"
                    f"  Result shape: {result['result_shape']}"
                )

            except Exception as e:
                logging.error(f"Error benchmarking {method_name}: {str(e)}")
                continue

    return all_results


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup logging
        setup_logging(LOG_DIR)
        logging.info("Starting correlation methods benchmark")
        logging.info(f"Benchmark shapes: {BENCHMARK_SHAPES}")
        logging.info(f"Methods to benchmark: {args.methods}")

        # Run benchmark suite
        results = run_benchmark_suite(BENCHMARK_SHAPES, args.methods)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = LOG_DIR / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        logging.info(f"Saved benchmark results to: {output_file}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
