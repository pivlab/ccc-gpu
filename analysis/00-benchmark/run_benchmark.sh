#!/bin/bash
#
# Run CCC benchmark notebook using papermill
#
# Usage:
#   ./run_benchmark.sh
#
# Prerequisites:
#   conda activate ccc-gpu-benchmark
#   pip install papermill
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK="$SCRIPT_DIR/benchmark_ccc_profiling.ipynb"
OUTPUT_NOTEBOOK="$SCRIPT_DIR/benchmark_ccc_profiling_executed.ipynb"

echo "========================================"
echo "CCC Benchmark Runner"
echo "========================================"
echo ""
echo "Input:  $NOTEBOOK"
echo "Output: $OUTPUT_NOTEBOOK"
echo ""

# Check if papermill is installed
if ! command -v papermill &> /dev/null; then
    echo "Error: papermill is not installed."
    echo "Install it with: pip install papermill"
    exit 1
fi

# Check if notebook exists
if [ ! -f "$NOTEBOOK" ]; then
    echo "Error: Notebook not found: $NOTEBOOK"
    exit 1
fi

echo "Starting benchmark (this may take a while)..."
echo ""

# Run the notebook with papermill
papermill "$NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    --cwd "$SCRIPT_DIR" \
    --progress-bar

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "========================================"
echo ""
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo ""
echo "Generated figures:"
ls -la "$SCRIPT_DIR"/*.png 2>/dev/null || echo "  (no figures found)"
