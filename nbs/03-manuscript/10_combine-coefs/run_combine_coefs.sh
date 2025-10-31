#!/bin/bash

# Convenience script for running combine_coefs.py with common parameter combinations
# Usage: ./run_combine_coefs.sh [tissue_name] [gene_selection_strategy] [log_level]

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_TOOL="${SCRIPT_DIR}/combine_coefs.py"

# Default parameters
TISSUE="${1:-whole_blood}"
GENE_STRATEGY="${2:-var_pc_log2}"
LOG_LEVEL="${3:-INFO}"

# Check if CLI tool exists
if [[ ! -f "$CLI_TOOL" ]]; then
    echo "Error: CLI tool not found at $CLI_TOOL"
    exit 1
fi

# Make sure CLI tool is executable
chmod +x "$CLI_TOOL"

echo "Running combine coefficients tool with:"
echo "  Tissue: $TISSUE"
echo "  Gene Selection Strategy: $GENE_STRATEGY"
echo "  Log Level: $LOG_LEVEL"
echo ""

# Run the CLI tool
"$CLI_TOOL" \
    --gtex-tissue "$TISSUE" \
    --gene-selection-strategy "$GENE_STRATEGY" \
    --log-level "$LOG_LEVEL"

echo ""
echo "Combine coefficients completed successfully!"
echo "Check the logs directory for detailed output."
