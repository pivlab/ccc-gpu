#!/bin/bash
# Script to list and display information about built wheels

# Project root directory (assumes script is in scripts/pypi/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DIST_DIR="dist"

echo "========================================================================="
echo "Built Wheels Summary"
echo "========================================================================="
echo ""

# Check if dist directory exists
if [ ! -d "$DIST_DIR" ]; then
    echo "✗ Error: dist/ directory not found"
    echo "  No wheels have been built yet"
    echo ""
    echo "  To build wheels, run:"
    echo "    ./scripts/pypi/01-build-multi-python-conda.sh  (fast, local)"
    echo "    ./scripts/pypi/02-build-with-cibuildwheel.sh   (production)"
    exit 1
fi

# Find all wheel files
WHEELS=("$DIST_DIR"/*.whl)

# Check if any wheels exist
if [ ! -e "${WHEELS[0]}" ]; then
    echo "✗ No wheel files found in dist/"
    echo ""
    echo "  To build wheels, run:"
    echo "    ./scripts/pypi/01-build-multi-python-conda.sh  (fast, local)"
    echo "    ./scripts/pypi/02-build-with-cibuildwheel.sh   (production)"
    exit 1
fi

# Count wheels
WHEEL_COUNT=${#WHEELS[@]}
echo "Total wheels found: ${WHEEL_COUNT}"
echo ""

# Print table header
printf "%-55s %12s %10s\n" "Filename" "Python" "Size"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Parse and display each wheel
for WHEEL in "${WHEELS[@]}"; do
    FILENAME=$(basename "$WHEEL")

    # Parse wheel filename: {distribution}-{version}-{python}-{abi}-{platform}.whl
    # Example: cccgpu-0.2.1-cp310-cp310-linux_x86_64.whl

    # Extract Python version (e.g., cp310 -> 3.10)
    if [[ $FILENAME =~ cp([0-9]+)- ]]; then
        PY_TAG="${BASH_REMATCH[1]}"
        PY_MAJOR="${PY_TAG:0:1}"
        PY_MINOR="${PY_TAG:1}"
        PY_VERSION="${PY_MAJOR}.${PY_MINOR}"
    else
        PY_VERSION="unknown"
    fi

    # Get file size in human-readable format
    if command -v numfmt &> /dev/null; then
        SIZE=$(stat -c%s "$WHEEL" | numfmt --to=iec-i --suffix=B)
    else
        # Fallback if numfmt not available
        SIZE=$(du -h "$WHEEL" | cut -f1)
    fi

    # Print row
    printf "%-55s %12s %10s\n" "$FILENAME" "$PY_VERSION" "$SIZE"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for manylinux wheels
MANYLINUX_COUNT=$(ls "$DIST_DIR"/*manylinux*.whl 2>/dev/null | wc -l)
LINUX_COUNT=$(ls "$DIST_DIR"/*linux*.whl 2>/dev/null | wc -l)
LOCAL_COUNT=$((LINUX_COUNT - MANYLINUX_COUNT))

echo "Wheel types:"
if [ $MANYLINUX_COUNT -gt 0 ]; then
    echo "  • manylinux wheels: ${MANYLINUX_COUNT} (portable, PyPI-ready)"
fi
if [ $LOCAL_COUNT -gt 0 ]; then
    echo "  • local linux wheels: ${LOCAL_COUNT} (may not be portable)"
fi
echo ""

# Validate wheels if wheel tool is available
if command -v wheel &> /dev/null; then
    echo "Validating wheels..."
    ALL_VALID=true
    for WHEEL in "${WHEELS[@]}"; do
        if ! wheel unpack --quiet "$WHEEL" &> /dev/null; then
            echo "  ✗ ${FILENAME}: Invalid wheel"
            ALL_VALID=false
        fi
    done
    if $ALL_VALID; then
        echo "  ✓ All wheels are valid"
    fi
    echo ""
fi

# Display total size
TOTAL_SIZE=$(du -sh "$DIST_DIR" | cut -f1)
echo "Total size: ${TOTAL_SIZE}"
echo ""

echo "========================================================================="
echo "Next steps:"
echo "  1. Test wheels manually (see scripts/pypi/README.md)"
echo "  2. Upload to test PyPI: ./scripts/pypi/10-upload_to_test_pypi.sh"
echo "========================================================================="