#!/bin/bash
#
# Background Notebook Execution Script (Papermill)
# Runs benchmark_analysis.ipynb with output preservation and completion notification
#

# Configuration
INPUT_NOTEBOOK="benchmark_analysis.ipynb"
OUTPUT_NOTEBOOK="benchmark_analysis_output.ipynb"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="benchmark_execution_${TIMESTAMP}.log"
PIDFILE="benchmark_execution.pid"

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GPU Benchmark Execution Script${NC}"
echo -e "${GREEN}Using Papermill${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if input notebook exists
if [ ! -f "$INPUT_NOTEBOOK" ]; then
    echo -e "${RED}Error: Notebook '$INPUT_NOTEBOOK' not found!${NC}"
    exit 1
fi

# Check if papermill is available
if ! command -v papermill &> /dev/null; then
    echo -e "${RED}Error: papermill command not found!${NC}"
    echo "Please install papermill: pip install papermill"
    exit 1
fi

# Check if already running
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Benchmark appears to be already running (PID: $OLD_PID)${NC}"
        echo "Check with: tail -f $(ls -t benchmark_execution_*.log | head -1)"
        echo "Kill with: kill $OLD_PID"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$PIDFILE"
    fi
fi

# Start execution in background with nohup
echo -e "${YELLOW}Starting benchmark execution in background...${NC}"
echo ""
echo "Input notebook:  $INPUT_NOTEBOOK"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Log file:        $LOGFILE"
echo "PID file:        $PIDFILE"
echo ""

# Create a temporary script to run in background
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'EOFSCRIPT'
#!/bin/bash

INPUT_NOTEBOOK="benchmark_analysis.ipynb"
OUTPUT_NOTEBOOK="benchmark_analysis_output.ipynb"
LOGFILE="$1"

echo "========================================" >> "$LOGFILE"
echo "Benchmark Execution Started: $(date)" >> "$LOGFILE"
echo "Input:  $INPUT_NOTEBOOK" >> "$LOGFILE"
echo "Output: $OUTPUT_NOTEBOOK" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

# Execute notebook with papermill
papermill \
    "$INPUT_NOTEBOOK" \
    "$OUTPUT_NOTEBOOK" \
    --log-output \
    --log-level INFO \
    --cwd "$(pwd)" >> "$LOGFILE" 2>&1

EXIT_CODE=$?

echo "" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"
echo "Benchmark Execution Completed: $(date)" >> "$LOGFILE"
echo "Exit Code: $EXIT_CODE" >> "$LOGFILE"
echo "========================================" >> "$LOGFILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Benchmark completed successfully!" >> "$LOGFILE"
    echo "Results saved to: $OUTPUT_NOTEBOOK" >> "$LOGFILE"

    if command -v notify-send &> /dev/null; then
        notify-send "GPU Benchmark Complete" "Results saved to $OUTPUT_NOTEBOOK" 2>/dev/null || true
    fi
else
    echo "✗ ERROR: Benchmark execution failed with exit code $EXIT_CODE" >> "$LOGFILE"

    if command -v notify-send &> /dev/null; then
        notify-send "GPU Benchmark Failed" "Check log file: $1" -u critical 2>/dev/null || true
    fi
fi

rm -f benchmark_execution.pid
rm -f "$0"  # Remove this temporary script

exit $EXIT_CODE
EOFSCRIPT

chmod +x "$TEMP_SCRIPT"

# Run in background with nohup
nohup "$TEMP_SCRIPT" "$LOGFILE" &
PID=$!

# Save PID
echo $PID > "$PIDFILE"

echo -e "${GREEN}✓ Background execution started!${NC}"
echo ""
echo "Process ID: $PID"
echo ""
echo -e "${YELLOW}Monitoring commands:${NC}"
echo "  Quick status:       ./check_status.sh"
echo "  View live output:   tail -f $LOGFILE"
echo "  Check if running:   ps -p $PID"
echo "  Stop execution:     kill $PID"
echo "  View final result:  cat $LOGFILE | tail -30"
echo ""
echo -e "${GREEN}The notebook will continue running even if you close this terminal.${NC}"
echo -e "${GREEN}Results will be saved to '$OUTPUT_NOTEBOOK' when complete.${NC}"
echo ""

# Optional: Start tailing the log file
read -p "Start monitoring log output now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}Monitoring log (Press Ctrl+C to stop monitoring, execution continues)...${NC}"
    echo ""
    sleep 1
    tail -f "$LOGFILE"
fi
