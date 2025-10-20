#!/bin/bash
# Quick progress check for running ingestion
#
# Usage: ./scripts/check_progress.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== LCL Ingestion Progress ==="
echo ""

# Check if process is running
if pgrep -f "caffeinate.*ingestion" > /dev/null; then
    echo "Status: RUNNING"
    echo ""

    # Get latest log lines
    echo "Latest progress:"
    tail -3 ingestion.log | grep "Processed"

    echo ""
    echo "Files created:"
    find data/processed/dataset=lcl -name "*.parquet" 2>/dev/null | wc -l

    echo ""
    echo "Current size:"
    du -sh data/processed/dataset=lcl 2>/dev/null || echo "0 (just started)"

    echo ""
    echo "Running since:"
    ps -p $(pgrep -f "caffeinate.*ingestion" | head -1) -o etime | tail -1

else
    echo "Status: NOT RUNNING (either completed or failed)"
    echo ""
    echo "Check log for status:"
    echo "  tail -50 ingestion.log"
    echo ""
    echo "Or verify completion:"
    echo "  poetry run python scripts/cleanup_and_verify.py --check-only"
fi
