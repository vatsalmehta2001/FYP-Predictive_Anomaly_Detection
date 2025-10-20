#!/bin/bash
# Quick progress check for LCL ingestion

cd /Users/vatsalmehta/Developer/FYP-Predictive_Anomaly_Detection

clear
echo "=========================================="
echo "LCL INGESTION PROGRESS"
echo "=========================================="
echo ""

# Check if running
if pgrep -f "python.*lcl" > /dev/null 2>&1; then
    echo "Status: ✅ RUNNING"
else
    echo "Status: ⏹️  STOPPED/COMPLETED"
fi

echo ""
echo "Latest progress:"
tail -3 lcl_ingestion.log 2>/dev/null || echo "No log yet"
echo ""

poetry run python << 'EOF'
import polars as pl
from pathlib import Path
import datetime

lcl_dir = Path("data/processed/lcl_data")
if lcl_dir.exists() and list(lcl_dir.glob("*.parquet")):
    files = list(lcl_dir.glob("*.parquet"))
    df = pl.scan_parquet(str(lcl_dir / "*.parquet"))
    total = df.select(pl.len()).collect().item()

    # Get log count
    with open("lcl_ingestion.log") as f:
        for line in reversed(f.readlines()):
            if "Processed" in line:
                try:
                    processed = int(line.split("Processed ")[1].split(" ")[0])
                    break
                except:
                    pass

    progress = (processed / 167_932_474) * 100
    ratio = total / processed

    print(f"Files: {len(files):,}")
    print(f"Log: {processed:,} ({progress:.1f}%)")
    print(f"Disk: {total:,}")
    print(f"Match: {ratio:.6f}x", end=" ")

    if 0.999 < ratio < 1.001:
        print("✅")
    else:
        print(f"⚠️")
else:
    print("No data yet...")
EOF

echo ""
echo "=========================================="
echo "Commands:"
echo "  Watch live: tail -f lcl_ingestion.log"
echo "  This script: ./check_progress.sh"
echo "=========================================="
