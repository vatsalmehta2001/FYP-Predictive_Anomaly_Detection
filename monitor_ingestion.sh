#!/bin/bash
# Monitor LCL ingestion progress

cd /Users/vatsalmehta/Developer/FYP-Predictive_Anomaly_Detection

clear
echo "=========================================="
echo "LCL INGESTION MONITOR"
echo "=========================================="
echo ""

# Check if process is running
if pgrep -f "python.*lcl.*ingestion" > /dev/null; then
    echo "Status: ✅ RUNNING"
else
    echo "Status: ⏹️  STOPPED (completed or failed)"
fi

echo ""
echo "Latest log entries:"
echo "------------------------------------------"
tail -5 lcl_ingestion_final.log
echo ""

echo "Files & Counts:"
echo "------------------------------------------"
poetry run python << 'EOF'
import polars as pl
from pathlib import Path
import datetime

lcl_dir = Path("data/processed/lcl_data")
if lcl_dir.exists():
    files = list(lcl_dir.glob("*.parquet"))
    df = pl.scan_parquet(str(lcl_dir / "*.parquet"))
    total = df.select(pl.len()).collect().item()

    # Get log count
    with open("lcl_ingestion_final.log") as f:
        for line in reversed(f.readlines()):
            if "Processed" in line and "records" in line:
                try:
                    processed = int(line.split("Processed ")[1].split(" ")[0])
                    break
                except:
                    pass

    progress_pct = (processed / 167_932_474) * 100

    print(f"Files: {len(files):,}")
    print(f"Log: {processed:,} records ({progress_pct:.1f}%)")
    print(f"Disk: {total:,} records")
    print(f"Match: {total/processed:.6f}x")

    if 0.999 < (total/processed) < 1.001:
        print("Status: ✅ PERFECT MATCH")
    else:
        print(f"Status: ⚠️ {abs((total/processed-1)*100):.2f}% discrepancy")
else:
    print("No data yet...")
EOF

echo ""
echo "=========================================="
echo "Press Ctrl+C to exit"
echo "Run: ./monitor_ingestion.sh"
echo "=========================================="
