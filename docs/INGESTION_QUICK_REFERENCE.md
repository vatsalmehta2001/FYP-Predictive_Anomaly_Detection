# Data Ingestion Quick Reference

## Quick Start

### First Time Setup
```bash
# Install dependencies
poetry install

# Make scripts executable (already done)
chmod +x scripts/*.sh scripts/*.py

# Check current status
python scripts/cleanup_and_verify.py --check-only
```

### Complete Fresh Ingestion
```bash
# Clean start (deletes existing processed data)
./scripts/run_complete_ingestion.sh --clean

# Expected duration:
# - LCL: 4-8 hours (8.54 GB, 167M records)
# - UK-DALE: 2-4 hours (6.33 GB, 114M records)
# Total: 6-12 hours depending on hardware
```

### Resume Interrupted Ingestion
```bash
# If laptop closed or process killed, just re-run
PYTHONPATH=$(pwd)/src python -m fyp.ingestion.cli lcl
PYTHONPATH=$(pwd)/src python -m fyp.ingestion.cli ukdale

# Pipeline automatically resumes from last checkpoint
```

---

## Verification Commands

### Check Dataset Status
```bash
python scripts/cleanup_and_verify.py --check-only
```

Expected output:
```
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Dataset    ┃ Status            ┃ Files ┃ Size(GB) ┃ Records(est.) ┃ Intervals ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ LCL        │ COMPLETE          │ 1,234 │ 8.54     │ 167,000,000   │ [30]      │
│ UK-DALE    │ COMPLETE (30-min) │ 456   │ 6.33     │ 114,000,000   │ [30]      │
│ SSEN       │ COMPLETE          │ 1     │ 0.04     │ 416,609       │ [-]       │
└────────────┴───────────────────┴───────┴──────────┴───────────────┴───────────┘
```

### Verify UK-DALE Intervals (Critical)
```bash
python scripts/verify_ukdale_intervals.py
```

### Run Validation Tests
```bash
# Quick validation
pytest tests/test_ingestion_complete.py -v

# Include slow quality tests
pytest tests/test_ingestion_complete.py -v --run-slow
```

---

## Cleanup Commands

### Clean Specific Dataset
```bash
# Check before cleanup
python scripts/cleanup_and_verify.py --check-only

# Clean LCL only
python scripts/cleanup_and_verify.py --cleanup lcl

# Clean UK-DALE only
python scripts/cleanup_and_verify.py --cleanup ukdale

# Clean all (with confirmation)
python scripts/cleanup_and_verify.py --cleanup all
```

### Force Cleanup (No Confirmation)
```bash
python scripts/cleanup_and_verify.py --cleanup all --force
```

---

## Troubleshooting

### Problem: LCL Stopped at 91%
**Solution:**
```bash
# Just re-run, it will resume from checkpoint
PYTHONPATH=$(pwd)/src python -m fyp.ingestion.cli lcl
```

### Problem: UK-DALE Has Wrong Intervals
**Symptom:** Intervals are not [30]

**Solution:**
```bash
# Clean and re-ingest
python scripts/cleanup_and_verify.py --cleanup ukdale --force
PYTHONPATH=$(pwd)/src python -m fyp.ingestion.cli ukdale
```

### Problem: Out of Memory
**Solution:**
```bash
# Close other applications
# Ingestion will automatically:
# - Monitor memory usage
# - Force garbage collection at >80%
# - Reduce batch size at >85%
```

### Problem: Disk Space Full
**Solution:**
```bash
# Check disk space
df -h .

# LCL needs ~9GB, UK-DALE needs ~7GB processed space
```

### Problem: Can't Find Raw Data
**Solution:**
```bash
# Check raw data locations
ls data/raw/lcl/
ls data/raw/ukdale/

# Should see:
# data/raw/lcl/CC_LCL-FullData.csv
# data/raw/ukdale/ukdale.h5
```

---

## Success Checklist

Before proceeding to notebooks/modeling:

- [ ] LCL ingestion 100% complete
- [ ] UK-DALE ingestion complete with 30-min data
- [ ] Both datasets verified: `python scripts/cleanup_and_verify.py --check-only`
- [ ] UK-DALE intervals verified: `python scripts/verify_ukdale_intervals.py`
- [ ] Tests passing: `pytest tests/test_ingestion_complete.py -v`
- [ ] Both datasets have matching schema
- [ ] Both datasets use UTC timezone
- [ ] Both datasets at 30-minute resolution

---

## Expected Statistics

### LCL (London Smart Meters)
- **Source size**: 8.54 GB (CSV)
- **Processed size**: ~9 GB (Parquet with compression)
- **Records**: 167,000,000
- **Households**: 5,567
- **Time period**: 2011-2014
- **Interval**: 30 minutes
- **Files**: ~1,200 Parquet files

### UK-DALE
- **Source size**: 6.33 GB (HDF5)
- **Processed size**: ~7 GB (Parquet, 30-min)
- **Records**: ~114,000,000 (30-min downsampled)
- **Households**: 5
- **Time period**: 2012-2017
- **Interval**: 30 minutes (downsampled from 1-6 seconds)
- **Files**: ~400 Parquet files

### SSEN
- **Size**: 36.7 MB
- **Records**: 416,609 feeder metadata
- **No time series** (metadata only)

---

## Current Action Plan

Given the current situation (LCL at 91%, stuck process):

1. **Kill the stuck process**:
   ```bash
   kill 97755
   ```

2. **Clean and restart WITH caffeinate**:
   ```bash
   python scripts/cleanup_and_verify.py --cleanup lcl --force
   ./scripts/run_complete_ingestion.sh --clean
   ```

3. **Monitor progress** (new terminal):
   ```bash
   # Watch file count
   watch -n 60 'find data/processed/dataset=lcl -name "*.parquet" | wc -l'
   ```

4. **Let it run overnight** (6-12 hours total)

5. **Verify completion** tomorrow:
   ```bash
   python scripts/cleanup_and_verify.py --check-only
   pytest tests/test_ingestion_complete.py -v
   ```

