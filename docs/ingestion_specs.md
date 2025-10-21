# Data Ingestion Specifications

This document describes the unified schema, ingestion processes, and quality standards for the three energy datasets.

## Unified Schema

All energy consumption data is transformed to a standardized schema stored as partitioned Parquet files:

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `dataset` | string | Source dataset identifier | ∈ {"lcl", "ukdale", "ssen"} |
| `entity_id` | string | Household or feeder identifier | Non-null, unique per dataset |
| `ts_utc` | timestamp | UTC timestamp (timezone-aware) | Aligned to interval boundaries |
| `interval_mins` | int8 | Reading interval in minutes | ∈ {1, 5, 10, 15, 30, 60} |
| `energy_kwh` | float32 | Energy consumption in kWh | ≥ 0, finite |
| `source` | string | File or API resource identifier | Non-null |
| `extras` | string | JSON metadata | Valid JSON object |

### Partitioning Strategy

Data is partitioned by `dataset`, `year`, and `month` for efficient querying:

```
data/processed/
├── dataset=lcl/
│   ├── year=2023/
│   │   ├── month=01/
│   │   │   └── part-*.parquet
│   │   └── month=02/
│   └── year=2024/
├── dataset=ukdale/
└── dataset=ssen/
```

## Timezone Normalization

All timestamps are converted to UTC to ensure consistency across datasets:

1. **LCL**: Source data in UK local time → UTC (handles BST/GMT transitions)
2. **UK-DALE**: UNIX timestamps → UTC datetime objects
3. **SSEN**: API timestamps assumed UTC (verified during ingestion)

### Interval Alignment

For 30-minute data, timestamps are aligned to :00 and :30 boundaries:
- `2023-01-01 12:15:00` → `2023-01-01 12:00:00`
- `2023-01-01 12:45:00` → `2023-01-01 12:30:00`

### DST (Daylight Saving Time) Handling

**Spring Transition (Clocks Forward)**:
- UK local time 01:00 → 02:00 BST (missing hour)
- UTC timestamps remain monotonic
- No data loss or duplication

**Fall Transition (Clocks Back)**:
- UK local time 02:00 BST → 01:00 GMT (repeated hour)
- Ambiguous local times resolved using DST context
- UTC conversion ensures no duplicate timestamps per entity

**Implementation**:
- LCL data: Parse local times with `Europe/London` timezone
- UK-DALE data: Already in UTC (no conversion needed)
- Validation: Ensure monotonic UTC timestamps per entity

## Dataset-Specific Details

### LCL (London Smart Meters)

**Source Format**: CSV with columns `LCLid,DateTime,KWH/hh,Acorn,Acorn_grouped`

**Transformations**:
- Parse DateTime with UK timezone awareness
- Validate 30-minute cadence
- Store Acorn demographics in `extras`

**CLI Usage**:
```bash
# Full dataset ingestion
python -m fyp.ingestion.cli lcl --input-root data/raw --output-root data/processed

# Sample-based testing
python -m fyp.ingestion.cli lcl --use-samples
```

### UK-DALE

**Source Format**: HDF5 hierarchical structure `/building{N}/elec/meter{M}/power/{active,apparent}`

**Energy Conversion**:
- Read instantaneous power (W) at native sampling rate from HDF5 metadata
- Calculate energy per sample: `Energy = Power × Δt` (not averaged!)
- For 30-min downsampling: sum energy within bins, don't average power
- Native resolution preserves original sampling intervals (1-6 seconds)

**Quality Assurance**:
- File SHA256 hash for provenance
- Missing data percentage calculation
- Duplicate timestamp detection and removal
- Outlier identification (>3σ)

**Entity ID Convention**:
- Aggregate: `house_1`
- Appliance: `house_1_dishwasher`

**Enhanced Extras**:
```json
{
  "channel": "dishwasher",
  "meter_id": 5,
  "source_uri": "ukdale.h5/building1/meter5",
  "ingestion_version": "v2.0",
  "sha256": "1a2b3c4d5e6f",
  "missing_pct": 2.1,
  "duplicates": 0
}
```

**CLI Usage**:
```bash
# With 30-minute downsampling (default)
python -m fyp.ingestion.cli ukdale --downsample-30min

# Native resolution only
python -m fyp.ingestion.cli ukdale --no-downsample-30min
```

### SSEN (Time-Series Consumption Data)

**Source Format**:
- CSV metadata: `ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv` (100K feeders)
- CSV time-series: `ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv` (100K consumption records)

**Processing Strategy**:
1. Load metadata lookup into fast dictionary (O(1) enrichment)
2. Read consumption CSV in 10K row chunks (memory-efficient)
3. Vectorized validation (filter missing fields in batch)
4. Enrich each record with feeder metadata (network hierarchy, customer counts)
5. Write to unified Parquet schema with 14 enriched fields

**Performance Optimizations**:
- Dictionary lookup instead of DataFrame scanning (45x speedup)
- Vectorized pandas operations for validation
- Timestamp conversion once per chunk (not per row)
- Processing speed: 4 seconds for 100K records

**Quality Assurance**:
- Vectorized validation for required fields (lv_feeder_id, timestamp, consumption)
- Skipped records logged with detailed counts
- Metadata enrichment success tracking
- Data quality: 99.97% valid records (33 skipped due to missing consumption)

**Enhanced Extras**:
```json
{
  "source_file": "ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv",
  "ingestion_version": "v2.1_timeseries",
  "device_count": 61,
  "reactive_kwh": 0.339,
  "primary_consumption_kwh": 6.64,
  "secondary_consumption_kwh": null,
  "dno_name": "Scottish and Southern Electricity Networks",
  "secondary_substation_id": "050",
  "lv_feeder_name": "ANSON DRIVE",
  "total_mpan_count": 152.0,
  "postcode": "EH11 3NF",
  "primary_substation_id": "64070",
  "primary_substation_name": "Gorgie",
  "secondary_substation_name": "ANSON DRIVE",
  "hv_feeder_id": "05",
  "hv_feeder_name": "Gorgie 05"
}
```

**CLI Usage**:
```bash
# Full time-series ingestion (100K records, ~4 seconds)
python -m fyp.ingestion.cli ssen --input-root data/raw --output-root data/processed

# Sample-based testing (30 real records)
python -m fyp.ingestion.cli ssen --use-samples
```

**Data Fields Preserved**:
- **Consumption**: total_consumption_active_import (Wh), reactive power (Wh)
- **Device Counts**: aggregated_device_count_active
- **Primary/Secondary Consumption**: Separate imports where available
- **Network Hierarchy**: DNO, primary/secondary substations, HV/LV feeders
- **Geography**: Postcodes, substation locations
- **Customers**: total_mpan_count for accurate LCL-to-feeder scaling

## Quality Validation

### Schema Validation (Pydantic)
- Type checking for all fields
- Range validation (non-negative energy, valid intervals)
- Timezone awareness enforcement

### Business Logic Validation
- Monotonic timestamps per entity
- Interval alignment verification
- Detection of unrealistic values (>100 kWh/30min for household)
- Gap detection and reporting

### Error Handling
- Invalid records logged and skipped
- Summary statistics tracked: processed, errors, skipped
- Graceful degradation on partial failures

## Running Ingestion

### Quick Test (CI/Development)
```bash
# All datasets with samples
python -m fyp.ingestion.cli lcl --use-samples
python -m fyp.ingestion.cli ukdale --use-samples
python -m fyp.ingestion.cli ssen --use-samples
```

### Full Production Run
```bash
# Ensure raw data is present
ls data/raw/lcl/CC_LCL-FullData.csv
ls data/raw/ukdale/ukdale.h5
ls data/raw/ssen/ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv
ls data/raw/ssen/ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv

# Run ingestion
python -m fyp.ingestion.cli lcl --input-root data/raw --output-root data/processed
python -m fyp.ingestion.cli ukdale --downsample-30min --input-root data/raw --output-root data/processed
python -m fyp.ingestion.cli ssen --input-root data/raw --output-root data/processed
```

### DVC Pipeline
```bash
# Run all ingestion stages
dvc repro ingest_lcl ingest_ukdale ingest_ssen

# Check outputs
ls data/processed/dataset=*/
```

## Troubleshooting

### Common Issues

**Timezone Errors**
- Symptom: "Timestamp must be timezone-aware"
- Fix: Source data may have naive timestamps; ingestion adds UK timezone

**Memory Issues (UK-DALE)**
- Symptom: OOM with large HDF5 files
- Fix: Ingestion samples data (every 60th reading by default)

**Missing SSEN CSV Files**
- Symptom: "No SSEN feeder lookup file found"
- Fix: Download both CSV files from SSEN open data portal (see download_links.md)

**Missing Dependencies**
- Symptom: Import errors
- Fix: Install with `poetry install` or `pip install h5py pandas pyarrow requests`

### Units Conversion

| Dataset | Source Unit | Target Unit | Conversion |
|---------|-------------|-------------|------------|
| LCL | kWh per 30min | kWh | Direct |
| UK-DALE | Watts | kWh | W × (interval_mins/60) / 1000 |
| SSEN | Wh per 30min | kWh | Wh / 1000 |

### Performance Tips

1. **Chunked Processing**:
   - LCL: 100k row chunks for large CSVs
   - UK-DALE: Streaming HDF5 reads to avoid memory issues
   - Parquet: 128MB row groups for optimal compression/query performance

2. **Memory Optimization**:
   - UK-DALE: Sample high-frequency data (every 60th reading) for manageable memory
   - Batch writes: Process 10k records at a time
   - Schema enforcement: Use PyArrow for type safety and compression

3. **SSEN Optimization**:
   - Dictionary-based metadata lookup (O(1) instead of O(n) DataFrame scans)
   - Vectorized pandas validation (filter entire chunks, not row-by-row)
   - Timestamp conversion per chunk (not per row)

4. **Parallel Processing**: Run dataset ingestions concurrently (different processes)

### Troubleshooting

**Energy Unit Issues**:
- **Symptom**: Unrealistic energy values (too high/low)
- **Cause**: Incorrect power-to-energy conversion
- **Fix**: Verify time intervals and unit conversion factors

**DST Timestamp Issues**:
- **Symptom**: Non-monotonic or duplicate timestamps
- **Cause**: Incorrect timezone handling during DST transitions
- **Fix**: Use `Europe/London` timezone for UK data, handle ambiguous times

**SSEN Data Quality**:
- **Symptom**: High number of skipped records
- **Cause**: Missing required fields (feeder_id, timestamp, or consumption)
- **Fix**: Check source CSV data quality; some skipped records are expected

**Memory Issues (UK-DALE)**:
- **Symptom**: Out of memory errors with large HDF5 files
- **Fix**: Increase sampling rate or process houses individually

## Output Verification

After successful ingestion:

```bash
# Check Parquet files
find data/processed -name "*.parquet" | head

# Verify schema
python -c "import pyarrow.parquet as pq; print(pq.read_schema('data/processed/dataset=lcl/year=2023/month=01/part-0.parquet'))"

# Read sample data
python -c "import pandas as pd; print(pd.read_parquet('data/processed/dataset=lcl').head())"

# Check summary
cat data/processed/ingestion_summary.json
```

## Next Steps

With ingested data in unified Parquet format:

1. **Feature Engineering**: Extract temporal, weather, and lag features
2. **Model Training**: Use partitioned data for efficient training
3. **Cross-Dataset Analysis**: Compare patterns across LCL, UK-DALE, and SSEN
4. **Anomaly Detection**: Leverage unified schema for consistent algorithms
