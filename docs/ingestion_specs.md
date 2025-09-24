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

### SSEN

**Source Format**: 
- CSV lookup: `LV_FEEDER_LOOKUP.csv` with feeder metadata
- CKAN API: Time series data via paginated JSON responses

**API Robustness**:
- Rate limiting with exponential backoff (429/5xx errors)
- HTTP response caching with ETag/Last-Modified support
- Automatic retry on transient failures (max 3 attempts)
- On-disk cache keyed by URL+headers for DVC idempotency

**Quality Assurance**:
- Resource ID tracking for API provenance
- HTTP cache timestamps and ETags
- Response validation and error logging
- Feeder metadata enrichment from lookup CSV

**Enhanced Extras**:
```json
{
  "feeder_name": "Main Street Primary",
  "substation": "WEST_SUB_01", 
  "postcode_sector": "EH1 2",
  "capacity_kva": 315.0,
  "source_uri": "api:resource_abc123",
  "resource_id": "abc123",
  "retrieved_at": "2023-01-01T12:00:00Z",
  "ingestion_version": "v2.0"
}
```

**API Configuration**:
```bash
# Environment variables
export SSEN_CKAN_URL="https://data.ssen.co.uk"
export SSEN_API_KEY="your-api-key"  # Optional

# CLI usage with caching
python -m fyp.ingestion.cli ssen --ckan-url $SSEN_CKAN_URL

# Force refresh cached responses
python -m fyp.ingestion.cli ssen --force-refresh
```

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
ls data/raw/lcl/*.csv
ls data/raw/ukdale/*.h5
ls data/raw/ssen/LV_FEEDER_LOOKUP.csv

# Run ingestion
python -m fyp.ingestion.cli lcl
python -m fyp.ingestion.cli ukdale --downsample-30min
python -m fyp.ingestion.cli ssen  # Requires network for API
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

**API Rate Limiting (SSEN)**
- Symptom: HTTP 429 errors
- Fix: Adjust `--rate-limit` parameter (default 1.0 seconds)

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

3. **Caching Strategy**:
   - SSEN API: HTTP response caching with ETag/Last-Modified
   - Rate limiting: 1-second delays with exponential backoff
   - Cache invalidation: `--force-refresh` flag for fresh data

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

**API Rate Limiting**:
- **Symptom**: HTTP 429 errors from SSEN API
- **Fix**: Increase rate limit delay: `--rate-limit 2.0`

**Memory Issues (UK-DALE)**:
- **Symptom**: Out of memory errors with large HDF5 files
- **Fix**: Increase sampling rate or process houses individually

**Cache Corruption**:
- **Symptom**: JSON decode errors from cached responses
- **Fix**: Clear cache directory or use `--force-refresh`

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
