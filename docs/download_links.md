# Dataset Download Links & Setup

This document provides official download sources and local placement instructions for the three primary datasets used in this project.

## UK-DALE (UK Domestic Appliance-Level Electricity)

### Official Source
- **Provider**: University of Edinburgh, UK Energy Research Centre (UKERC)
- **Access**: [UKERC Energy Data Centre](https://ukerc.rl.ac.uk/DC/cgi-bin/edc_search.pl?GoButton=Detail&WantComp=46&EndNote=on)
- **Registration**: Required for download (free academic use)
- **License**: Academic use only, cite original paper

### File Details
- **Primary File**: `ukdale.h5` (HDF5 format)
- **Size**: ~6.3 GB (high-resolution household consumption data)
- **Format**: HDF5 with hierarchical structure (houses → meters → readings)
- **Alternative**: Individual CSV files per house (larger total size)

### Local Placement
```
data/raw/ukdale/
├── ukdale.h5                    # Main HDF5 file
├── metadata/                    # Appliance and house metadata
│   ├── meter_devices.csv
│   └── building_metadata.csv
└── README.txt                   # Original dataset documentation
```

---

## London Smart Meters (LCL Dataset)

### Official Source
- **Provider**: UK Power Networks (via London Datastore)
- **Access**: [London Datastore - Smart Meter Data](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- **Registration**: None required (open data)
- **License**: Open Government License

### File Details
- **Primary File**: `LCL-FullData.csv` or `CC_LCL-FullData.csv`
- **Size**: ~8.5 GB (CSV format, 30-minute resolution)
- **Format**: CSV with columns: LCLid, DateTime, KWH/hh (half-hourly), Acorn, Acorn_grouped
- **Compressed**: Often available as ZIP (~800 MB compressed)

### Local Placement
```
data/raw/lcl/
├── CC_LCL-FullData.csv          # Main consumption data
├── LCL-FullData.zip             # Compressed version (optional)
├── informations_households.csv  # Household metadata
└── weather_hourly_darksky.csv   # Weather data (if available)
```

---

## SSEN LV Feeder Data

### Official Source
- **Provider**: Scottish and Southern Electricity Networks (SSEN)
- **Access**: [SSEN Open Data Portal](https://data.ssen.co.uk)
- **Registration**: None required (open data)
- **License**: Open Government License - validation and research use

### File Details
This dataset provides REAL operational distribution network data:

**1. Metadata Lookup CSV**
- **File**: `ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv`
- **Size**: ~25 MB
- **Records**: 100,000 LV feeders with network hierarchy
- **Contains**: Feeder IDs, primary/secondary substations, HV feeder linkages, postcodes, customer counts (total_mpan_count)

**2. Time-Series Consumption CSV**
- **File**: `ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv`
- **Size**: ~12 MB
- **Records**: 100,000 half-hourly consumption readings (October 2025)
- **Contains**: Actual feeder loads (Wh), device counts, reactive power, timestamps
- **Coverage**: 28 unique feeders with operational measurements

### Local Placement
```
data/raw/ssen/
├── ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv    # Metadata
├── ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv      # Time-series
└── LV_FEEDER_LOOKUP.csv                                                 # Legacy (fallback)
```

---

## Data Tracking with DVC

**IMPORTANT: Track with DVC; never commit raw data to Git**

After downloading and placing datasets in their respective directories:

### 1. Track with DVC
```bash
# Track each dataset directory
dvc add data/raw/ukdale
dvc add data/raw/lcl
dvc add data/raw/ssen

# This creates .dvc pointer files (tracked by Git)
```

### 2. Commit pointers to Git
```bash
# Add DVC pointer files and metadata
git add data/raw/*.dvc dvc.lock .gitignore

# Commit (only pointers, not data)
git commit -m "DVC: track raw datasets via pointers"
```

### 3. Optional: Set up remote storage
```bash
# Configure remote storage (S3, Azure, etc.)
dvc remote add -d myremote s3://my-bucket/fyp-data/
# or: dvc remote add -d myremote azure://container/path/

# Push data to remote
dvc push
```

## Download Instructions

### UK-DALE
1. Register at UKERC Energy Data Centre
2. Download `ukdale.h5` (main file) and metadata CSVs
3. Place in `data/raw/ukdale/` as shown above
4. Run `dvc add data/raw/ukdale`

### London Smart Meters
1. Download directly from London Datastore (no registration)
2. Get `LCL-FullData.csv` or compressed version
3. Extract to `data/raw/lcl/` as shown above
4. Run `dvc add data/raw/lcl`

### SSEN Feeder Data
1. Visit [SSEN Open Data Portal](https://data.ssen.co.uk)
2. Download both CSV files:
   - `ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv`
   - `ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv`
3. Place in `data/raw/ssen/` as shown above
4. Run `dvc add data/raw/ssen`

## Size Expectations

| Dataset | Uncompressed | Compressed | Records |
|---------|-------------|------------|---------|
| UK-DALE | ~6.3 GB | ~2.1 GB | ~114M readings |
| LCL | ~8.5 GB | ~800 MB | ~167M readings |
| SSEN (Metadata) | ~25 MB | ~8 MB | ~100K feeders |
| SSEN (Time-series) | ~12 MB | ~4 MB | ~100K consumption readings |
| **Total** | **~14.8 GB** | **~2.9 GB** | **~281M records** |

## Access Notes

- **UK-DALE**: Requires academic registration; allow 1-2 days for approval
- **LCL**: Immediate download; largest file (consider downloading compressed version first)
- **SSEN**: Open data portal; immediate download of both CSV files (no registration)
- **All datasets**: Respect usage licenses and cite appropriately in publications

## Troubleshooting

### Download Issues
- **Large files**: Use wget/curl with resume capability for robust downloads
- **Network timeouts**: Download compressed versions when available
- **Access denied**: Verify registration status and license compliance

### DVC Issues
- **"File not found"**: Ensure directories are non-empty before `dvc add`
- **Git ignores .dvc files**: Check `.gitignore` has proper DVC exceptions
- **Slow operations**: Large datasets take time; use `--force` for re-tracking

For detailed DVC usage, see [`data/README_raw.md`](../data/README_raw.md).
