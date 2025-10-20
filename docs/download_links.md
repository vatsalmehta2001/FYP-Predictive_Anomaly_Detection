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
- **Access**: Research partnership/collaboration required
- **Registration**: Formal data sharing agreement needed
- **License**: Restricted use - validation purposes only

### File Details
- **Primary File**: `LV_FEEDER_LOOKUP.csv` (feeder metadata)
- **Size**: ~35 MB (feeder characteristics and locations)
- **Format**: CSV with feeder identifiers, capacity, geographic info
- **Time Series**: Additional feeder load data (if provided, varies by agreement)

### Local Placement
```
data/raw/ssen/
├── LV_FEEDER_LOOKUP.csv         # Feeder metadata and characteristics
├── feeder_timeseries/           # Time series data (if available)
│   ├── feeder_001_2023.csv
│   ├── feeder_002_2023.csv
│   └── ...
└── documentation/               # Technical specifications
    └── data_dictionary.pdf
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
1. Establish research partnership with SSEN
2. Sign data sharing agreement
3. Receive `LV_FEEDER_LOOKUP.csv` and any time series files
4. Place in `data/raw/ssen/` as shown above
5. Run `dvc add data/raw/ssen`

## Size Expectations

| Dataset | Uncompressed | Compressed | Records |
|---------|-------------|------------|---------|
| UK-DALE | ~6.3 GB | ~2.1 GB | ~114M readings |
| LCL | ~8.5 GB | ~800 MB | ~167M readings |
| SSEN | ~35 MB | ~10 MB | ~50K feeders |
| **Total** | **~14.8 GB** | **~2.9 GB** | **~281M records** |

## Access Notes

- **UK-DALE**: Requires academic registration; allow 1-2 days for approval
- **LCL**: Immediate download; largest file (consider downloading compressed version first)
- **SSEN**: Requires formal partnership; contact SSEN research team for data sharing agreement
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
