# Datasets Overview

This document provides detailed information about the three primary datasets used in this project for energy forecasting and anomaly detection research.

## UK-DALE (UK Domestic Appliance-Level Electricity)

### Description
UK-DALE is a comprehensive dataset containing electricity consumption measurements from UK domestic buildings, recorded at both the household and individual appliance levels. The dataset provides high-resolution power measurements that enable detailed analysis of residential energy consumption patterns.

### Key Characteristics
- **Temporal Coverage**: 2012-2015 (varies by house)
- **Spatial Coverage**: 5 houses across UK
- **Resolution**: 1-second to 1-minute intervals (will be resampled to 30-minute)
- **Measurements**: Real power (Watts), apparent power, voltage, current
- **Appliance Coverage**: ~50 appliance types including heating, lighting, computing, kitchen appliances

### Usage in This Project
- **Primary Role**: Household-level energy forecasting model training
- **Self-Play Training**: Ground truth for verifying proposed consumption scenarios
- **Feature Engineering**: Appliance-level patterns for enhanced forecasting accuracy
- **Anomaly Detection**: Training models to identify unusual consumption patterns

### Data Processing Notes
- **Native Format**: CSV files with half-hourly consumption data
- **Ingestion**: Converted to unified Parquet schema with UTC timestamps
- **Resolution**: Native 30-minute intervals preserved
- **Metadata**: Acorn demographic groups stored in extras field
- **Quality**: Missing data periods logged, invalid readings skipped

---

## London Smart Meters (LCL Dataset)

### Description
The Low Carbon London (LCL) dataset contains smart meter electricity consumption data from approximately 5,567 London households collected as part of the UK Power Networks' Low Carbon London trial. This dataset provides valuable insights into urban residential electricity consumption patterns.

### Key Characteristics
- **Temporal Coverage**: 2011-2014
- **Spatial Coverage**: Greater London area
- **Resolution**: 30-minute intervals (native resolution - perfect for our needs)
- **Households**: ~5,567 residential customers
- **Additional Data**: Some households include time-of-use tariff information
- **Weather Data**: Accompanying weather measurements from London

### Usage in This Project
- **Pseudo-Feeder Construction**: Aggregating multiple households to simulate distribution feeder loads
- **Population Validation**: Large sample size enables robust statistical validation
- **Urban Pattern Analysis**: Understanding city-specific consumption characteristics
- **Cross-Dataset Validation**: Comparing patterns with UK-DALE for model generalization

### Data Processing Notes
- **Native Format**: HDF5 hierarchical structure with high-resolution power data
- **Ingestion**: Dual output - native resolution and 30-minute downsampled
- **Resolution**: 1-6 second native, optional 30-minute aggregation
- **Metadata**: Appliance types and meter IDs preserved in extras
- **Entity Mapping**: Separate entity IDs for aggregate and appliance-level data

---

## SSEN LV Feeder Data

### Description
Scottish and Southern Electricity Networks (SSEN) Low Voltage (LV) feeder data provides REAL distribution network measurements from operational electricity networks. This dataset represents actual aggregated load patterns at the distribution transformer level, enabling validation against genuine grid behavior rather than synthetic simulations.

### Key Characteristics
- **Temporal Coverage**: October 2025 (actual operational data)
- **Spatial Coverage**: SSEN distribution network areas across Scotland and Southern England
- **Resolution**: 30-minute intervals (perfect alignment with LCL/UK-DALE)
- **Network Level**: Low voltage distribution feeders (400V secondary substations)
- **Measurements**: Active power (kWh), reactive power, device counts, primary/secondary consumption
- **Scale**: 100,000 LV feeders with comprehensive metadata
- **Records**: 100,000 time-series consumption readings
- **Coverage**: 28 unique feeders with actual measurements

### Usage in This Project
- **Real-World Validation**: PRIMARY ground truth for pseudo-feeder realism assessment
- **Distributional Comparison**: Validating that LCL aggregations match REAL network behavior
- **Peak Load Analysis**: Understanding actual peak demand patterns and operational constraints
- **Network Hierarchy**: Complete primary/secondary substation and HV feeder relationships
- **Customer Counts**: total_mpan_count enables weighted aggregation from LCL households
- **Constraint Verification**: Informing verifier reward functions with realistic network limits

### Data Processing Notes
- **Native Format**: Two CSV files from SSEN open data portal
  1. Metadata lookup (100K feeders with network hierarchy and customer counts)
  2. Time-series consumption (100K records with half-hourly readings)
- **Ingestion**: Direct CSV reading with metadata enrichment (fast dictionary lookup)
- **Processing Speed**: 4 seconds for 100K records (optimized with vectorized validation)
- **Metadata Enrichment**: Each consumption record enhanced with feeder characteristics
- **Fields Preserved**:
  - Consumption: total_consumption_active_import (Wh), reactive power, device counts
  - Network: DNO name, primary/secondary substations, HV feeder linkage
  - Geography: Postcodes, substation locations
  - Customers: total_mpan_count for accurate household-to-feeder scaling
- **Data Quality**: 99,967 valid records (33 skipped due to missing consumption)
- **Output Format**: Unified Parquet schema with 14 enriched metadata fields

---

## Cross-Dataset Integration Strategy

### Temporal Resolution Alignment
All datasets will be standardized to **30-minute intervals** to enable meaningful comparisons:
- UK-DALE: Downsampled from higher resolution
- LCL: Native 30-minute resolution (no processing needed)
- SSEN: Native 30-minute resolution

### Feature Harmonization
Common features extracted across all datasets:
- **Temporal Features**: Hour of day, day of week, month, holiday indicators
- **Load Features**: Peak demand, daily energy, load factor
- **Variability Features**: Standard deviation, coefficient of variation
- **Weather Features**: Temperature, humidity (where available)

### Quality Assurance
- **Missing Data**: Consistent handling strategy across datasets
- **Outlier Detection**: Standardized anomaly identification procedures
- **Validation Periods**: Aligned time periods for fair model comparison
- **Data Lineage**: Complete provenance tracking through DVC

### Ethical Considerations
- **UK-DALE**: Anonymous household data, no customer identification possible
- **LCL**: Anonymized smart meter data with privacy protections
- **SSEN**: Aggregated network data only, no individual customer exposure
- **Cross-Dataset**: No linking of individual households across datasets
- **Usage Limitation**: SSEN data used exclusively for validation, never for training

## Data Access and Sources

### Official Sources
- **UK-DALE**: [University of Edinburgh](https://ukerc.rl.ac.uk/DC/cgi-bin/edc_search.pl?GoButton=Detail&WantComp=46&EndNote=on)
- **LCL**: [UK Power Networks](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- **SSEN**: Partnership agreement for research validation purposes

### Local Data Storage
All datasets will be stored in the `data/raw/` directory and tracked with DVC:
```
data/raw/
├── ukdale/           # UK-DALE raw files
├── lcl/              # London Smart Meters raw files
└── ssen/             # SSEN feeder data (validation only)
```

### Remote Storage Configuration
Future remote storage (S3/Azure) will be configured following the guidelines in [Data Governance](data_governance.md).

## Synthetic Sample Datasets

For development, testing, and CI purposes, this project includes tiny synthetic sample datasets that replicate the structure and basic patterns of the real datasets without requiring large downloads.

### Sample Files Location
```
data/samples/
├── lcl_sample.csv      # London Smart Meters sample (48 rows, ~2KB)
├── ukdale_sample.csv   # UK-DALE sample (48 rows, ~2KB)
└── ssen_sample.csv     # SSEN feeder sample (30 real records from production data)
```

### Sample Characteristics
- **Temporal Coverage**: 24 hours (2023-01-01) at 30-minute resolution for LCL/UK-DALE; Oct 2025 for SSEN
- **Pattern**: Realistic daily seasonality with morning and evening peaks
- **Size**: <60KB total for all three files
- **Purpose**: CI testing, development setup, algorithm prototyping
- **SSEN Sample**: Real production data (30 rows) with full metadata enrichment fields

### Usage Notes
- **For CI/Testing**: Use `data/samples/` to verify data loading and processing logic
- **For Development**: Switch to real datasets in `data/raw/` for actual model training
- **For Demos**: Samples provide quick visualization and pipeline validation
- **For Experiments**: Real datasets required for meaningful research results

**Important:** These synthetic samples are **not suitable for research or publication**. They are purely for development convenience and CI automation. All experimental work should use the complete real datasets described above.
