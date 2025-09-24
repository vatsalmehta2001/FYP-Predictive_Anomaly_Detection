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
Scottish and Southern Electricity Networks (SSEN) Low Voltage (LV) feeder data provides real distribution network measurements from operational electricity networks. This dataset represents actual aggregated load patterns at the distribution transformer level.

### Key Characteristics
- **Temporal Coverage**: Recent operational data (specific timeframes TBD)
- **Spatial Coverage**: SSEN distribution network areas
- **Resolution**: 30-minute intervals
- **Network Level**: Low voltage distribution feeders (11kV and below)
- **Measurements**: Active power, reactive power, voltage levels
- **Scale**: Multiple feeders across diverse geographical areas

### Usage in This Project
- **External Validation**: Primary ground truth for pseudo-feeder realism assessment
- **Distributional Comparison**: Validating that synthetic aggregations match real network behavior
- **Peak Load Analysis**: Understanding real-world peak demand patterns and constraints
- **Constraint Verification**: Informing verifier reward functions with realistic network limits

### Data Processing Notes
- **Native Format**: CSV lookup + CKAN API for time series data
- **Ingestion**: API client with rate limiting and pagination support
- **Resolution**: 30-minute feeder aggregates
- **Metadata**: Feeder names, locations, capacities from lookup joined to time series
- **Access Mode**: Public API (no auth required) or mock data fallback for testing

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
└── ssen_sample.csv     # SSEN feeder sample (48 rows, ~2KB)
```

### Sample Characteristics
- **Temporal Coverage**: 24 hours (2023-01-01) at 30-minute resolution
- **Pattern**: Realistic daily seasonality with morning and evening peaks
- **Size**: <60KB total for all three files
- **Purpose**: CI testing, development setup, algorithm prototyping

### Usage Notes
- **For CI/Testing**: Use `data/samples/` to verify data loading and processing logic
- **For Development**: Switch to real datasets in `data/raw/` for actual model training
- **For Demos**: Samples provide quick visualization and pipeline validation
- **For Experiments**: Real datasets required for meaningful research results

**⚠️ Important:** These synthetic samples are **not suitable for research or publication**. They are purely for development convenience and CI automation. All experimental work should use the complete real datasets described above.
