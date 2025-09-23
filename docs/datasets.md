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
- Will be aggregated to 30-minute resolution for consistency across datasets
- Missing data periods will be identified and handled appropriately
- Seasonal and holiday patterns will be extracted for model features
- Appliance-level data will be used for explainability analysis

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
- Already at target 30-minute resolution
- Comprehensive coverage allows for robust train/validation/test splits
- Weather data will be aligned for enhanced forecasting models
- Household clustering will identify representative consumption archetypes

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
- **Access Restrictions**: Used only for validation purposes, not for training
- **Privacy Preservation**: All comparisons will be statistical/distributional, no reverse engineering
- **Temporal Alignment**: Matched with household data periods where possible
- **Aggregation Level**: Focus on feeder-level patterns, not individual customer identification

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
