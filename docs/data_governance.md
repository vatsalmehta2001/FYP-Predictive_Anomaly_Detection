# Data Governance & Management

This document outlines the data management policies, version control strategies, and governance procedures for the FYP Energy Forecasting project.

## Data Layout Policy

### Directory Structure
```
data/
├── raw/              # Original, immutable datasets (gitignored, DVC tracked)
│   ├── ukdale/       # UK-DALE household consumption data
│   ├── lcl/          # London Smart Meters data
│   └── ssen/         # SSEN LV feeder data (validation only)
├── processed/        # Cleaned and harmonized data (DVC tracked)
│   ├── ukdale_30min/ # UK-DALE resampled to 30-minute intervals
│   ├── lcl_clean/    # LCL with outliers removed and gaps filled
│   └── features/     # Engineered features across datasets
└── derived/          # Model outputs and experiment artifacts (DVC tracked)
    ├── models/       # Trained model checkpoints
    ├── predictions/  # Forecast outputs
    ├── metrics/      # Evaluation results
    └── figures/      # Generated plots and visualizations
```

### Data Immutability Principles
- **Raw Data**: Never modified after initial ingestion, tracked as read-only
- **Processed Data**: Versioned outputs of deterministic transformations
- **Derived Data**: Reproducible artifacts from experiments and models
- **Audit Trail**: Complete lineage from raw data to final results

## DVC (Data Version Control) Usage

### Core DVC Principles
- **Raw data** in `data/raw/` is gitignored but DVC tracked
- **All data transformations** are defined in `dvc.yaml` pipeline stages
- **Remote storage** will be configured for team collaboration (S3/Azure)
- **Data lineage** is automatically maintained through stage dependencies

### DVC Configuration
```yaml
# .dvc/config (example for future remote setup)
[core]
    analytics = false
    check_update = false

['remote "s3remote"']
    url = s3://fyp-energy-data/
    # credentials will be handled via environment variables

['remote "azureremote"']
    url = azure://fypdata/datasets/
    # credentials handled via service principal
```

### Adding Data to DVC
```bash
# Track new raw dataset
dvc add data/raw/ukdale/house1.csv
git add data/raw/ukdale/house1.csv.dv .gitignore
git commit -m "Add UK-DALE house 1 data"

# Define processing stage
dvc run -n process_ukdale \
  -d data/raw/ukdale/ \
  -o data/processed/ukdale_30min/ \
  python src/fyp/data/process_ukdale.py

# Push data to remote (when configured)
dvc push
```

### Data Access Patterns
- **Local Development**: Use `dvc pull` to sync latest data versions
- **CI/CD**: Automated pipeline validation with minimal data subsets
- **Production**: Full data pipeline execution with complete datasets

## Remote Storage Configuration

### Cloud Storage Setup
The project supports multiple cloud providers for remote data storage:

#### AWS S3 Configuration
```bash
# Set up S3 remote (admin only)
dvc remote add s3remote s3://fyp-energy-data/
dvc remote modify s3remote region eu-west-2

# Configure credentials (never store in repo)
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

#### Azure Blob Storage Configuration
```bash
# Set up Azure remote (admin only)
dvc remote add azureremote azure://fypdata/datasets/

# Configure credentials via service principal
export AZURE_STORAGE_ACCOUNT=fypdata
export AZURE_STORAGE_KEY=your_storage_key
```

### Credential Management
- **Environment Variables**: All credentials stored as environment variables
- **CI/CD Secrets**: GitHub Actions secrets for automated workflows
- **Local .env**: Individual developer environment files (gitignored)
- **No Hardcoding**: Never commit credentials to version control

### Access Control
- **Read-Only**: Most team members have read-only access to datasets
- **Write Access**: Limited to designated data administrators
- **Audit Logging**: All data access logged for security compliance

## Data Provenance & Lineage

### Metadata Tracking
Each dataset includes comprehensive metadata:
```yaml
# Example: data/raw/ukdale/metadata.yaml
dataset:
  name: "UK-DALE House 1"
  source: "University of Edinburgh"
  license: "Academic Use Only"
  temporal_coverage: "2012-04-01 to 2015-01-01"
  resolution: "6-second intervals"
  size_mb: 1250
  checksum_md5: "a1b2c3d4e5f6..."

provenance:
  acquired_date: "2024-09-15"
  acquired_by: "project_admin"
  processing_notes: "Original format, no modifications"
  validation_status: "checksums verified"
```

### Processing Documentation
Every transformation stage includes:
- **Input data versions** (DVC tracked)
- **Code version** (git commit hash)
- **Parameter values** (hyperparameters, thresholds)
- **Output data versions** (DVC tracked)
- **Execution environment** (Python version, dependencies)

### Reproducibility Requirements
- **Deterministic Processing**: All random seeds fixed
- **Environment Specification**: Exact dependency versions in `pyproject.toml`
- **Parameter Documentation**: All processing parameters version controlled
- **Validation Checksums**: Data integrity verification at each stage

## Data Retention & Lifecycle

### Retention Policies
- **Raw Data**: Permanent retention (reference datasets)
- **Processed Data**: 2-year retention for major versions
- **Intermediate Results**: 6-month retention for debugging
- **Model Artifacts**: Permanent retention for published results
- **Temporary Files**: Daily cleanup of scratch directories

### Archival Strategy
- **Active Data**: Immediately accessible in cloud storage
- **Archive Tier**: Infrequently accessed data moved to cheaper storage
- **Long-term Archive**: Historical versions stored in glacier storage
- **Documentation**: All archived data includes complete metadata

### Data Deletion Procedures
1. **Review Request**: Data deletion requires admin approval
2. **Dependency Check**: Verify no downstream processes depend on data
3. **Backup Verification**: Ensure critical data has alternative sources
4. **Secure Deletion**: Use appropriate tools for secure data removal
5. **Audit Log**: Document all deletion actions with justification

## Privacy & Compliance

### Data Anonymization
- **UK-DALE**: Already anonymized at source
- **LCL**: Customer identifiers removed, only temporal patterns retained
- **SSEN**: Feeder-level aggregation only, no individual customer data

### Usage Restrictions
- **SSEN Data**: Validation purposes only, never used for training
- **Cross-Dataset Linking**: Explicitly prohibited between datasets
- **Publication Requirements**: All results must maintain household anonymity

### Compliance Framework
- **GDPR Compliance**: No personal data processing or storage
- **Academic Ethics**: Institutional review board approval obtained
- **Data Agreements**: All usage within bounds of original data licenses
- **Audit Trail**: Complete documentation of all data handling procedures

## Quality Assurance

### Data Validation Pipeline
```python
# Example validation checks
def validate_dataset(data_path):
    checks = [
        check_temporal_continuity(),
        check_value_ranges(),
        check_missing_data_patterns(),
        check_statistical_properties(),
        verify_data_integrity()
    ]
    return all(checks)
```

### Monitoring & Alerting
- **Data Drift Detection**: Statistical tests for distribution changes
- **Pipeline Monitoring**: Automated alerts for processing failures
- **Quality Metrics**: Tracked in MLflow for trend analysis
- **Anomaly Detection**: Automated flagging of unusual data patterns

### Error Handling
- **Graceful Degradation**: Pipeline continues with warnings for minor issues
- **Fail-Fast**: Critical errors halt processing immediately
- **Recovery Procedures**: Documented steps for common failure scenarios
- **Rollback Capability**: Ability to revert to previous known-good state

## Team Collaboration

### Access Management
- **Role-Based Access**: Different permissions for different team roles
- **Onboarding Process**: New team members guided through data access setup
- **Training Requirements**: All team members trained on data governance policies
- **Regular Reviews**: Quarterly access permission audits

### Communication Protocols
- **Data Changes**: All significant data updates communicated to team
- **Issue Reporting**: Clear procedures for reporting data quality issues
- **Documentation Updates**: Regular reviews and updates of governance documents
- **Knowledge Sharing**: Regular sessions on data best practices
