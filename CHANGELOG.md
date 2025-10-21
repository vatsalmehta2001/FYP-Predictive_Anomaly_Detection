# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-10-21

### Added
- **SSEN Real-World Time-Series Data**: Integrated actual operational distribution network consumption data
  - 100,000 LV feeder metadata records with complete network hierarchy
  - 100,000 half-hourly consumption readings from 28 unique operational feeders
  - Comprehensive metadata enrichment (14 fields) including customer counts (total_mpan_count)
  - Network hierarchy: primary/secondary substations, HV/LV feeder relationships, postcodes
  - Primary/secondary consumption breakdown and reactive power measurements
- **Performance Optimizations**: 45x speedup in SSEN ingestion (4 seconds for 100K records)
  - Dictionary-based metadata lookup (O(1) instead of O(n) DataFrame scans)
  - Vectorized pandas validation (filter entire chunks, not row-by-row)
  - Timestamp conversion per chunk (not per row)
- **Real Production Sample Data**: Updated `data/samples/ssen_sample.csv` with 30 real records from production CSV

### Changed
- **SSEN Ingestor**: Complete rewrite to process CSV time-series instead of CKAN API
  - Direct CSV reading with chunked processing (10K rows/batch)
  - Metadata enrichment during ingestion (not post-processing)
  - Ingestion version upgraded to `v2.1_timeseries`
- **Documentation Updates**:
  - `docs/datasets.md`: SSEN section rewritten to highlight real-world validation capabilities
  - `docs/ingestion_specs.md`: New SSEN time-series section with performance details
  - `docs/download_links.md`: Updated with both SSEN CSV download instructions
  - `README.md`: Added "Real-World Grid Validation" to USP, updated datasets table

### Fixed
- **Data Quality**: 99.97% valid record rate (only 33 skipped due to missing consumption values)
- **Metadata Enrichment**: All consumption records enhanced with feeder characteristics and customer counts

### Technical Details
- **Processing Speed**: 100K records in 4 seconds (vs 3+ minutes with previous approach)
- **Memory Efficiency**: 10K row chunks with vectorized validation
- **Data Quality Tracking**: Detailed logging of skipped records by reason
- **Output Format**: Unified Parquet schema with 14 enriched metadata fields
- **Source Files**:
  - `ssen_smart_meter_prod_lv_feeder_lookup_optimized_10_21_2025.csv` (metadata)
  - `ssen_smart_meter_prod_lv_feeder_usage_optimized_10_21_2025.csv` (time-series)

---

## [0.2.0] - 2025-09-24

### Added
- **Modern Neural Models**: PatchTST forecaster with quantile regression and temporal autoencoder for anomaly detection
- **Canonical Import Path**: `fyp.anomaly.autoencoder` for intuitive anomaly model imports (backward compatible)
- **Configuration System**: YAML-based experiment configuration with Pydantic validation
- **MLflow Integration**: Automatic experiment tracking for parameters, metrics, and artifacts
- **Enhanced Ingestion**: Production-grade data pipeline with energy correctness, CKAN API robustness, and DST handling
- **Comprehensive Testing**: 58+ tests including energy conversion validation and DST transition tests
- **CLI Determinism**: Global seeding and CI-safe model configurations for reproducible results

### Changed
- **CI Badge**: Updated to modern GitHub Actions workflow badge format
- **Quickstart Guide**: Replaced `dvc init` with optional `dvc pull` for better user experience
- **DVC Pipeline**: Enhanced stages with comprehensive output declarations and artifact tracking
- **Test Robustness**: Improved shape and ordering assertions instead of absolute numeric comparisons

### Fixed
- **Energy Conversion**: UK-DALE power-to-energy conversion now uses native sampling rates and sums energy (not averages power)
- **Timezone Handling**: Proper DST transition handling for Europe/London timezone across all datasets
- **API Robustness**: SSEN CKAN client with rate limiting, caching, and exponential backoff
- **Memory Efficiency**: Chunked Parquet writes with 128MB row groups for large dataset processing

### Deprecated
- **Import Path**: `fyp.models.autoencoder` deprecated in favor of `fyp.anomaly.autoencoder` (compatibility maintained)

### Technical Details
- **Energy Physics**: Proper instantaneous power to energy conversion: `Energy = Power × Δt`
- **Quantile Regression**: PatchTST outputs prediction intervals with pinball loss optimization
- **Data Quality**: Enhanced provenance tracking with SHA256 hashes and quality metrics
- **CI Performance**: Sample mode runs in <15 seconds with minimal model architectures

---

## [0.1.0] - 2025-09-23

### Added
- Initial project structure with GitHub workflows and issue templates
- Comprehensive documentation (datasets, data governance, self-play design, experiments, feeder evaluation)
- DVC data version control setup with placeholder pipeline stages
- Basic test suite with smoke tests for CI validation
- Community files (CONTRIBUTING.md, CODE_OF_CONDUCT.md, LICENSE, CITATION.cff)
- Project configuration (pyproject.toml, .gitignore, .pre-commit-config.yaml)

### Documentation
- **README.md**: Project vision, architecture diagram, quick start guide
- **docs/datasets.md**: UK-DALE, London Smart Meters, and SSEN dataset descriptions
- **docs/data_governance.md**: DVC usage, provenance, and retention policies
- **docs/selfplay_design.md**: Propose→solve→verify architecture for time series
- **docs/experiments.md**: MLflow organization and naming conventions
- **docs/feeder_eval.md**: Validation methodology against real networks

---

## Release Notes

To create a new release:
```bash
git tag -a v0.2.0 -m "Release v0.2.0: Modern neural models with enhanced ingestion"
git push origin v0.2.0
```
