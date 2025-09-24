# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
