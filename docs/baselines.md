# Baseline Models

This document describes the baseline forecasting and anomaly detection models implemented for energy consumption data.

## Overview

The baseline models provide standard benchmarks for evaluating more sophisticated approaches. They operate on the unified schema from the ingestion pipeline and produce standardized metrics and visualizations.

## Forecasting Baselines

### Seasonal Naive
**Implementation**: `SeasonalNaive`

Repeats values from the same time in the previous seasonal period (24 hours by default).

**Algorithm**:
```python
forecast[t] = historical_data[t - seasonal_period]
```

**Use Case**: Simple baseline that captures daily seasonality in energy consumption.

**Strengths**:
- Zero-parameter model
- Captures strong daily patterns
- Fast and interpretable

**Weaknesses**:
- No adaptation to trends
- Cannot handle irregular patterns
- Poor performance on non-stationary data

### Linear Trend Forecaster
**Implementation**: `LinearTrendForecaster`

Ridge regression with engineered features including trend, lags, and cyclical time components.

**Features**:
- Linear trend component
- Lagged values (30min, 1h, 12h, 24h)
- Hour-of-day cyclic encoding: `sin(2π·hour/24)`, `cos(2π·hour/24)`
- Day-of-week cyclic encoding (when sufficient data available)

**Use Case**: Simple learnable baseline that can adapt to trends and calendar patterns.

**Strengths**:
- Learns from data
- Handles trends and calendar effects
- Regularization prevents overfitting
- CPU-friendly

**Weaknesses**:
- Linear assumptions
- Limited non-linear pattern capture
- Requires feature engineering

### Ensemble Forecaster
**Implementation**: `EnsembleForecaster`

Weighted combination of multiple forecasters with uncertainty quantification.

**Default Configuration**:
- 30% Seasonal Naive
- 70% Linear Trend

**Uncertainty**: Uses ensemble spread to estimate prediction intervals.

## Anomaly Detection Baselines

### Decomposition-Based Detector
**Implementation**: `DecompositionAnomalyDetector`

Decomposes time series into trend, seasonal, and residual components, then scores residuals.

**Algorithm**:
1. **Trend Extraction**: Moving average or linear trend
2. **Seasonal Pattern**: Average pattern across seasonal periods
3. **Residual Scoring**: Standardized residual magnitude
4. **Adaptive Thresholding**: Time-of-day adjustment

**Adaptive Thresholds**:
- Higher sensitivity during low-consumption hours (2-6 AM)
- Lower sensitivity during peak hours (6-10 PM)

**Use Case**: Physics-informed approach leveraging energy consumption's seasonal structure.

### Statistical Detector
**Implementation**: `StatisticalAnomalyDetector`

Rolling window statistical analysis with z-score based scoring.

**Algorithm**:
1. Compute rolling statistics (mean, std) over window
2. Calculate z-score: `|value - rolling_mean| / rolling_std`
3. Normalize by threshold: `z_score / n_sigma`

**Use Case**: Distribution-based approach sensitive to local statistical deviations.

### Ensemble Detector
**Implementation**: `EnsembleAnomalyDetector`

Weighted combination of detectors for robust anomaly scoring.

**Default Configuration**:
- 70% Decomposition-based
- 30% Statistical

## Metrics

### Forecasting Metrics

| Metric | Description | Formula | Interpretation |
|--------|-------------|---------|----------------|
| **MAE** | Mean Absolute Error | `mean(\|y_true - y_pred\|)` | Average forecast error in kWh |
| **RMSE** | Root Mean Squared Error | `sqrt(mean((y_true - y_pred)²))` | Penalizes large errors more |
| **MAPE** | Mean Absolute Percentage Error | `mean(\|y_true - y_pred\| / y_true) × 100` | Relative error as percentage |
| **MASE** | Mean Absolute Scaled Error | `MAE / MAE_seasonal_naive` | Error relative to seasonal baseline |
| **Pinball Loss** | Quantile loss | `mean(max(τ(y_true - y_pred), (τ-1)(y_true - y_pred)))` | Quantile forecast accuracy |
| **Coverage** | Prediction interval coverage | `mean(y_lower ≤ y_true ≤ y_upper)` | Interval reliability |

### Anomaly Detection Metrics

| Metric | Description | Formula | Interpretation |
|--------|-------------|---------|----------------|
| **Precision** | True positive rate among predictions | `TP / (TP + FP)` | Fraction of alerts that are real anomalies |
| **Recall** | True positive rate among actual anomalies | `TP / (TP + FN)` | Fraction of anomalies detected |
| **F1 Score** | Harmonic mean of precision and recall | `2 × (Precision × Recall) / (Precision + Recall)` | Balanced detection performance |
| **Detection Latency** | Average delay in anomaly detection | `mean(detection_time - anomaly_start)` | Responsiveness in time steps |

## Usage

### Command Line Interface

```bash
# Forecasting baselines on sample data
python -m fyp.runner forecast --dataset lcl --use-samples

# Anomaly detection on real data
python -m fyp.runner anomaly --dataset ukdale

# Custom parameters
python -m fyp.runner forecast --dataset ssen --horizon 96 --output-dir results/
```

### Programmatic Usage

```python
from fyp.baselines.forecasting import create_default_forecasters
from fyp.baselines.anomaly import create_default_detectors
from fyp.data_loader import EnergyDataLoader

# Load data
loader = EnergyDataLoader()
df = loader.load_dataset("lcl")

# Create forecasting windows
windows = loader.create_forecasting_windows(df, horizon=48)

# Initialize models
forecasters = create_default_forecasters()
detectors = create_default_detectors()

# Use models
for window in windows[:5]:
    history = window["history_energy"]

    # Forecasting
    for name, forecaster in forecasters.items():
        forecaster.fit(history)
        forecast = forecaster.predict(history, steps=48)

    # Anomaly detection
    for name, detector in detectors.items():
        detector.fit(history)
        scores = detector.predict_scores(history)
```

### DVC Pipeline

```bash
# Run baseline training stage
dvc repro train_baselines

# Check outputs
ls data/derived/evaluation/
# forecast_metrics.csv
# anomaly_metrics.csv
# forecast_mae_by_model.png
# anomaly_precision_recall.png
```

## Expected Performance

### Forecasting (on sample data)

| Model | Typical MAE (kWh) | Typical MAPE (%) | Use Case |
|-------|-------------------|------------------|----------|
| Seasonal Naive | 0.1-0.3 | 15-25% | Daily pattern baseline |
| Linear Trend | 0.08-0.25 | 12-20% | Trend-aware baseline |
| Ensemble | 0.07-0.22 | 10-18% | Robust combination |

### Anomaly Detection (synthetic anomalies)

| Model | Typical Precision | Typical Recall | Use Case |
|-------|-------------------|----------------|----------|
| Decomposition | 0.6-0.8 | 0.5-0.7 | Seasonal anomalies |
| Statistical | 0.5-0.7 | 0.6-0.8 | Statistical outliers |
| Ensemble | 0.6-0.8 | 0.6-0.8 | Balanced performance |

## Interpreting Results

### Forecasting Analysis

1. **MAE vs MAPE**: MAE shows absolute error scale, MAPE shows relative error
2. **MASE < 1**: Better than seasonal naive baseline
3. **Coverage**: Prediction intervals should achieve target coverage (e.g., 80% for 80% intervals)
4. **Model Comparison**: Ensemble should be competitive with best individual model

### Anomaly Detection Analysis

1. **Precision-Recall Trade-off**: Higher precision = fewer false alarms, higher recall = catch more anomalies
2. **F1 Score**: Balanced metric when precision and recall are equally important
3. **Detection Latency**: Lower is better for real-time applications
4. **Score Distribution**: Anomaly scores should separate normal and anomalous behavior

## Limitations and Future Work

### Current Limitations

1. **Simple Models**: Linear assumptions and basic decomposition
2. **Limited Features**: No weather, calendar holidays, or external factors
3. **Single-Entity**: No cross-entity learning or population modeling
4. **Static Thresholds**: Fixed anomaly detection thresholds

### Extensions for Advanced Models

The baseline framework provides interfaces for integrating:

- **PatchTST**: Patch-based transformer forecasting
- **N-BEATS**: Neural basis expansion forecasting
- **Isolation Forest**: ML-based anomaly detection
- **LSTM/GRU**: Recurrent neural networks
- **Prophet**: Facebook's forecasting tool

### Performance Optimization

- **Batched Processing**: Process multiple entities simultaneously
- **Parallel Fitting**: Fit models across entities in parallel
- **Incremental Updates**: Update models with new data
- **Hyperparameter Tuning**: Grid search or Bayesian optimization

## Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce batch size or number of windows
- Process entities sequentially

**Poor Performance**:
- Check data quality and stationarity
- Verify seasonal period alignment
- Ensure sufficient training data

**Missing Plots**:
- Check matplotlib backend
- Verify output directory permissions
- Ensure results data exists

**Import Errors**:
- Set PYTHONPATH: `export PYTHONPATH="$(pwd)/src"`
- Install dependencies: `poetry install`

## Modern Neural Models

### PatchTST Forecaster
**Implementation**: `PatchTSTForecaster`

Patch-based transformer with quantile regression heads for uncertainty quantification.

**Architecture**:
- **Patch Embedding**: Convert time series segments to patch embeddings
- **Transformer Encoder**: Multi-head attention with positional encoding
- **Quantile Head**: Outputs multiple quantiles (0.1, 0.5, 0.9) simultaneously

**Key Features**:
- **Quantile Regression**: Provides prediction intervals for uncertainty quantification
- **Patch-based Processing**: Efficient handling of long sequences
- **CPU-friendly**: Optimized for production deployment without GPU requirements

**Usage**:
```bash
# Modern forecasting with uncertainty
python -m fyp.runner forecast --dataset lcl --model-type patchtst --use-samples

# Full training on real data
python -m fyp.runner forecast --dataset ukdale --model-type patchtst
```

### Temporal Autoencoder
**Implementation**: `AutoencoderAnomalyDetector`

Lightweight autoencoder for reconstruction-based anomaly detection.

**Architecture**:
- **Encoder**: Progressive dimensionality reduction (48 → 32 → 16 → 8)
- **Decoder**: Reconstruction back to original dimensions
- **Scoring**: Standardized reconstruction error with adaptive thresholding

**Key Features**:
- **Unsupervised Learning**: Trains on normal data only
- **Temporal Windows**: Sliding window analysis for sequence anomalies
- **Adaptive Thresholds**: Contamination-based threshold setting

**Usage**:
```bash
# Modern anomaly detection
python -m fyp.runner anomaly --dataset ssen --model-type autoencoder --use-samples

# Production anomaly detection
python -m fyp.runner anomaly --dataset ukdale --model-type autoencoder

# Programmatic usage (canonical import)
from fyp.anomaly.autoencoder import AutoencoderAnomalyDetector
```

### Configuration System

Models support YAML configuration for reproducible experiments:

```yaml
# config.yaml
dataset: lcl
use_samples: true

forecasting:
  model_type: patchtst
  patch_len: 16
  d_model: 128
  n_heads: 8
  n_layers: 4
  forecast_horizon: 48
  quantiles: [0.1, 0.5, 0.9]
  max_epochs: 50
  batch_size: 32
  learning_rate: 0.001

anomaly:
  model_type: autoencoder
  window_size: 48
  hidden_sizes: [32, 16, 8]
  max_epochs: 30
  contamination: 0.05
```

**Usage with Config**:
```bash
python -m fyp.runner forecast --config config.yaml
```

### MLflow Integration

Automatic experiment tracking when MLflow is available:

```bash
# Experiments tracked automatically
python -m fyp.runner forecast --model-type patchtst --mlflow-experiment energy_forecasting

# View results
mlflow ui
```

**Logged Artifacts**:
- Model parameters and hyperparameters
- Training and validation metrics
- Forecast accuracy plots
- Quantile coverage analysis
- Model checkpoints (optional)

For more advanced forecasting and anomaly detection approaches, see the self-play design documentation.
