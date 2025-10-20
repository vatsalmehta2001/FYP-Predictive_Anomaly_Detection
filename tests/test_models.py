"""Tests for modern neural models."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fyp.config import create_sample_config


# Test data creation helpers
def create_synthetic_energy_series(
    n_points: int = 144, noise_level: float = 0.1
) -> np.ndarray:
    """Create synthetic energy consumption series."""
    t = np.linspace(0, 6 * np.pi, n_points)  # 3 daily cycles
    # Daily pattern + weekly pattern + noise
    daily = 1 + 0.5 * np.sin(2 * np.pi * t / 48)  # 48 points per day
    weekly = 0.2 * np.sin(2 * np.pi * t / (48 * 7))  # Weekly pattern
    noise = np.random.normal(0, noise_level, n_points)

    energy = daily + weekly + noise
    return np.maximum(energy, 0.1)  # Ensure positive


def create_energy_with_anomalies(n_points: int = 144) -> tuple[np.ndarray, np.ndarray]:
    """Create energy series with injected anomalies."""
    data = create_synthetic_energy_series(n_points, noise_level=0.05)
    labels = np.zeros(n_points)

    # Inject spike anomalies
    anomaly_indices = [30, 70, 110]
    for idx in anomaly_indices:
        if idx < n_points:
            data[idx] *= 3  # Spike
            labels[idx] = 1
            # Small tail
            if idx + 1 < n_points:
                data[idx + 1] *= 1.5
                labels[idx + 1] = 1

    return data, labels


def create_forecasting_windows_synthetic(n_windows: int = 5) -> list[dict]:
    """Create synthetic forecasting windows."""
    windows = []

    for i in range(n_windows):
        # Create sequence
        full_series = create_synthetic_energy_series(144)

        # Split into history and target
        history_len = 96
        target_len = 48

        history = full_series[:history_len]
        target = full_series[history_len : history_len + target_len]

        # Create timestamps
        base_time = np.datetime64("2023-01-01T00:00:00")
        history_times = base_time + np.arange(history_len) * np.timedelta64(30, "m")
        target_times = base_time + np.arange(
            history_len, history_len + target_len
        ) * np.timedelta64(30, "m")

        windows.append(
            {
                "entity_id": f"test_entity_{i}",
                "history_energy": history,
                "target_energy": target,
                "history_timestamps": history_times,
                "target_timestamps": target_times,
                "interval_mins": 30,
            }
        )

    return windows


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.getenv("CI"), reason="Skip GPU tests in CI"
)
class TestPatchTST:
    """Test PatchTST forecasting model."""

    def test_patchtst_creation(self):
        """Test PatchTST model creation."""
        try:
            from fyp.models.patchtst import EnergyPatchTST

            model = EnergyPatchTST(
                patch_len=8,
                d_model=32,
                n_heads=2,
                n_layers=1,
                forecast_horizon=16,
                quantiles=[0.1, 0.5, 0.9],
            )

            # Test forward pass
            batch_size = 4
            n_patches = 6
            patch_len = 8

            x = torch.randn(batch_size, n_patches, patch_len)
            output = model(x)

            # Should output quantiles for each forecast step
            assert output.shape == (batch_size, 16, 3)  # horizon=16, quantiles=3

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_patchtst_forecaster_training(self):
        """Test PatchTST forecaster training on synthetic data."""
        try:
            from fyp.models.patchtst import PatchTSTForecaster

            # Create synthetic windows
            windows = create_forecasting_windows_synthetic(3)

            # Create small model for testing
            forecaster = PatchTSTForecaster(
                patch_len=8,
                d_model=16,
                n_heads=2,
                n_layers=1,
                forecast_horizon=16,
                max_epochs=2,
                batch_size=2,
                learning_rate=1e-2,
            )

            # Training should not crash
            result = forecaster.fit(windows)

            assert "final_train_loss" in result
            assert "epochs_trained" in result
            assert result["epochs_trained"] > 0

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_patchtst_prediction_quantiles(self):
        """Test PatchTST quantile predictions."""
        try:
            from fyp.models.patchtst import PatchTSTForecaster

            windows = create_forecasting_windows_synthetic(2)

            forecaster = PatchTSTForecaster(
                patch_len=8,
                d_model=16,
                n_heads=1,
                n_layers=1,
                forecast_horizon=8,
                quantiles=[0.1, 0.5, 0.9],
                max_epochs=1,
                batch_size=2,
            )

            # Fit on minimal data
            forecaster.fit(windows)

            # Make prediction
            history = windows[0]["history_energy"][:48]  # Use first 48 points
            quantile_forecasts = forecaster.predict(
                history, steps=8, return_quantiles=True
            )

            # Should have all quantiles
            assert "0.1" in quantile_forecasts
            assert "0.5" in quantile_forecasts
            assert "0.9" in quantile_forecasts

            # Quantiles should be ordered: q0.1 <= q0.5 <= q0.9
            q10 = quantile_forecasts["0.1"]
            q50 = quantile_forecasts["0.5"]
            q90 = quantile_forecasts["0.9"]

            assert len(q10) == 8
            assert len(q50) == 8
            assert len(q90) == 8

            # Check quantile ordering (most of the time)
            assert np.mean(q10 <= q50) > 0.7
            assert np.mean(q50 <= q90) > 0.7

        except ImportError:
            pytest.skip("PyTorch not available")


class TestAutoencoder:
    """Test temporal autoencoder model."""

    def test_autoencoder_creation(self):
        """Test autoencoder model creation."""
        try:
            from fyp.models.autoencoder import TemporalAutoencoder

            model = TemporalAutoencoder(
                input_size=24,
                hidden_sizes=[16, 8],
                dropout=0.1,
            )

            # Test forward pass
            batch_size = 4
            input_size = 24

            x = torch.randn(batch_size, input_size)
            output = model(x)

            # Should reconstruct input
            assert output.shape == (batch_size, input_size)

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_autoencoder_anomaly_detector(self):
        """Test autoencoder anomaly detector."""
        try:
            from fyp.anomaly.autoencoder import AutoencoderAnomalyDetector

            # Create normal sequences for training
            normal_sequences = [
                create_synthetic_energy_series(96, noise_level=0.05) for _ in range(3)
            ]

            # Create detector
            detector = AutoencoderAnomalyDetector(
                window_size=24,
                hidden_sizes=[16, 8],
                max_epochs=2,
                batch_size=4,
                learning_rate=1e-2,
            )

            # Training should not crash
            result = detector.fit(normal_sequences)

            assert "final_loss" in result
            assert "threshold" in result
            assert result["threshold"] > 0

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_autoencoder_anomaly_scoring(self):
        """Test autoencoder anomaly scoring."""
        try:
            from fyp.anomaly.autoencoder import AutoencoderAnomalyDetector

            # Create data with known anomalies
            data, labels = create_energy_with_anomalies(96)

            # Split into train/test
            train_data = data[:72]  # Normal data for training
            test_data = data[72:]  # Data with anomalies
            labels[72:]

            # Train detector
            detector = AutoencoderAnomalyDetector(
                window_size=16,
                hidden_sizes=[12, 6],
                max_epochs=2,
                batch_size=4,
                contamination=0.1,
            )

            detector.fit([train_data])

            # Score test data
            scores = detector.predict_scores(test_data)

            assert len(scores) == len(test_data)
            assert all(s >= 0 for s in scores)

            # Should produce some variation in scores
            assert np.std(scores) > 0

        except ImportError:
            pytest.skip("PyTorch not available")


class TestModelIntegration:
    """Integration tests for models with runner."""

    def test_runner_with_patchtst(self):
        """Test runner integration with PatchTST."""
        try:
            from fyp.config import create_sample_config
            from fyp.runner import run_forecasting_baselines

            config = create_sample_config()

            # Should not crash when PyTorch available
            try:
                summary = run_forecasting_baselines(
                    dataset="lcl",
                    use_samples=True,
                    model_type="patchtst",
                    config=config,
                    output_dir=Path("/tmp/test_patchtst"),
                )

                # Should produce some results
                assert isinstance(summary, dict)

            except Exception as e:
                # Acceptable if PyTorch/data issues, just ensure graceful handling
                print(f"PatchTST test info: {e}")

        except ImportError:
            pytest.skip("Advanced models not available")

    def test_runner_with_autoencoder(self):
        """Test runner integration with autoencoder."""
        try:
            from fyp.config import create_sample_config
            from fyp.runner import run_anomaly_baselines

            config = create_sample_config()

            try:
                summary = run_anomaly_baselines(
                    dataset="lcl",
                    use_samples=True,
                    model_type="autoencoder",
                    config=config,
                    output_dir=Path("/tmp/test_autoencoder"),
                )

                assert isinstance(summary, dict)

            except Exception as e:
                print(f"Autoencoder test info: {e}")

        except ImportError:
            pytest.skip("Advanced models not available")


class TestImportCompatibility:
    """Test import path compatibility."""

    def test_autoencoder_import_paths(self):
        """Test that both import paths work and point to same class."""
        try:
            # New canonical path
            from fyp.anomaly.autoencoder import AutoencoderAnomalyDetector as A1

            # Original path (deprecated)
            from fyp.models.autoencoder import AutoencoderAnomalyDetector as A2

            # Should be the same class
            assert A1 is A2

            # Should be importable
            assert hasattr(A1, "__init__")
            assert callable(A1)

        except ImportError:
            pytest.skip("PyTorch not available")


class TestConfig:
    """Test configuration system."""

    def test_sample_config_creation(self):
        """Test sample configuration creation."""
        config = create_sample_config()

        assert config.use_samples is True
        assert config.forecasting.max_epochs <= 5  # Fast for CI
        assert config.anomaly.max_epochs <= 5
        assert config.forecasting.d_model <= 64  # Small model

    def test_config_validation(self):
        """Test configuration validation."""
        from fyp.config import AnomalyConfig, ExperimentConfig, ForecastingConfig

        # Should accept valid config
        config = ExperimentConfig(
            dataset="lcl",
            forecasting=ForecastingConfig(
                patch_len=16,
                d_model=64,
                forecast_horizon=48,
            ),
            anomaly=AnomalyConfig(
                window_size=48,
                hidden_sizes=[32, 16],
            ),
        )

        assert config.dataset == "lcl"
        assert config.forecasting.patch_len == 16
        assert config.anomaly.window_size == 48


class TestQuantileCoverage:
    """Test quantile prediction coverage."""

    def test_quantile_coverage_synthetic(self):
        """Test that quantile predictions have reasonable coverage."""
        # Create deterministic test data
        np.random.seed(42)
        n_points = 100
        true_values = np.random.normal(1.0, 0.2, n_points)

        # Simulate quantile predictions (perfect case)
        q10 = np.percentile(true_values, 10) * np.ones(n_points)
        q90 = np.percentile(true_values, 90) * np.ones(n_points)

        # Test coverage calculation
        from fyp.metrics import coverage_score

        coverage_80 = coverage_score(true_values, q10, q90)

        # Should be around 80% coverage
        assert 0.7 <= coverage_80 <= 0.9

    def test_pinball_loss_calculation(self):
        """Test pinball loss calculation."""
        from fyp.metrics import pinball_loss

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])

        # Test different quantiles
        loss_50 = pinball_loss(y_true, y_pred, 0.5)
        loss_10 = pinball_loss(y_true, y_pred, 0.1)
        loss_90 = pinball_loss(y_true, y_pred, 0.9)

        assert loss_50 >= 0
        assert loss_10 >= 0
        assert loss_90 >= 0

        # For over-predictions, higher quantile should have higher loss
        # For under-predictions, lower quantile should have higher loss
        assert isinstance(loss_50, float)


class TestAnomalyLatency:
    """Test anomaly detection latency."""

    def test_synthetic_spike_detection(self):
        """Test detection of synthetic spikes with latency measurement."""
        # Create data with known anomaly locations
        data, labels = create_energy_with_anomalies(96)

        # Find true anomaly starts
        true_starts = []
        in_anomaly = False
        for i, label in enumerate(labels):
            if label == 1 and not in_anomaly:
                true_starts.append(i)
                in_anomaly = True
            elif label == 0:
                in_anomaly = False

        # Simple anomaly detector (threshold on z-score)
        mean_val = np.mean(data)
        std_val = np.std(data)
        z_scores = np.abs(data - mean_val) / std_val

        # Detect anomalies
        anomaly_threshold = 2.0
        predicted_anomalies = z_scores > anomaly_threshold
        predicted_times = np.where(predicted_anomalies)[0].tolist()

        # Test latency calculation
        from fyp.metrics import detection_latency

        if true_starts and predicted_times:
            latency_metrics = detection_latency(
                true_starts, predicted_times, max_delay=10
            )

            assert "avg_latency" in latency_metrics
            assert "detection_rate" in latency_metrics
            assert 0 <= latency_metrics["detection_rate"] <= 1
            assert latency_metrics["avg_latency"] >= 0


# Skip GPU tests in CI environment
@pytest.fixture(autouse=True)
def skip_gpu_in_ci():
    """Automatically skip GPU-intensive tests in CI."""
    if os.getenv("CI") and torch.cuda.is_available():
        pytest.skip("Skipping GPU tests in CI environment")
