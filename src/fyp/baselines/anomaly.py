"""Baseline anomaly detection models."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """Base class for anomaly detectors."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fit the detector on normal data."""
        pass

    @abstractmethod
    def predict_scores(self, data: np.ndarray) -> np.ndarray:
        """Generate anomaly scores."""
        pass

    def predict_labels(self, data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Generate binary anomaly labels."""
        scores = self.predict_scores(data)
        return (scores > threshold).astype(int)


class DecompositionAnomalyDetector(BaseAnomalyDetector):
    """Anomaly detector based on seasonal decomposition."""

    def __init__(
        self,
        seasonal_period: int = 48,  # 24 hours at 30-min resolution
        contamination: float = 0.05,
        window_size: int = 24,  # For adaptive thresholding
    ):
        super().__init__("decomposition")
        self.seasonal_period = seasonal_period
        self.contamination = contamination
        self.window_size = window_size
        self.seasonal_pattern = None
        self.trend_model = None
        self.residual_stats = None
        self.scaler = StandardScaler()

    def _decompose_simple(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple seasonal decomposition."""
        n = len(data)

        # Trend (moving average)
        if n >= self.seasonal_period * 2:
            trend = (
                pd.Series(data)
                .rolling(window=self.seasonal_period, center=True, min_periods=1)
                .mean()
                .values
            )
        else:
            # Linear trend for short series
            x = np.arange(n)
            trend = np.polyval(np.polyfit(x, data, 1), x)

        # Detrended
        detrended = data - trend

        # Seasonal (average pattern)
        if n >= self.seasonal_period:
            seasonal = np.zeros_like(data)
            for i in range(self.seasonal_period):
                season_values = detrended[i :: self.seasonal_period]
                seasonal[i :: self.seasonal_period] = np.mean(season_values)
        else:
            seasonal = np.zeros_like(data)

        # Residual
        residual = data - trend - seasonal

        return trend, seasonal, residual

    def fit(self, data: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fit decomposition model."""
        if len(data) < self.seasonal_period:
            logger.warning("Not enough data for seasonal decomposition")
            self.seasonal_pattern = np.zeros(self.seasonal_period)
        else:
            # Decompose training data
            trend, seasonal, residual = self._decompose_simple(data)

            # Store seasonal pattern
            self.seasonal_pattern = seasonal[: self.seasonal_period].copy()

            # Fit trend model (simple linear)
            x = np.arange(len(data))
            self.trend_model = np.polyfit(x, trend, 1)

            # Compute residual statistics for thresholding
            self.residual_stats = {
                "mean": np.mean(residual),
                "std": np.std(residual),
                "percentiles": np.percentile(residual, [5, 95]),
            }

        self.is_fitted = True

    def predict_scores(self, data: np.ndarray) -> np.ndarray:
        """Generate anomaly scores based on residuals."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")

        n = len(data)
        scores = np.zeros(n)

        # Decompose new data
        trend, seasonal, residual = self._decompose_simple(data)

        # Score based on residual magnitude
        if self.residual_stats["std"] > 0:
            # Standardized residual magnitude
            scores = (
                np.abs(residual - self.residual_stats["mean"])
                / self.residual_stats["std"]
            )
        else:
            scores = np.abs(residual)

        # Apply adaptive thresholding
        scores = self._apply_adaptive_threshold(scores)

        return scores

    def _apply_adaptive_threshold(self, scores: np.ndarray) -> np.ndarray:
        """Apply time-of-day adaptive thresholding."""
        n = len(scores)
        adaptive_scores = scores.copy()

        # Apply time-of-day weights (simple hour-based pattern)
        for i in range(n):
            hour = (i % self.seasonal_period) * 24 // self.seasonal_period

            # Higher sensitivity during typical low-consumption hours
            if 2 <= hour <= 6:  # Early morning
                adaptive_scores[i] *= 1.2
            elif 18 <= hour <= 22:  # Evening peak
                adaptive_scores[i] *= 0.8

        return adaptive_scores


class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """Statistical anomaly detector using rolling statistics."""

    def __init__(
        self,
        window_size: int = 48,  # 24 hours
        n_sigma: float = 3.0,
        contamination: float = 0.05,
    ):
        super().__init__("statistical")
        self.window_size = window_size
        self.n_sigma = n_sigma
        self.contamination = contamination
        self.baseline_stats = None

    def fit(self, data: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fit statistical model."""
        # Compute baseline statistics
        self.baseline_stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
        }

        self.is_fitted = True

    def predict_scores(self, data: np.ndarray) -> np.ndarray:
        """Generate anomaly scores using rolling statistics."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")

        n = len(data)
        scores = np.zeros(n)

        # Rolling z-scores
        for i in range(n):
            start_idx = max(0, i - self.window_size + 1)
            window_data = data[start_idx : i + 1]

            if len(window_data) >= 5:  # Minimum window
                window_mean = np.mean(window_data)
                window_std = np.std(window_data)

                if window_std > 0:
                    z_score = abs(data[i] - window_mean) / window_std
                    scores[i] = z_score / self.n_sigma
                else:
                    scores[i] = 0.0
            else:
                # Use baseline stats for initial points
                if self.baseline_stats["std"] > 0:
                    z_score = (
                        abs(data[i] - self.baseline_stats["mean"])
                        / self.baseline_stats["std"]
                    )
                    scores[i] = z_score / self.n_sigma

        return scores


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """Ensemble of anomaly detectors."""

    def __init__(
        self, detectors: list[BaseAnomalyDetector], weights: list[float] | None = None
    ):
        super().__init__("ensemble")
        self.detectors = detectors
        self.weights = weights or [1.0 / len(detectors)] * len(detectors)

    def fit(self, data: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fit all detectors."""
        for detector in self.detectors:
            try:
                detector.fit(data, timestamps)
            except Exception as e:
                logger.warning(f"Failed to fit {detector.name}: {e}")

        self.is_fitted = True

    def predict_scores(self, data: np.ndarray) -> np.ndarray:
        """Generate ensemble anomaly scores."""
        all_scores = []
        weights = []

        for detector, weight in zip(self.detectors, self.weights, strict=False):
            try:
                scores = detector.predict_scores(data)
                all_scores.append(scores)
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Failed to predict with {detector.name}: {e}")

        if not all_scores:
            return np.zeros(len(data))

        # Weighted average
        all_scores = np.array(all_scores)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        return np.average(all_scores, axis=0, weights=weights)


def detect_anomaly_events(
    scores: np.ndarray,
    threshold: float = 0.5,
    min_duration: int = 2,
    max_gap: int = 1,
) -> list[dict]:
    """Convert anomaly scores to event list."""
    binary_anomalies = scores > threshold
    events = []

    i = 0
    while i < len(binary_anomalies):
        if binary_anomalies[i]:
            # Start of anomaly
            start = i

            # Find end (allowing small gaps)
            while i < len(binary_anomalies):
                if binary_anomalies[i]:
                    i += 1
                else:
                    # Check if gap is small enough to bridge
                    gap_start = i
                    while i < len(binary_anomalies) and not binary_anomalies[i]:
                        i += 1
                        if i - gap_start > max_gap:
                            break

                    if i < len(binary_anomalies) and binary_anomalies[i]:
                        # Gap was small, continue anomaly
                        continue
                    else:
                        # End of anomaly
                        break

            end = min(i - 1, len(binary_anomalies) - 1)
            duration = end - start + 1

            if duration >= min_duration:
                events.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "max_score": np.max(scores[start : end + 1]),
                        "avg_score": np.mean(scores[start : end + 1]),
                    }
                )
        else:
            i += 1

    return events


def create_default_detectors() -> dict[str, BaseAnomalyDetector]:
    """Create default set of anomaly detectors."""
    return {
        "decomposition": DecompositionAnomalyDetector(
            seasonal_period=48,
            contamination=0.05,
        ),
        "statistical": StatisticalAnomalyDetector(
            window_size=48,
            n_sigma=3.0,
        ),
        "ensemble": EnsembleAnomalyDetector(
            [
                DecompositionAnomalyDetector(seasonal_period=48),
                StatisticalAnomalyDetector(window_size=48),
            ],
            weights=[0.7, 0.3],
        ),
    }
