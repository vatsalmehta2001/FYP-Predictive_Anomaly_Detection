"""Baseline forecasting models."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Base class for forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, history: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fit the model on historical data."""
        pass

    @abstractmethod
    def predict(
        self,
        history: np.ndarray,
        steps: int,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate forecasts for given number of steps."""
        pass

    def predict_quantiles(
        self,
        history: np.ndarray,
        steps: int,
        quantiles: list = None,
        timestamps: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        """Generate quantile forecasts (default: point forecast for all quantiles)."""
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        point_forecast = self.predict(history, steps, timestamps)
        return {q: point_forecast.copy() for q in quantiles}


class SeasonalNaive(BaseForecaster):
    """Seasonal naive forecaster - repeats values from same time last period."""

    def __init__(self, seasonal_period: int = 48):  # 24 hours at 30-min resolution
        super().__init__("seasonal_naive")
        self.seasonal_period = seasonal_period

    def fit(self, history: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Seasonal naive doesn't require fitting."""
        self.is_fitted = True

    def predict(
        self,
        history: np.ndarray,
        steps: int,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Repeat values from same time last seasonal period."""
        if len(history) < self.seasonal_period:
            # Fallback to last value if not enough history
            return np.full(steps, history[-1])

        # Extract seasonal pattern
        seasonal_pattern = history[-self.seasonal_period :]

        # Repeat pattern for forecast horizon
        forecast = []
        for i in range(steps):
            forecast.append(seasonal_pattern[i % self.seasonal_period])

        return np.array(forecast)


class LinearTrendForecaster(BaseForecaster):
    """Linear forecaster with trend and calendar features."""

    def __init__(self, include_trend: bool = True, seasonal_period: int = 48):
        super().__init__("linear_trend")
        self.include_trend = include_trend
        self.seasonal_period = seasonal_period
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.seasonal_components = None

    def _create_features(
        self, values: np.ndarray, timestamps: np.ndarray | None = None
    ) -> np.ndarray:
        """Create features for linear model."""
        n = len(values)
        features = []

        # Trend
        if self.include_trend:
            features.append(np.arange(n).reshape(-1, 1))

        # Lags
        lag_features = []
        for lag in [1, 2, 24, 48]:  # 30min, 1h, 12h, 24h lags
            if lag < n:
                lagged = np.concatenate([np.full(lag, values[0]), values[:-lag]])
                lag_features.append(lagged.reshape(-1, 1))

        if lag_features:
            features.extend(lag_features)

        # Hour of day (if timestamps available)
        if timestamps is not None:
            try:
                if isinstance(timestamps[0], pd.Timestamp | np.datetime64):
                    hours = pd.to_datetime(timestamps).hour
                    # Encode hour cyclically
                    hour_sin = np.sin(2 * np.pi * hours / 24).reshape(-1, 1)
                    hour_cos = np.cos(2 * np.pi * hours / 24).reshape(-1, 1)
                    features.extend([hour_sin, hour_cos])
            except Exception:
                pass  # Skip time features if parsing fails

        # Day of week encoding (simple pattern for samples)
        if len(values) >= 7 * self.seasonal_period:
            day_pattern = np.arange(n) % (7 * self.seasonal_period)
            day_sin = np.sin(
                2 * np.pi * day_pattern / (7 * self.seasonal_period)
            ).reshape(-1, 1)
            day_cos = np.cos(
                2 * np.pi * day_pattern / (7 * self.seasonal_period)
            ).reshape(-1, 1)
            features.extend([day_sin, day_cos])

        if not features:
            # Fallback: just trend
            features = [np.arange(n).reshape(-1, 1)]

        return np.hstack(features)

    def fit(self, history: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fit linear model."""
        if len(history) < 5:
            raise ValueError("Need at least 5 points for linear model")

        X = self._create_features(history, timestamps)
        y = history

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        logger.debug(f"Linear model fitted with {X.shape[1]} features")

    def predict(
        self,
        history: np.ndarray,
        steps: int,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate linear forecasts."""
        if not self.is_fitted:
            self.fit(history, timestamps)

        # Generate future timestamps if provided
        future_timestamps = None
        if timestamps is not None:
            try:
                last_time = pd.to_datetime(timestamps[-1])
                freq = pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[-2])
                future_timestamps = pd.date_range(
                    start=last_time + freq,
                    periods=steps,
                    freq=freq,
                ).values
            except Exception:
                pass

        # Extend history for feature creation
        extended_history = history.copy()
        forecasts = []

        for step in range(steps):
            # Create features for current extended history
            current_timestamps = None
            if timestamps is not None and future_timestamps is not None:
                current_timestamps = np.concatenate(
                    [timestamps, future_timestamps[:step]]
                )

            X = self._create_features(extended_history, current_timestamps)
            X_scaled = self.scaler.transform(X)

            # Predict next value
            next_pred = self.model.predict(X_scaled[-1:])
            forecasts.append(next_pred[0])

            # Extend history with prediction
            extended_history = np.append(extended_history, next_pred[0])

        return np.array(forecasts)


class EnsembleForecaster(BaseForecaster):
    """Simple ensemble of forecasters."""

    def __init__(self, forecasters: list, weights: list | None = None):
        super().__init__("ensemble")
        self.forecasters = forecasters
        self.weights = weights or [1.0 / len(forecasters)] * len(forecasters)

    def fit(self, history: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fit all forecasters."""
        for forecaster in self.forecasters:
            try:
                forecaster.fit(history, timestamps)
            except Exception as e:
                logger.warning(f"Failed to fit {forecaster.name}: {e}")

        self.is_fitted = True

    def predict(
        self,
        history: np.ndarray,
        steps: int,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate ensemble forecasts."""
        forecasts = []
        weights = []

        for forecaster, weight in zip(self.forecasters, self.weights, strict=False):
            try:
                pred = forecaster.predict(history, steps, timestamps)
                forecasts.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Failed to predict with {forecaster.name}: {e}")

        if not forecasts:
            # Fallback to last value
            return np.full(steps, history[-1])

        # Weighted average
        forecasts = np.array(forecasts)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        return np.average(forecasts, axis=0, weights=weights)

    def predict_quantiles(
        self,
        history: np.ndarray,
        steps: int,
        quantiles: list = None,
        timestamps: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        """Generate quantile forecasts using ensemble spread."""
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        all_forecasts = []

        for forecaster in self.forecasters:
            try:
                pred = forecaster.predict(history, steps, timestamps)
                all_forecasts.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict with {forecaster.name}: {e}")

        if not all_forecasts:
            point_forecast = np.full(steps, history[-1])
            return {q: point_forecast.copy() for q in quantiles}

        all_forecasts = np.array(all_forecasts)

        # Use ensemble spread for uncertainty
        quantile_forecasts = {}
        for q in quantiles:
            quantile_forecasts[q] = np.percentile(all_forecasts, q * 100, axis=0)

        return quantile_forecasts


def create_default_forecasters() -> dict[str, BaseForecaster]:
    """Create default set of forecasters."""
    return {
        "seasonal_naive": SeasonalNaive(seasonal_period=48),
        "linear_trend": LinearTrendForecaster(include_trend=True),
        "ensemble": EnsembleForecaster(
            [
                SeasonalNaive(seasonal_period=48),
                LinearTrendForecaster(include_trend=True),
            ],
            weights=[0.3, 0.7],
        ),
    }
