"""Utility functions for self-play training system."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_sliding_windows(
    data: np.ndarray,
    context_length: int = 336,  # 7 days at 30-min intervals
    forecast_horizon: int = 48,  # 24 hours at 30-min intervals
    stride: int = 24,  # 12 hours stride
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create sliding windows for time series training.

    Args:
        data: Time series data array
        context_length: Number of historical points for context
        forecast_horizon: Number of points to forecast
        stride: Step size between windows

    Returns:
        List of (context, target) pairs
    """
    windows = []
    total_length = context_length + forecast_horizon

    for i in range(0, len(data) - total_length + 1, stride):
        context = data[i : i + context_length]
        target = data[i + context_length : i + total_length]
        windows.append((context, target))

    logger.debug(f"Created {len(windows)} windows from {len(data)} data points")
    return windows


def normalize_consumption(
    data: np.ndarray, method: str = "standard", stats: dict[str, float] | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize energy consumption data.

    Args:
        data: Raw consumption values
        method: "standard" (z-score) or "minmax"
        stats: Pre-computed statistics for consistent normalization

    Returns:
        (normalized_data, statistics_dict)
    """
    if method == "standard":
        if stats is None:
            mean = np.mean(data)
            std = np.std(data) + 1e-6  # Avoid division by zero
            stats = {"mean": mean, "std": std}
        else:
            mean = stats["mean"]
            std = stats["std"]

        normalized = (data - mean) / std

    elif method == "minmax":
        if stats is None:
            min_val = np.min(data)
            max_val = np.max(data)
            stats = {"min": min_val, "max": max_val}
        else:
            min_val = stats["min"]
            max_val = stats["max"]

        normalized = (data - min_val) / (max_val - min_val + 1e-6)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, stats


def denormalize_consumption(
    data: np.ndarray, stats: dict[str, float], method: str = "standard"
) -> np.ndarray:
    """Denormalize energy consumption data.

    Args:
        data: Normalized consumption values
        stats: Statistics used for normalization
        method: "standard" or "minmax"

    Returns:
        Denormalized data
    """
    if method == "standard":
        return data * stats["std"] + stats["mean"]
    elif method == "minmax":
        return data * (stats["max"] - stats["min"]) + stats["min"]
    else:
        raise ValueError(f"Unknown denormalization method: {method}")


def compute_temporal_features(
    timestamps: pd.DatetimeIndex, include_cyclical: bool = True
) -> np.ndarray:
    """Extract temporal features from timestamps.

    Args:
        timestamps: Datetime index
        include_cyclical: Whether to include sine/cosine encodings

    Returns:
        Feature array with shape (n_timestamps, n_features)
    """
    features = []

    # Basic temporal features
    hour = timestamps.hour + timestamps.minute / 60.0
    day_of_week = timestamps.dayofweek
    day_of_month = timestamps.day
    month = timestamps.month

    features.extend([hour, day_of_week, day_of_month, month])

    if include_cyclical:
        # Cyclical encoding for periodic features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        features.extend([hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos])

    return np.column_stack(features)


def apply_scenario_transformation(
    baseline: np.ndarray,
    scenario_type: str,
    magnitude: float,
    duration: int,
    start_idx: int = 0,
) -> np.ndarray:
    """Apply scenario transformation to baseline consumption.

    Args:
        baseline: Original consumption time series
        scenario_type: Type of scenario (EV_SPIKE, COLD_SNAP, etc.)
        magnitude: Intensity multiplier
        duration: Number of intervals affected
        start_idx: Start index for transformation

    Returns:
        Transformed time series
    """
    transformed = baseline.copy()
    end_idx = min(start_idx + duration, len(baseline))

    if scenario_type == "EV_SPIKE":
        # Add EV charging spike (typically 3.5-7 kW)
        spike_magnitude = magnitude * 3.5  # kW to kWh per 30min
        transformed[start_idx:end_idx] += (
            spike_magnitude / 2
        )  # Convert to 30-min consumption

    elif scenario_type == "COLD_SNAP":
        # Multiply baseline by magnitude during cold snap
        transformed[start_idx:end_idx] *= magnitude

    elif scenario_type == "PEAK_SHIFT":
        # Shift evening peak by ±2 hours
        shift_hours = int(magnitude * 2)  # magnitude controls shift direction/amount
        shift_intervals = shift_hours * 2  # Convert to 30-min intervals

        # Find evening peak hours (typically 17:00-21:00)
        peak_mask = np.zeros(len(baseline), dtype=bool)
        for i in range(len(baseline)):
            hour = (i % 48) / 2  # Convert interval to hour of day
            if 17 <= hour <= 21:
                peak_mask[i] = True

        # Shift peak consumption
        if shift_intervals > 0:
            # Shift forward
            peak_values = transformed[peak_mask]
            if len(peak_values) > 0:
                transformed[peak_mask] *= 0.7  # Reduce original peak
                shift_mask = np.roll(peak_mask, shift_intervals)
                transformed[shift_mask] += peak_values.mean() * 0.3
        else:
            # Shift backward
            peak_values = transformed[peak_mask]
            if len(peak_values) > 0:
                transformed[peak_mask] *= 0.7
                shift_mask = np.roll(peak_mask, shift_intervals)
                transformed[shift_mask] += peak_values.mean() * 0.3

    elif scenario_type == "OUTAGE":
        # Zero consumption during outage
        transformed[start_idx:end_idx] = 0.0

    elif scenario_type == "MISSING_DATA":
        # Replace with NaN for missing data scenario
        transformed[start_idx:end_idx] = np.nan

    return transformed


def calculate_pinball_loss(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float
) -> float:
    """Calculate pinball loss for quantile regression.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        quantile: Target quantile (e.g., 0.1, 0.5, 0.9)

    Returns:
        Average pinball loss
    """
    errors = y_true - y_pred
    pinball = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(pinball)


def estimate_scenario_difficulty(
    scenario_type: str, magnitude: float, duration: int, historical_volatility: float
) -> float:
    """Estimate difficulty score for a proposed scenario.

    Args:
        scenario_type: Type of scenario
        magnitude: Intensity of change
        duration: Length of scenario
        historical_volatility: Baseline consumption volatility

    Returns:
        Difficulty score in [0, 1]
    """
    # Base difficulty by scenario type
    base_difficulty = {
        "EV_SPIKE": 0.3,  # Relatively easy - clear pattern
        "COLD_SNAP": 0.4,  # Moderate - weather-driven
        "PEAK_SHIFT": 0.6,  # Harder - temporal shift
        "OUTAGE": 0.2,  # Easy - zero consumption
        "MISSING_DATA": 0.5,  # Moderate - interpolation needed
    }

    difficulty = base_difficulty.get(scenario_type, 0.5)

    # Adjust for magnitude (larger changes are harder)
    magnitude_factor = min(abs(magnitude - 1.0), 1.0)
    difficulty += 0.2 * magnitude_factor

    # Adjust for duration (longer scenarios are harder)
    duration_factor = min(duration / 96, 1.0)  # Normalize by 2 days
    difficulty += 0.2 * duration_factor

    # Adjust for volatility (high volatility makes detection harder)
    volatility_factor = min(historical_volatility / 0.5, 1.0)
    difficulty += 0.1 * volatility_factor

    return np.clip(difficulty, 0.0, 1.0)


def create_scenario_mask(
    length: int, scenario_start: int, scenario_duration: int
) -> np.ndarray:
    """Create boolean mask for scenario application.

    Args:
        length: Total time series length
        scenario_start: Start index of scenario
        scenario_duration: Number of intervals affected

    Returns:
        Boolean mask array
    """
    mask = np.zeros(length, dtype=bool)
    end_idx = min(scenario_start + scenario_duration, length)
    mask[scenario_start:end_idx] = True
    return mask
