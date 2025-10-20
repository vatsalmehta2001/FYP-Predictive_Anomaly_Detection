"""Utility functions for data ingestion."""

import hashlib
import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def convert_power_to_energy(
    power_watts: np.ndarray, timestamps: np.ndarray, target_interval_mins: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert instantaneous power to energy using native sampling rate.

    Args:
        power_watts: Instantaneous power readings in watts
        timestamps: Corresponding timestamps (unix seconds or datetime64)
        target_interval_mins: Target interval for downsampling in minutes

    Returns:
        Tuple of (downsampled_timestamps, energy_kwh, interval_mins_array)
    """
    if len(power_watts) != len(timestamps):
        raise ValueError("Power and timestamp arrays must have same length")

    if len(power_watts) < 2:
        # Not enough data for interval calculation
        return (
            timestamps,
            power_watts / 1000.0,
            np.full(len(power_watts), target_interval_mins),
        )

    # Convert timestamps to pandas datetime if needed
    if isinstance(timestamps[0], int | float):
        # Unix timestamps
        ts_series = pd.to_datetime(timestamps, unit="s", utc=True)
    else:
        ts_series = pd.to_datetime(timestamps, utc=True)

    # Calculate native sampling interval
    time_diffs = np.diff(ts_series.astype(np.int64)) / 1e9  # Convert to seconds
    median_interval_sec = np.median(time_diffs[time_diffs > 0])

    if median_interval_sec <= 0:
        logger.warning("Cannot determine sampling interval, using 1 minute default")
        median_interval_sec = 60.0

    # Convert instantaneous power to energy for each sample
    # Energy = Power × Time, converting W×s to kWh
    interval_hours = median_interval_sec / 3600.0
    energy_kwh_native = power_watts * interval_hours / 1000.0

    # Create DataFrame for resampling
    df = pd.DataFrame(
        {"energy_kwh": energy_kwh_native, "power_watts": power_watts}, index=ts_series
    )

    # Resample to target interval by summing energy (not averaging power!)
    target_freq = f"{target_interval_mins}min"
    resampled = df.resample(target_freq).agg(
        {
            "energy_kwh": "sum",  # Sum energy over interval
            "power_watts": "count",  # Count samples for quality check
        }
    )

    # Filter out empty bins
    valid_bins = resampled["power_watts"] > 0
    resampled = resampled[valid_bins]

    return (
        resampled.index.values,
        resampled["energy_kwh"].values,
        np.full(len(resampled), target_interval_mins),
    )


def calculate_data_quality_metrics(
    data: pd.DataFrame, timestamp_col: str = "ts_utc", value_col: str = "energy_kwh"
) -> dict[str, Any]:
    """Calculate data quality metrics for ingested data.

    Args:
        data: DataFrame with timestamp and value columns
        timestamp_col: Name of timestamp column
        value_col: Name of value column

    Returns:
        Dictionary with quality metrics
    """
    if data.empty:
        return {
            "missing_pct": 100.0,
            "duplicates": 0,
            "outliers_pct": 0.0,
            "gaps_count": 0,
            "negative_values": 0,
        }

    total_records = len(data)

    # Missing values
    missing_count = data[value_col].isna().sum()
    missing_pct = (missing_count / total_records) * 100.0

    # Duplicate timestamps
    duplicates = data[timestamp_col].duplicated().sum()

    # Outliers (values beyond 3 sigma)
    if data[value_col].std() > 0:
        z_scores = np.abs(
            (data[value_col] - data[value_col].mean()) / data[value_col].std()
        )
        outliers_count = (z_scores > 3).sum()
        outliers_pct = (outliers_count / total_records) * 100.0
    else:
        outliers_pct = 0.0

    # Time gaps (if data is sorted by timestamp)
    data_sorted = data.sort_values(timestamp_col)
    time_diffs = data_sorted[timestamp_col].diff()
    median_interval = time_diffs.median()

    if pd.notna(median_interval):
        # Count gaps larger than 2x median interval
        large_gaps = time_diffs > (median_interval * 2)
        gaps_count = large_gaps.sum()
    else:
        gaps_count = 0

    # Negative values (should not exist for energy)
    negative_values = (data[value_col] < 0).sum()

    return {
        "missing_pct": float(missing_pct),
        "duplicates": int(duplicates),
        "outliers_pct": float(outliers_pct),
        "gaps_count": int(gaps_count),
        "negative_values": int(negative_values),
    }


def calculate_sha256(data: bytes) -> str:
    """Calculate SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def ensure_monotonic_utc_timestamps(
    df: pd.DataFrame, timestamp_col: str = "ts_utc", entity_col: str = "entity_id"
) -> pd.DataFrame:
    """Ensure timestamps are monotonic within each entity and remove duplicates.

    Args:
        df: DataFrame with timestamp and entity columns
        timestamp_col: Name of timestamp column
        entity_col: Name of entity column

    Returns:
        DataFrame with monotonic timestamps per entity
    """
    if df.empty:
        return df

    original_count = len(df)

    # Sort by entity and timestamp
    df_sorted = df.sort_values([entity_col, timestamp_col])

    # Remove duplicate timestamps within each entity
    df_dedupe = df_sorted.drop_duplicates(
        subset=[entity_col, timestamp_col], keep="first"
    )

    # Check for remaining non-monotonic timestamps
    for entity in df_dedupe[entity_col].unique():
        entity_df = df_dedupe[df_dedupe[entity_col] == entity]
        timestamps = entity_df[timestamp_col]

        if not timestamps.is_monotonic_increasing:
            logger.warning(f"Non-monotonic timestamps detected for entity {entity}")
            # Sort within entity
            entity_indices = entity_df.index
            df_dedupe.loc[entity_indices] = entity_df.sort_values(timestamp_col)

    removed_count = original_count - len(df_dedupe)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate/non-monotonic timestamps")

    return df_dedupe


class RateLimitedSession:
    """HTTP session with rate limiting, caching, and retry logic."""

    def __init__(
        self,
        rate_limit: float = 1.0,
        cache_dir: str | None = None,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ):
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.cache_dir = cache_dir

        # Setup session with retry strategy
        self.session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if cache_dir:
            import os

            os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, url: str, headers: dict[str, str]) -> str | None:
        """Get cache file path for URL and headers."""
        if not self.cache_dir:
            return None

        # Create cache key from URL and relevant headers
        cache_key = hashlib.md5(
            (url + json.dumps(sorted(headers.items()))).encode()
        ).hexdigest()

        return f"{self.cache_dir}/{cache_key}.json"

    def _load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
        """Load response from cache if valid."""
        try:
            import os

            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    cached = json.load(f)

                # Check if cache is still valid (simple time-based for now)
                cache_time = cached.get("cached_at", 0)
                max_age = 3600  # 1 hour cache

                if time.time() - cache_time < max_age:
                    return cached
        except Exception as e:
            logger.debug(f"Cache read error: {e}")

        return None

    def _save_to_cache(self, cache_path: str, response_data: dict[str, Any]) -> None:
        """Save response to cache."""
        try:
            response_data["cached_at"] = time.time()
            with open(cache_path, "w") as f:
                json.dump(response_data, f)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        force_refresh: bool = False,
        **kwargs,
    ) -> requests.Response:
        """Make rate-limited GET request with caching."""
        headers = headers or {}

        # Check cache first (unless force refresh)
        cache_path = self._get_cache_path(url, headers)
        if cache_path and not force_refresh:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                logger.debug(f"Using cached response for {url}")
                # Create mock response from cached data
                response = requests.Response()
                response.status_code = cached_data["status_code"]
                response._content = cached_data["content"].encode()
                response.headers.update(cached_data["headers"])
                return response

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Make request
        logger.debug(f"Making request to {url}")
        response = self.session.get(url, headers=headers, params=params, **kwargs)
        self.last_request_time = time.time()

        # Cache successful responses
        if cache_path and response.status_code == 200:
            cache_data = {
                "status_code": response.status_code,
                "content": response.text,
                "headers": dict(response.headers),
                "url": url,
            }
            self._save_to_cache(cache_path, cache_data)

        return response
