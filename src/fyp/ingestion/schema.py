"""Unified schema definition for energy time series data."""

from datetime import datetime
from typing import Any, Literal

import pyarrow as pa
from pydantic import BaseModel, Field, field_validator


class EnergyReading(BaseModel):
    """Unified schema for energy consumption readings across all datasets."""

    dataset: Literal["lcl", "ukdale", "ssen"]
    entity_id: str = Field(..., description="Household ID or Feeder ID")
    ts_utc: datetime = Field(..., description="UTC timestamp")
    interval_mins: int = Field(..., gt=0, description="Reading interval in minutes")
    energy_kwh: float = Field(..., ge=0, description="Energy consumption in kWh")
    source: str = Field(..., description="File or API resource identifier")
    extras: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Optional metadata (postcode, feeder name, appliance, etc.)",
    )

    @field_validator("ts_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware and in UTC."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v

    @field_validator("interval_mins")
    @classmethod
    def validate_interval(cls, v: int) -> int:
        """Validate common energy reading intervals."""
        valid_intervals = {1, 5, 10, 15, 30, 60}
        if v not in valid_intervals:
            raise ValueError(f"Interval must be one of {valid_intervals}")
        return v


# PyArrow schema for efficient Parquet storage
UNIFIED_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("entity_id", pa.string()),
        ("ts_utc", pa.timestamp("ns", tz="UTC")),
        ("interval_mins", pa.int8()),
        ("energy_kwh", pa.float32()),
        ("source", pa.string()),
        ("extras", pa.string()),  # JSON string for flexibility
    ]
)


def get_partition_keys(reading: EnergyReading) -> dict[str, str]:
    """Extract partition keys from a reading."""
    return {
        "dataset": reading.dataset,
        "year": str(reading.ts_utc.year),
        "month": f"{reading.ts_utc.month:02d}",
    }
