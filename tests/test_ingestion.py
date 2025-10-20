"""Tests for data ingestion module."""

from datetime import UTC, datetime

import pytest

from fyp.ingestion.schema import EnergyReading, get_partition_keys


class TestSchema:
    """Test unified schema validation."""

    def test_valid_reading(self):
        """Test creating a valid energy reading."""
        reading = EnergyReading(
            dataset="lcl",
            entity_id="household_123",
            ts_utc=datetime(2023, 1, 1, 12, 0, tzinfo=UTC),
            interval_mins=30,
            energy_kwh=0.5,
            source="test.csv",
            extras={"acorn": "A1"},
        )

        assert reading.dataset == "lcl"
        assert reading.entity_id == "household_123"
        assert reading.energy_kwh == 0.5
        assert reading.extras["acorn"] == "A1"

    def test_invalid_dataset(self):
        """Test invalid dataset name."""
        with pytest.raises(ValueError):
            EnergyReading(
                dataset="invalid",
                entity_id="test",
                ts_utc=datetime.now(UTC),
                interval_mins=30,
                energy_kwh=0.5,
                source="test",
            )

    def test_negative_energy(self):
        """Test negative energy validation."""
        with pytest.raises(ValueError):
            EnergyReading(
                dataset="lcl",
                entity_id="test",
                ts_utc=datetime.now(UTC),
                interval_mins=30,
                energy_kwh=-0.5,
                source="test",
            )

    def test_naive_timestamp(self):
        """Test timezone-naive timestamp rejection."""
        with pytest.raises(ValueError, match="timezone-aware"):
            EnergyReading(
                dataset="lcl",
                entity_id="test",
                ts_utc=datetime(2023, 1, 1, 12, 0),  # No timezone
                interval_mins=30,
                energy_kwh=0.5,
                source="test",
            )

    def test_invalid_interval(self):
        """Test invalid interval validation."""
        with pytest.raises(ValueError):
            EnergyReading(
                dataset="lcl",
                entity_id="test",
                ts_utc=datetime.now(UTC),
                interval_mins=7,  # Not a valid interval
                energy_kwh=0.5,
                source="test",
            )

    def test_partition_keys(self):
        """Test partition key extraction."""
        reading = EnergyReading(
            dataset="ukdale",
            entity_id="house_1",
            ts_utc=datetime(2023, 3, 15, tzinfo=UTC),
            interval_mins=1,
            energy_kwh=0.1,
            source="test",
        )

        keys = get_partition_keys(reading)
        assert keys == {
            "dataset": "ukdale",
            "year": "2023",
            "month": "03",
        }
