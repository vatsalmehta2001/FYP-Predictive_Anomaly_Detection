"""Ingestor for UK-DALE dataset."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseIngestor, ensure_timezone_aware
from .schema import EnergyReading
from .utils import (
    convert_power_to_energy,
)


class UKDALEIngestor(BaseIngestor):
    """Ingest UK-DALE HDF5 data."""

    def __init__(self, *args, downsample_30min: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsample_30min = downsample_30min
        self.logger = self._get_logger()

    def _get_logger(self):
        import logging

        return logging.getLogger(self.__class__.__name__)

    def get_input_files(self) -> list[Path]:
        """Get input files based on mode."""
        if self.use_samples:
            sample_file = Path("data/samples/ukdale_sample.csv")
            if sample_file.exists():
                return [sample_file]
            else:
                raise FileNotFoundError(f"Sample file not found: {sample_file}")

        # Look for UK-DALE HDF5 file
        # input_root is already data/raw/ukdale, no need to add another 'ukdale'
        ukdale_dir = self.input_root
        h5_files = list(ukdale_dir.glob("*.h5"))

        if not h5_files:
            raise FileNotFoundError(f"No UK-DALE HDF5 files found in {ukdale_dir}")

        return h5_files[:1]  # Process first file

    def read_raw_data(self) -> Iterator[dict[str, Any]]:
        """Read UK-DALE data and yield records."""
        files = self.get_input_files()

        for file_path in files:
            if file_path.suffix == ".csv":
                # Sample format
                yield from self._read_sample_format(file_path)
            else:
                # HDF5 format
                yield from self._read_h5_format(file_path)

    def _read_sample_format(self, file_path: Path) -> Iterator[dict[str, Any]]:
        """Read sample CSV format."""
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            yield {
                "house_id": int(row["house_id"]),
                "timestamp": pd.to_datetime(row["timestamp"]),
                "value_watts": float(row["value_watts"]),
                "channel": "aggregate",
                "source": file_path.name,
            }

    def _read_h5_format(self, file_path: Path) -> Iterator[dict[str, Any]]:
        """Read UK-DALE HDF5 format (pandas HDFStore) with proper energy conversion."""
        import warnings

        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        self.logger.info(f"Opening UK-DALE HDF5 file: {file_path}")

        # Use pandas HDFStore for reading
        with pd.HDFStore(file_path, mode="r") as store:
            # Calculate file hash for provenance
            file_hash = self._calculate_file_hash(file_path)

            # Get all keys (tables) in the store
            keys = store.keys()
            self.logger.info(f"Found {len(keys)} tables in HDF5 file")

            # Process each building
            buildings = set()
            for key in keys:
                if "/building" in key:
                    building_num = key.split("/building")[1].split("/")[0]
                    buildings.add(int(building_num))

            self.logger.info(f"Processing {len(buildings)} buildings")

            for house_id in sorted(buildings):
                # Find all meters for this building
                building_pattern = f"/building{house_id}/elec/meter"
                meter_keys = [k for k in keys if k.startswith(building_pattern)]

                self.logger.info(f"Building{house_id}: found {len(meter_keys)} meters")

                for meter_key in meter_keys:
                    # Extract meter number
                    meter_id = int(meter_key.split("meter")[-1].replace("/table", ""))

                    # Determine appliance type
                    appliance = (
                        "aggregate" if meter_id == 1 else f"appliance_{meter_id}"
                    )

                    try:
                        # Read the table for this meter
                        # Use chunksize to avoid loading entire table into memory
                        chunk_size = 50000

                        for chunk_num, df_chunk in enumerate(
                            store.select(meter_key, chunksize=chunk_size)
                        ):
                            # Skip empty chunks
                            if df_chunk.empty:
                                continue

                            # DataFrame has MultiIndex columns like ('power', 'active') or ('power', 'apparent')
                            # and DatetimeIndex for timestamps

                            # Find power column (prefer 'active', fall back to 'apparent')
                            power_col = None
                            if ("power", "active") in df_chunk.columns:
                                power_col = ("power", "active")
                            elif ("power", "apparent") in df_chunk.columns:
                                power_col = ("power", "apparent")
                            else:
                                # Try to find any column with 'power' in it
                                power_cols = [
                                    col
                                    for col in df_chunk.columns
                                    if "power" in str(col).lower()
                                ]
                                if power_cols:
                                    power_col = power_cols[0]
                                else:
                                    self.logger.warning(
                                        f"No power column in {meter_key}, columns: {df_chunk.columns.tolist()}"
                                    )
                                    continue

                            # Extract timestamps from index (already datetime)
                            timestamps = df_chunk.index

                            # Extract power values in Watts
                            power_watts = df_chunk[power_col].values

                            # Convert timestamps to UTC seconds for processing
                            timestamps_sec = timestamps.astype("int64") / 1e9

                            # Calculate time differences to determine intervals
                            if len(timestamps_sec) > 1:
                                time_diffs = np.diff(timestamps_sec)
                                median_interval_sec = (
                                    np.median(time_diffs[time_diffs > 0])
                                    if len(time_diffs[time_diffs > 0]) > 0
                                    else 6.0
                                )
                            else:
                                median_interval_sec = (
                                    6.0  # Default 6 seconds for UK-DALE
                                )

                            # Skip if this is test/sample mode and we've processed enough
                            if self.use_samples and chunk_num > 0:
                                break

                            if self.downsample_30min:
                                # Downsample to 30-minute intervals
                                (
                                    ts_30min,
                                    energy_kwh_30min,
                                    intervals,
                                ) = convert_power_to_energy(
                                    power_watts, timestamps_sec, target_interval_mins=30
                                )

                                for ts, energy, interval in zip(
                                    ts_30min, energy_kwh_30min, intervals, strict=False
                                ):
                                    yield {
                                        "house_id": house_id,
                                        "meter_id": meter_id,
                                        "timestamp": pd.to_datetime(
                                            ts, unit="s", utc=True
                                        ),
                                        "energy_kwh": float(energy),
                                        "interval_mins": int(interval),
                                        "channel": appliance,
                                        "source": f"building{house_id}/meter{meter_id}",
                                        "file_hash": file_hash,
                                    }
                            else:
                                # Native resolution: sample every Nth record to avoid excessive data
                                sample_rate = 10 if not self.use_samples else 1

                                for i in range(0, len(timestamps_sec), sample_rate):
                                    # Convert power to energy for this interval
                                    interval_hours = median_interval_sec / 3600.0
                                    energy_kwh = (
                                        power_watts[i] * interval_hours
                                    ) / 1000.0

                                    yield {
                                        "house_id": house_id,
                                        "meter_id": meter_id,
                                        "timestamp": pd.to_datetime(
                                            timestamps_sec[i], unit="s", utc=True
                                        ),
                                        "energy_kwh": float(energy_kwh),
                                        "interval_mins": int(median_interval_sec / 60),
                                        "channel": appliance,
                                        "source": f"building{house_id}/meter{meter_id}",
                                        "file_hash": file_hash,
                                    }

                    except Exception as e:
                        self.logger.error(
                            f"Error processing building{house_id}/meter{meter_id}: {e}"
                        )
                        import traceback

                        self.logger.debug(traceback.format_exc())
                        continue

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for provenance."""
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                hasher = hashlib.sha256()
                chunk_size = 8192
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
                return hasher.hexdigest()[:16]  # First 16 chars for brevity
        except Exception as e:
            self.logger.warning(f"Could not calculate file hash: {e}")
            return "unknown"

    def transform_record(self, record: dict[str, Any]) -> EnergyReading | None:
        """Transform UK-DALE record to unified schema."""
        try:
            # Convert timestamp to UTC
            ts_utc = ensure_timezone_aware(record["timestamp"])

            # Energy is already calculated correctly in _read_h5_format
            energy_kwh = record.get("energy_kwh", 0.0)
            interval_mins = record.get("interval_mins", 1)

            # For legacy records (sample format), calculate energy
            if "value_watts" in record and "energy_kwh" not in record:
                energy_kwh = (record["value_watts"] / 1000.0) * (interval_mins / 60.0)

            # Build enhanced extras with provenance
            extras = {
                "channel": record["channel"],
                "source_uri": f"ukdale.h5/{record['source']}",
                "ingestion_version": "v2.0",
            }

            if "meter_id" in record:
                extras["meter_id"] = record["meter_id"]
            if "file_hash" in record:
                extras["sha256"] = record["file_hash"]

            # Create entity ID
            entity_id = f"house_{record['house_id']}"
            if record["channel"] != "aggregate":
                entity_id += f"_{record['channel']}"

            return EnergyReading(
                dataset="ukdale",
                entity_id=entity_id,
                ts_utc=ts_utc,
                interval_mins=interval_mins,
                energy_kwh=energy_kwh,
                source=record["source"],
                extras=extras,
            )

        except Exception as e:
            self.logger.debug(f"Transform error: {e}")
            return None

    def run(self) -> None:
        """Run ingestion with optional downsampling."""
        # Note: Downsampling is now done during read phase (in _read_h5_format)
        # when downsample_30min=True, so we don't need post-processing
        super().run()

    def _create_downsampled_version(self) -> None:
        """Create 30-minute downsampled version from native data."""
        self.logger.info("Creating 30-minute downsampled version")

        # Read native resolution data
        native_path = self.output_root / "dataset=ukdale"
        if not native_path.exists():
            self.logger.warning("No native data found to downsample")
            return

        # Read all native data
        df = pd.read_parquet(native_path)

        # Parse extras JSON
        df["extras"] = df["extras"].apply(json.loads)

        # Group by entity and 30-min window
        df["ts_30min"] = df["ts_utc"].dt.floor("30min")

        # Aggregate energy
        grouped = (
            df.groupby(["entity_id", "ts_30min"])
            .agg(
                {
                    "energy_kwh": "sum",
                    "source": "first",
                    "extras": "first",
                }
            )
            .reset_index()
        )

        # Create new readings
        downsampled_readings = []
        for _, row in grouped.iterrows():
            reading = EnergyReading(
                dataset="ukdale",
                entity_id=row["entity_id"],
                ts_utc=row["ts_30min"],
                interval_mins=30,
                energy_kwh=row["energy_kwh"],
                source=row["source"] + "_30min",
                extras=row["extras"],
            )
            downsampled_readings.append(reading)

        # Write to separate directory
        output_dir = self.output_root.parent / "ukdale_30min"
        self.write_parquet_batch(downsampled_readings, output_dir)

        self.logger.info(f"Wrote {len(downsampled_readings)} downsampled records")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest UK-DALE dataset")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw"),
        help="Input root directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Output root directory",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use sample data instead of full dataset",
    )
    parser.add_argument(
        "--downsample-30min",
        action="store_true",
        default=True,
        help="Create 30-minute downsampled version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output",
    )

    args = parser.parse_args()

    ingestor = UKDALEIngestor(
        input_root=args.input_root,
        output_root=args.output_root,
        use_samples=args.use_samples,
        downsample_30min=args.downsample_30min,
        dry_run=args.dry_run,
    )
    ingestor.run()


if __name__ == "__main__":
    main()
