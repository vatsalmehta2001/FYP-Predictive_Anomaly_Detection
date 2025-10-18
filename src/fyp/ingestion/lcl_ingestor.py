"""Ingestor for London Smart Meters (LCL) dataset."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from .base import BaseIngestor, ensure_timezone_aware
from .schema import EnergyReading
from .utils import calculate_data_quality_metrics, calculate_sha256


class LCLIngestor(BaseIngestor):
    """Ingest London Smart Meters CSV data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_input_files(self) -> list[Path]:
        """Get input CSV files based on mode."""
        if self.use_samples:
            sample_file = Path("data/samples/lcl_sample.csv")
            if sample_file.exists():
                return [sample_file]
            else:
                raise FileNotFoundError(f"Sample file not found: {sample_file}")
        
        # Look for main LCL data files
        lcl_dir = self.input_root / "lcl"
        patterns = ["*LCL*.csv", "*.csv"]
        
        files = []
        for pattern in patterns:
            files.extend(lcl_dir.glob(pattern))
        
        # Filter out metadata files
        files = [f for f in files if "information" not in f.name.lower()]
        
        if not files:
            raise FileNotFoundError(f"No LCL data files found in {lcl_dir}")
        
        return sorted(files)[:1]  # Process first file for now
    
    def read_raw_data(self) -> Iterator[Dict[str, Any]]:
        """Read LCL CSV files and yield records."""
        files = self.get_input_files()
        
        for file_path in files:
            self.logger.info(f"Reading {file_path}")
            
            # Detect format by reading header
            with open(file_path, "r") as f:
                header = f.readline().strip()
            
            if "household_id" in header:
                # Sample format
                yield from self._read_sample_format(file_path)
            else:
                # Full LCL format
                yield from self._read_lcl_format(file_path)
    
    def _read_sample_format(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Read sample CSV format."""
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            yield {
                "household_id": row["household_id"],
                "timestamp": pd.to_datetime(row["timestamp"]),
                "kwh_30m": float(row["kwh_30m"]),
                "source": file_path.name,
            }
    
    def _read_lcl_format(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Read full LCL CSV format with chunking."""
        # LCL format can be:
        # - LCLid,stdorToU,DateTime,KWH/hh (per half hour) [4 columns]
        # - LCLid,DateTime,KWH/hh,Acorn,Acorn_grouped [5 columns]
        chunk_size = 100000
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Dynamically handle column count
            num_cols = len(chunk.columns)
            
            if num_cols == 4:
                # Format: LCLid, stdorToU, DateTime, KWH/hh
                chunk.columns = ["household_id", "tariff_type", "timestamp", "kwh_30m"]
            elif num_cols == 5:
                # Format: LCLid, DateTime, KWH/hh, Acorn, Acorn_grouped
                chunk.columns = ["household_id", "timestamp", "kwh_30m", "acorn", "acorn_grouped"]
            else:
                self.logger.warning(f"Unexpected column count: {num_cols}")
                continue
            
            # Parse timestamp
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors='coerce')
            
            # Convert kwh_30m to float, handling any whitespace/errors
            chunk["kwh_30m"] = pd.to_numeric(chunk["kwh_30m"], errors='coerce')
            
            # Yield records
            for _, row in chunk.iterrows():
                # Skip rows with invalid data
                if pd.isna(row["timestamp"]) or pd.isna(row["kwh_30m"]):
                    continue
                    
                record = {
                    "household_id": str(row["household_id"]),
                    "timestamp": row["timestamp"],
                    "kwh_30m": float(row["kwh_30m"]),
                    "source": file_path.name,
                }
                
                # Add optional columns if present
                if "tariff_type" in row and pd.notna(row["tariff_type"]):
                    record["tariff_type"] = str(row["tariff_type"])
                if "acorn" in row and pd.notna(row["acorn"]):
                    record["acorn"] = str(row["acorn"])
                if "acorn_grouped" in row and pd.notna(row["acorn_grouped"]):
                    record["acorn_grouped"] = str(row["acorn_grouped"])
                
                yield record
    
    def transform_record(self, record: Dict[str, Any]) -> Optional[EnergyReading]:
        """Transform LCL record to unified schema."""
        try:
            # Convert timestamp to UTC
            ts_utc = ensure_timezone_aware(record["timestamp"])
            
            # Build enhanced extras with provenance
            extras = {
                "source_uri": f"lcl/{record['source']}",
                "ingestion_version": "v2.0",
            }
            
            # Add optional metadata fields
            if "tariff_type" in record:
                extras["tariff_type"] = record["tariff_type"]
            if "acorn" in record:
                extras["acorn"] = record["acorn"]
            if "acorn_grouped" in record:
                extras["acorn_grouped"] = record["acorn_grouped"]
            if "file_hash" in record:
                extras["sha256"] = record["file_hash"]
            
            return EnergyReading(
                dataset="lcl",
                entity_id=record["household_id"],
                ts_utc=ts_utc,
                interval_mins=30,
                energy_kwh=record["kwh_30m"],
                source=record["source"],
                extras=extras,
            )
            
        except Exception as e:
            self.logger.debug(f"Transform error: {e}")
            return None


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest LCL dataset")
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
        "--dry-run",
        action="store_true",
        help="Run without writing output",
    )
    
    args = parser.parse_args()
    
    ingestor = LCLIngestor(
        input_root=args.input_root,
        output_root=args.output_root,
        use_samples=args.use_samples,
        dry_run=args.dry_run,
    )
    ingestor.run()


if __name__ == "__main__":
    main()
