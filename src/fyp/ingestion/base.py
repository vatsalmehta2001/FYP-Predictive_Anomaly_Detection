"""Base utilities for data ingestion."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from .schema import UNIFIED_SCHEMA, EnergyReading, get_partition_keys
from .utils import calculate_data_quality_metrics

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """Base class for dataset-specific ingestors."""
    
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        use_samples: bool = False,
        dry_run: bool = False,
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.use_samples = use_samples
        self.dry_run = dry_run
        self.stats = {"processed": 0, "errors": 0, "skipped": 0}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logger
    
    @abstractmethod
    def read_raw_data(self) -> Iterator[Dict[str, Any]]:
        """Read raw data and yield records."""
        pass
    
    @abstractmethod
    def transform_record(self, record: Dict[str, Any]) -> Optional[EnergyReading]:
        """Transform raw record to unified schema."""
        pass
    
    def validate_reading(self, reading: EnergyReading) -> Optional[str]:
        """Validate a reading, return error message if invalid."""
        try:
            # Pydantic validation happens automatically
            # Additional business logic validation here
            
            # Check timestamp alignment for 30-min data
            if reading.interval_mins == 30:
                if reading.ts_utc.minute not in (0, 30):
                    return f"30-min reading not aligned: {reading.ts_utc}"
            
            # Check for future timestamps
            if reading.ts_utc > datetime.now(reading.ts_utc.tzinfo):
                return f"Future timestamp: {reading.ts_utc}"
            
            return None
            
        except Exception as e:
            return str(e)
    
    def write_parquet_batch(
        self,
        readings: List[EnergyReading],
        output_dir: Path
    ) -> None:
        """Write a batch of readings to Parquet files (simplified, no partitioning)."""
        if not readings:
            return
        
        # Convert to records
        records = []
        for reading in readings:
            record = reading.model_dump()
            # Convert extras dict to JSON string
            record["extras"] = json.dumps(record["extras"])
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Simple schema - no partitions
        schema = pa.schema([
            ("dataset", pa.string()),
            ("entity_id", pa.string()),
            ("ts_utc", pa.timestamp("ns", tz="UTC")),
            ("interval_mins", pa.int8()),
            ("energy_kwh", pa.float32()),
            ("source", pa.string()),
            ("extras", pa.string()),
        ])
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df, schema=schema)
        
        # Simple directory structure
        dataset_name = df['dataset'].iloc[0] if not df.empty else "unknown"
        dataset_dir = output_dir / f"{dataset_name}_data"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Use batch counter for predictable filenames
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        self._batch_counter += 1
        
        filename = f"batch_{self._batch_counter:06d}.parquet"
        filepath = dataset_dir / filename
        
        # Write the file
        pq.write_table(table, filepath, compression="snappy")
        
        logger.info(f"Wrote batch {self._batch_counter}: {len(table)} records → {filename}")
    
    def write_summary(self, output_dir: Path) -> None:
        """Write ingestion summary with final quality metrics."""
        # Calculate quality metrics on all data if available
        dataset_name = self.__class__.__name__.replace("Ingestor", "").lower()
        dataset_dir = output_dir / f"{dataset_name}_data"
        
        if dataset_dir.exists():
            try:
                # Read a sample to calculate quality metrics
                import polars as pl
                sample_df = pl.scan_parquet(str(dataset_dir / "*.parquet")).head(100000).collect()
                if len(sample_df) > 0:
                    # Convert to pandas for quality metrics calculation
                    sample_pd = sample_df.to_pandas()
                    quality_metrics = calculate_data_quality_metrics(sample_pd, 'ts_utc', 'energy_kwh')
                    self.stats.update(quality_metrics)
            except Exception as e:
                logger.warning(f"Could not calculate quality metrics: {e}")
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": dataset_name,
            "stats": self.stats,
            "use_samples": self.use_samples,
            "dry_run": self.dry_run,
        }
        
        summary_path = output_dir / "ingestion_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary written to {summary_path}")
    
    def run(self) -> None:
        """Main ingestion pipeline."""
        logger.info(f"Starting {self.__class__.__name__}")
        logger.info(f"Input: {self.input_root}, Output: {self.output_root}")
        logger.info(f"Samples: {self.use_samples}, Dry run: {self.dry_run}")
        
        batch = []
        batch_size = 10000
        
        try:
            for raw_record in self.read_raw_data():
                try:
                    # Transform to unified schema
                    reading = self.transform_record(raw_record)
                    if not reading:
                        self.stats["skipped"] += 1
                        continue
                    
                    # Validate
                    error = self.validate_reading(reading)
                    if error:
                        logger.debug(f"Validation error: {error}")
                        self.stats["errors"] += 1
                        continue
                    
                    batch.append(reading)
                    self.stats["processed"] += 1
                    
                    # Write batch if full
                    if len(batch) >= batch_size:
                        if not self.dry_run:
                            self.write_parquet_batch(batch, self.output_root)
                        batch = []
                        logger.info(f"Processed {self.stats['processed']} records")
                    
                except ValidationError as e:
                    logger.debug(f"Validation error: {e}")
                    self.stats["errors"] += 1
                except Exception as e:
                    logger.warning(f"Error processing record: {e}")
                    self.stats["errors"] += 1
            
            # Write final batch
            if batch and not self.dry_run:
                self.write_parquet_batch(batch, self.output_root)
            
            # Write summary
            if not self.dry_run:
                self.write_summary(self.output_root)
            
            logger.info(f"Ingestion complete: {self.stats}")
            
        except Exception as e:
            logger.error(f"Fatal error during ingestion: {e}")
            raise


def ensure_timezone_aware(
    dt: pd.Timestamp,
    source_tz: str = "Europe/London"
) -> datetime:
    """Convert timestamp to timezone-aware UTC datetime."""
    if dt.tz is None:
        # Localize to source timezone
        dt = dt.tz_localize(source_tz)
    # Convert to UTC
    return dt.tz_convert("UTC").to_pydatetime()


def round_to_interval(dt: datetime, interval_mins: int) -> datetime:
    """Round datetime down to nearest interval."""
    if interval_mins == 30:
        # Round to nearest 30 minutes
        minute = 30 if dt.minute >= 30 else 0
        return dt.replace(minute=minute, second=0, microsecond=0)
    elif interval_mins == 60:
        # Round to hour
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        # For other intervals, round down to nearest interval
        total_mins = dt.hour * 60 + dt.minute
        rounded_mins = (total_mins // interval_mins) * interval_mins
        hour = rounded_mins // 60
        minute = rounded_mins % 60
        return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
