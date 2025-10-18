"""Ingestor for SSEN feeder data via CKAN API and lookup CSV."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from urllib.parse import urljoin

import pandas as pd
import requests

from .base import BaseIngestor, ensure_timezone_aware
from .schema import EnergyReading
from .utils import RateLimitedSession, calculate_data_quality_metrics


class SSENIngestor(BaseIngestor):
    """Ingest SSEN feeder data from CKAN API and lookup CSV."""
    
    def __init__(
        self,
        *args,
        ckan_url: str = "https://data.ssen.co.uk",
        package_id: str = "low-voltage-feeder-data",
        api_key: Optional[str] = None,
        rate_limit: float = 1.0,  # seconds between requests
        force_refresh: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ckan_url = ckan_url
        self.package_id = package_id
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.force_refresh = force_refresh
        self.feeder_lookup = None
        self.logger = self._get_logger()
        
        # Initialize rate-limited session with caching
        cache_dir = str(self.output_root.parent / ".cache" / "ssen_api")
        self.session = RateLimitedSession(
            rate_limit=rate_limit,
            cache_dir=cache_dir,
            max_retries=3,
            backoff_factor=0.5,
        )
    
    def _get_logger(self):
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    def _load_feeder_lookup(self) -> pd.DataFrame:
        """Load feeder metadata from CSV."""
        lookup_path = self.input_root / "ssen" / "LV_FEEDER_LOOKUP.csv"
        
        if not lookup_path.exists():
            self.logger.warning(f"Feeder lookup not found: {lookup_path}")
            return pd.DataFrame()
        
        self.logger.info(f"Loading feeder lookup from {lookup_path}")
        df = pd.read_csv(lookup_path)
        
        # Standardize column names if needed
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        return df
    
    def _get_api_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"User-Agent": "FYP-Energy-Forecasting/0.1.0"}
        if self.api_key:
            headers["Authorization"] = self.api_key
        return headers
    
    def _api_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and caching."""
        url = urljoin(self.ckan_url, endpoint)
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=self._get_api_headers(),
                timeout=30,
                force_refresh=self.force_refresh,
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def _discover_resources(self) -> list[Dict[str, Any]]:
        """Discover available data resources."""
        try:
            # Get package metadata
            result = self._api_request(
                f"/api/3/action/package_show",
                params={"id": self.package_id}
            )
            
            if not result.get("success"):
                raise ValueError(f"Package not found: {self.package_id}")
            
            resources = result["result"].get("resources", [])
            
            # Filter for feeder time series data
            feeder_resources = [
                r for r in resources
                if "feeder" in r.get("name", "").lower()
                and r.get("format", "").upper() in ["CSV", "JSON"]
            ]
            
            return feeder_resources
            
        except Exception as e:
            self.logger.warning(f"API discovery failed: {e}")
            return []
    
    def read_raw_data(self) -> Iterator[Dict[str, Any]]:
        """Read SSEN feeder metadata from CSV.
        
        Note: This only processes the feeder lookup CSV (metadata).
        Time-series consumption data requires either:
        - Research partnership with SSEN
        - API access (currently not available for this project)
        - Pseudo-feeder generation from LCL aggregations (future work)
        """
        # Load feeder lookup - this is our only data source for now
        self.feeder_lookup = self._load_feeder_lookup()
        
        if self.use_samples:
            # Use sample data
            yield from self._read_sample_data()
        else:
            # Read metadata from CSV and generate placeholder records
            # NOTE: This is metadata only, not actual time-series data
            self.logger.info("Processing SSEN feeder metadata (no time-series data available)")
            yield from self._read_metadata_only()
    
    def _read_sample_data(self) -> Iterator[Dict[str, Any]]:
        """Read sample CSV data."""
        sample_path = Path("data/samples/ssen_sample.csv")
        
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        
        df = pd.read_csv(sample_path)
        for _, row in df.iterrows():
            yield {
                "feeder_id": row["feeder_id"],
                "timestamp": pd.to_datetime(row["timestamp"]),
                "wh_30m": float(row["wh_30m"]),
                "source": sample_path.name,
            }
    
    def _read_api_data(self) -> Iterator[Dict[str, Any]]:
        """Read data from CKAN API."""
        resources = self._discover_resources()
        
        if not resources:
            self.logger.warning("No resources found, using mock data")
            # Generate mock data for testing
            yield from self._generate_mock_data()
            return
        
        for resource in resources:
            resource_id = resource["id"]
            self.logger.info(f"Processing resource: {resource['name']}")
            
            # Paginate through data
            offset = 0
            limit = 1000
            
            while True:
                try:
                    result = self._api_request(
                        "/api/3/action/datastore_search",
                        params={
                            "resource_id": resource_id,
                            "limit": limit,
                            "offset": offset,
                        }
                    )
                    
                    if not result.get("success"):
                        break
                    
                    records = result["result"].get("records", [])
                    if not records:
                        break
                    
                    for record in records:
                        # Parse record based on expected format
                        yield self._parse_api_record(record, resource_id)
                    
                    offset += limit
                    
                    # Check if more data available
                    total = result["result"].get("total", 0)
                    if offset >= total:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error reading resource {resource_id}: {e}")
                    break
    
    def _parse_api_record(self, record: Dict, resource_id: str) -> Dict[str, Any]:
        """Parse API record to standard format."""
        # Adapt based on actual API response format
        # This is a placeholder implementation
        return {
            "feeder_id": record.get("feeder_id", "unknown"),
            "timestamp": pd.to_datetime(record.get("timestamp")),
            "wh_30m": float(record.get("energy_wh", 0)) if "energy_wh" in record else float(record.get("power_kw", 0)) * 500,  # Convert kW to Wh for 30min
            "source": f"api:{resource_id}",
            "retrieved_at": datetime.utcnow(),
        }
    
    def _read_metadata_only(self) -> Iterator[Dict[str, Any]]:
        """Read feeder metadata from CSV without time-series data.
        
        This generates a single metadata record per feeder with constraint information.
        Actual time-series consumption data is not available in this dataset.
        """
        if self.feeder_lookup is None or self.feeder_lookup.empty:
            self.logger.warning("No feeder lookup data available")
            return
        
        self.logger.info(f"Processing {len(self.feeder_lookup)} feeders from metadata CSV")
        
        # We don't have time-series data, so we won't yield consumption records
        # Instead, we save metadata separately
        metadata_path = self.output_root / "ssen_metadata.parquet"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata with constraints
        metadata_df = self.feeder_lookup.copy()
        
        # Add constraint columns if not present
        if 'voltage_nominal_v' not in metadata_df.columns:
            metadata_df['voltage_nominal_v'] = 230.0  # UK standard
        if 'voltage_tolerance_pct' not in metadata_df.columns:
            metadata_df['voltage_tolerance_pct'] = 10.0  # UK statutory ±10%
        if 'power_factor_min' not in metadata_df.columns:
            metadata_df['power_factor_min'] = 0.8  # Typical minimum
        if 'power_factor_max' not in metadata_df.columns:
            metadata_df['power_factor_max'] = 1.0
        
        # Save to parquet
        metadata_df.to_parquet(metadata_path, index=False)
        self.logger.info(f"Saved feeder metadata to {metadata_path}")
        self.logger.info(f"Metadata includes: voltage limits, capacity ratings, locations")
        self.logger.info("Note: Time-series consumption data not available - use for constraints only")
        
        # Don't yield any records since we don't have time-series data
        # This will result in 0 records processed, which is correct
        return
        yield  # Make this a generator
    
    def transform_record(self, record: Dict[str, Any]) -> Optional[EnergyReading]:
        """Transform SSEN record to unified schema."""
        try:
            # Convert timestamp to UTC
            ts_utc = ensure_timezone_aware(record["timestamp"])
            
            # Convert Wh to kWh
            energy_kwh = record["wh_30m"] / 1000.0
            
            # Build extras from feeder lookup
            extras = {}
            if self.feeder_lookup is not None and not self.feeder_lookup.empty:
                feeder_info = self.feeder_lookup[
                    self.feeder_lookup.get("feeder_id", "") == record["feeder_id"]
                ]
                
                if not feeder_info.empty:
                    row = feeder_info.iloc[0]
                    # Add relevant metadata
                    for col in ["feeder_name", "substation", "postcode_sector", "capacity_kva"]:
                        if col in row and pd.notna(row[col]):
                            extras[col] = str(row[col])
            
            # Add retrieval timestamp if from API
            if "retrieved_at" in record:
                extras["retrieved_at"] = record["retrieved_at"].isoformat()
            
            return EnergyReading(
                dataset="ssen",
                entity_id=record["feeder_id"],
                ts_utc=ts_utc,
                interval_mins=30,
                energy_kwh=energy_kwh,
                source=record["source"],
                extras=extras,
            )
            
        except Exception as e:
            self.logger.debug(f"Transform error: {e}")
            return None


def main():
    """CLI entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Ingest SSEN feeder data")
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
        help="Use sample data instead of API",
    )
    parser.add_argument(
        "--ckan-url",
        default=os.getenv("SSEN_CKAN_URL", "https://data.ssen.co.uk"),
        help="CKAN API base URL",
    )
    parser.add_argument(
        "--package-id",
        default="low-voltage-feeder-data",
        help="CKAN package ID",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SSEN_API_KEY"),
        help="API key for authentication",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of cached API responses",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output",
    )
    
    args = parser.parse_args()
    
    ingestor = SSENIngestor(
        input_root=args.input_root,
        output_root=args.output_root,
        use_samples=args.use_samples,
        ckan_url=args.ckan_url,
        package_id=args.package_id,
        api_key=args.api_key,
        force_refresh=args.force_refresh,
        dry_run=args.dry_run,
    )
    ingestor.run()


if __name__ == "__main__":
    main()
