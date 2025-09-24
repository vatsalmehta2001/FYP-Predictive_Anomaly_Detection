"""Data loader for unified Parquet datasets."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class EnergyDataLoader:
    """Load and prepare energy data from unified Parquet format."""
    
    def __init__(self, data_root: Path = Path("data/processed")):
        self.data_root = Path(data_root)
    
    def load_dataset(
        self,
        dataset: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        entities: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load dataset from partitioned Parquet."""
        dataset_path = self.data_root / f"dataset={dataset}"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Loading dataset: {dataset}")
        
        # Read partitioned data
        df = pd.read_parquet(dataset_path)
        
        # Parse extras JSON
        if "extras" in df.columns:
            df["extras"] = df["extras"].apply(
                lambda x: json.loads(x) if pd.notna(x) and x != "{}" else {}
            )
        
        # Convert timestamp
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        
        # Filter by date range
        if start_date:
            df = df[df["ts_utc"] >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            df = df[df["ts_utc"] <= pd.to_datetime(end_date, utc=True)]
        
        # Filter by entities
        if entities:
            df = df[df["entity_id"].isin(entities)]
        
        # Sort by entity and time
        df = df.sort_values(["entity_id", "ts_utc"])
        
        logger.info(f"Loaded {len(df)} records for {df['entity_id'].nunique()} entities")
        
        return df
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        split_by_time: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split."""
        if split_by_time:
            # Time-based split (last test_size fraction for each entity)
            train_list, test_list = [], []
            
            for entity_id in df["entity_id"].unique():
                entity_df = df[df["entity_id"] == entity_id].copy()
                
                if len(entity_df) < 10:  # Skip entities with too little data
                    continue
                
                split_idx = int(len(entity_df) * (1 - test_size))
                
                train_list.append(entity_df.iloc[:split_idx])
                test_list.append(entity_df.iloc[split_idx:])
            
            train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
            test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
        else:
            # Random split
            train_df = df.sample(frac=1-test_size, random_state=42)
            test_df = df.drop(train_df.index)
        
        logger.info(f"Split: {len(train_df)} train, {len(test_df)} test records")
        
        return train_df, test_df
    
    def create_forecasting_windows(
        self,
        df: pd.DataFrame,
        history_length: int = 48,  # 24 hours at 30-min resolution
        forecast_horizon: int = 48,  # 24 hours ahead
        step_size: int = 1,
    ) -> List[Dict]:
        """Create sliding windows for forecasting."""
        windows = []
        
        for entity_id in df["entity_id"].unique():
            entity_df = df[df["entity_id"] == entity_id].copy()
            
            if len(entity_df) < history_length + forecast_horizon:
                continue
            
            entity_df = entity_df.sort_values("ts_utc")
            
            for i in range(0, len(entity_df) - history_length - forecast_horizon + 1, step_size):
                history = entity_df.iloc[i:i + history_length]
                future = entity_df.iloc[i + history_length:i + history_length + forecast_horizon]
                
                windows.append({
                    "entity_id": entity_id,
                    "history_start": history["ts_utc"].iloc[0],
                    "history_end": history["ts_utc"].iloc[-1],
                    "forecast_start": future["ts_utc"].iloc[0],
                    "forecast_end": future["ts_utc"].iloc[-1],
                    "history_energy": history["energy_kwh"].values,
                    "history_timestamps": history["ts_utc"].values,
                    "target_energy": future["energy_kwh"].values,
                    "target_timestamps": future["ts_utc"].values,
                    "interval_mins": history["interval_mins"].iloc[0],
                })
        
        logger.info(f"Created {len(windows)} forecasting windows")
        
        return windows
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """Get basic dataset statistics."""
        return {
            "n_records": len(df),
            "n_entities": df["entity_id"].nunique(),
            "date_range": {
                "start": df["ts_utc"].min().isoformat(),
                "end": df["ts_utc"].max().isoformat(),
            },
            "energy_stats": {
                "mean": df["energy_kwh"].mean(),
                "std": df["energy_kwh"].std(),
                "min": df["energy_kwh"].min(),
                "max": df["energy_kwh"].max(),
                "q25": df["energy_kwh"].quantile(0.25),
                "q50": df["energy_kwh"].quantile(0.50),
                "q75": df["energy_kwh"].quantile(0.75),
            },
            "entities": df["entity_id"].unique().tolist()[:10],  # First 10
        }
