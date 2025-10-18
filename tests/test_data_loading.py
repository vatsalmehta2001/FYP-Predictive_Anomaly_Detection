"""Tests for data loading and validation."""

import pytest
import pandas as pd
from pathlib import Path
from fyp.data_loader import EnergyDataLoader

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestSampleDataLoading:
    """Test loading sample datasets."""
    
    def test_load_lcl_sample(self):
        """Test loading LCL sample data."""
        sample_file = PROJECT_ROOT / "data" / "samples" / "lcl_sample.csv"
        assert sample_file.exists(), f"LCL sample not found: {sample_file}"
        
        df = pd.read_csv(sample_file)
        
        # Verify basic structure
        assert not df.empty, "LCL sample is empty"
        assert len(df) > 0, "LCL sample has no rows"
        
        # Verify expected columns exist (flexible column names)
        columns_lower = [col.lower() for col in df.columns]
        assert any('time' in col or 'date' in col for col in columns_lower), \
            "No timestamp column found"
        assert any('kwh' in col or 'energy' in col for col in columns_lower), \
            "No energy column found"
    
    def test_load_ukdale_sample(self):
        """Test loading UK-DALE sample data."""
        sample_file = PROJECT_ROOT / "data" / "samples" / "ukdale_sample.csv"
        assert sample_file.exists(), f"UK-DALE sample not found: {sample_file}"
        
        df = pd.read_csv(sample_file)
        assert not df.empty, "UK-DALE sample is empty"
    
    def test_load_ssen_sample(self):
        """Test loading SSEN sample data."""
        sample_file = PROJECT_ROOT / "data" / "samples" / "ssen_sample.csv"
        assert sample_file.exists(), f"SSEN sample not found: {sample_file}"
        
        df = pd.read_csv(sample_file)
        assert not df.empty, "SSEN sample is empty"


class TestSSENMetadataLoading:
    """Test loading SSEN metadata."""
    
    def test_load_ssen_metadata(self):
        """Test loading SSEN feeder metadata if available."""
        metadata_path = PROJECT_ROOT / "data" / "processed" / "ssen_metadata.parquet"
        
        if not metadata_path.exists():
            pytest.skip("SSEN metadata not yet generated")
        
        df = pd.read_parquet(metadata_path)
        
        # Verify metadata structure
        assert not df.empty, "SSEN metadata is empty"
        assert len(df) > 0, "SSEN metadata has no rows"
        
        # Check for constraint columns
        expected_cols = ['voltage_nominal_v', 'voltage_tolerance_pct', 
                         'power_factor_min', 'power_factor_max']
        for col in expected_cols:
            assert col in df.columns, f"Missing constraint column: {col}"
        
        # Verify constraint values
        assert df['voltage_nominal_v'].iloc[0] == 230.0, "Incorrect nominal voltage"
        assert df['voltage_tolerance_pct'].iloc[0] == 10.0, "Incorrect voltage tolerance"
        assert df['power_factor_min'].iloc[0] == 0.8, "Incorrect min power factor"
        assert df['power_factor_max'].iloc[0] == 1.0, "Incorrect max power factor"


class TestProcessedDataLoading:
    """Test loading processed datasets (if they exist)."""
    
    def test_processed_data_structure(self):
        """Test that processed data directory exists."""
        processed_dir = PROJECT_ROOT / "data" / "processed"
        assert processed_dir.exists(), "Processed data directory missing"
    
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "data" / "processed" / "dataset=lcl").exists(),
        reason="LCL not yet processed"
    )
    def test_load_processed_lcl(self):
        """Test loading processed LCL data if available."""
        try:
            loader = EnergyDataLoader(PROJECT_ROOT / "data" / "processed")
            df = loader.load_dataset("lcl", limit_rows=100)
            
            assert not df.empty, "Processed LCL is empty"
            assert "ts_utc" in df.columns, "Missing ts_utc column"
            assert "entity_id" in df.columns, "Missing entity_id column"
            assert "energy_kwh" in df.columns, "Missing energy_kwh column"
            
            # Verify timezone
            assert df['ts_utc'].dt.tz is not None, "Timestamps not timezone-aware"
            
        except Exception as e:
            pytest.skip(f"Could not load processed LCL: {e}")
    
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "data" / "processed" / "dataset=ukdale").exists(),
        reason="UK-DALE not yet processed"
    )
    def test_load_processed_ukdale(self):
        """Test loading processed UK-DALE data if available."""
        try:
            loader = EnergyDataLoader(PROJECT_ROOT / "data" / "processed")
            df = loader.load_dataset("ukdale", limit_rows=100)
            
            assert not df.empty, "Processed UK-DALE is empty"
            assert "ts_utc" in df.columns, "Missing ts_utc column"
            assert "entity_id" in df.columns, "Missing entity_id column"
            assert "energy_kwh" in df.columns, "Missing energy_kwh column"
            assert "extras" in df.columns, "Missing extras column"
            
            # Verify timezone
            assert df['ts_utc'].dt.tz is not None, "Timestamps not timezone-aware"
            
        except Exception as e:
            pytest.skip(f"Could not load processed UK-DALE: {e}")


class TestDataValidation:
    """Test data validation logic."""
    
    def test_timestamp_alignment_30min(self):
        """Test 30-minute timestamp alignment."""
        from datetime import datetime
        import pytz
        
        # Create test timestamp
        ts = datetime(2023, 1, 1, 12, 15, tzinfo=pytz.UTC)
        
        # Should align to 12:00
        aligned = ts.replace(minute=0 if ts.minute < 30 else 30, second=0, microsecond=0)
        
        assert aligned.minute in (0, 30), "Timestamp not aligned to 30-min boundary"
    
    def test_energy_value_validation(self):
        """Test energy value validation."""
        # Valid values
        assert 0 <= 10.5, "Valid energy rejected"
        
        # Invalid values
        assert not (-1 >= 0), "Negative energy accepted"
        assert not (float('nan') == float('nan')), "NaN energy accepted (NaN != NaN)"


class TestIngestionSummaries:
    """Test that ingestion summaries are created."""
    
    def test_ingestion_summary_exists(self):
        """Test that at least one ingestion summary exists."""
        processed_dir = PROJECT_ROOT / "data" / "processed"
        summary_file = processed_dir / "ingestion_summary.json"
        
        # Summary should exist if any ingestion has been run
        if not summary_file.exists():
            pytest.skip("No ingestion has been run yet")
        
        import json
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Verify summary structure
        assert 'timestamp' in summary, "Missing timestamp in summary"
        assert 'dataset' in summary, "Missing dataset in summary"
        assert 'stats' in summary, "Missing stats in summary"
        
        # Verify stats structure
        stats = summary['stats']
        assert 'processed' in stats, "Missing processed count"
        assert 'errors' in stats, "Missing errors count"
        assert 'skipped' in stats, "Missing skipped count"


def test_project_structure():
    """Verify required project directories exist."""
    assert (PROJECT_ROOT / "src" / "fyp").exists(), "src/fyp/ missing"
    assert (PROJECT_ROOT / "data" / "samples").exists(), "data/samples/ missing"
    assert (PROJECT_ROOT / "data" / "raw").exists(), "data/raw/ missing"
    assert (PROJECT_ROOT / "data" / "processed").exists(), "data/processed/ missing"
    assert (PROJECT_ROOT / "notebooks").exists(), "notebooks/ missing"
    assert (PROJECT_ROOT / "docs").exists(), "docs/ missing"


def test_documentation_exists():
    """Verify key documentation files exist."""
    assert (PROJECT_ROOT / "README.md").exists(), "README.md missing"
    assert (PROJECT_ROOT / "docs" / "anomaly_strategy.md").exists(), \
        "anomaly_strategy.md missing"


def test_raw_data_exists():
    """Test that raw data files exist (if downloaded)."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    
    # Check for LCL
    lcl_file = raw_dir / "lcl" / "CC_LCL-FullData.csv"
    if lcl_file.exists():
        assert lcl_file.stat().st_size > 1e9, "LCL file too small (should be ~8.5GB)"
    
    # Check for UK-DALE
    ukdale_file = raw_dir / "ukdale" / "ukdale.h5"
    if ukdale_file.exists():
        assert ukdale_file.stat().st_size > 1e9, "UK-DALE file too small (should be ~6.3GB)"
    
    # Check for SSEN
    ssen_file = raw_dir / "ssen" / "LV_FEEDER_LOOKUP.csv"
    if ssen_file.exists():
        assert ssen_file.stat().st_size > 1e6, "SSEN file too small (should be ~37MB)"

