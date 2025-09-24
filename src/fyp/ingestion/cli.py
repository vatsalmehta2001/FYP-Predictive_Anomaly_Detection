"""Unified CLI for data ingestion."""

import argparse
import sys
from pathlib import Path

from .lcl_ingestor import LCLIngestor
from .ukdale_ingestor import UKDALEIngestor
from .ssen_ingestor import SSENIngestor


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Energy dataset ingestion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest LCL data using samples
  python -m fyp.ingestion.cli lcl --use-samples
  
  # Ingest UK-DALE with downsampling
  python -m fyp.ingestion.cli ukdale --downsample-30min
  
  # Ingest SSEN from API (requires network)
  python -m fyp.ingestion.cli ssen
  
  # Dry run to see what would happen
  python -m fyp.ingestion.cli lcl --dry-run
""",
    )
    
    # Common arguments
    parser.add_argument(
        "dataset",
        choices=["lcl", "ukdale", "ssen"],
        help="Dataset to ingest",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw"),
        help="Input root directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Output root directory (default: data/processed)",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use sample data for testing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output",
    )
    
    # Dataset-specific arguments
    parser.add_argument(
        "--downsample-30min",
        action="store_true",
        default=True,
        help="Create 30-minute downsampled version (UK-DALE only)",
    )
    parser.add_argument(
        "--ckan-url",
        default="https://data.ssen.co.uk",
        help="CKAN API URL (SSEN only)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (SSEN only)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of cached API responses (SSEN only)",
    )
    
    args = parser.parse_args()
    
    # Create appropriate ingestor
    if args.dataset == "lcl":
        ingestor = LCLIngestor(
            input_root=args.input_root,
            output_root=args.output_root,
            use_samples=args.use_samples,
            dry_run=args.dry_run,
        )
    elif args.dataset == "ukdale":
        ingestor = UKDALEIngestor(
            input_root=args.input_root,
            output_root=args.output_root,
            use_samples=args.use_samples,
            downsample_30min=args.downsample_30min,
            dry_run=args.dry_run,
        )
    elif args.dataset == "ssen":
        ingestor = SSENIngestor(
            input_root=args.input_root,
            output_root=args.output_root,
            use_samples=args.use_samples,
            ckan_url=args.ckan_url,
            api_key=args.api_key,
            force_refresh=args.force_refresh,
            dry_run=args.dry_run,
        )
    else:
        parser.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)
    
    try:
        ingestor.run()
    except Exception as e:
        print(f"Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
