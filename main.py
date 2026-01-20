#!/usr/bin/env python3
"""
Main Pipeline - Complete Data Pipeline Orchestrator
====================================================
Orchestrates the complete data pipeline:
1. Load configuration
2. Setup paths and logging
3. Fetch raw data from all enabled sources
4. Transform and clean the data
5. Generate summary reports

Usage:
    python main.py                    # Run with default config.yaml
    python main.py --config custom.yaml  # Run with custom config
    python main.py --date 2026-01-10  # Override date
    python main.py --help             # Show help

made by moez khan
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from config_loader import Config, load_config
from path_manager import PathManager
from fetch import Fetcher, setup_logging as setup_fetch_logging
from transform import Transformer, setup_logging as setup_transform_logging


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_main_logging(log_file: Path, level: str = "INFO"):
    """Configure main pipeline logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Try to set UTF-8 encoding on Windows
    try:
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[console_handler, file_handler],
        force=True
    )

    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


# ============================================================================
# PIPELINE STAGES
# ============================================================================

class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_time = None
        self.end_time = None

    def run(self, skip_fetch: bool = False, skip_transform: bool = False):
        """
        Run the complete pipeline.

        Args:
            skip_fetch: Skip the fetch stage (use existing raw data)
            skip_transform: Skip the transform stage (use existing transformed data)
        """
        self.start_time = datetime.now()

        self._print_header()
        self._print_config_summary()

        try:
            # Stage 1: Fetch
            if not skip_fetch:
                self._run_fetch_stage()
            else:
                self.logger.info("⏭️  Skipping fetch stage (using existing raw data)")

            # Stage 2: Transform
            if not skip_transform:
                self._run_transform_stage()
            else:
                self.logger.info("⏭️  Skipping transform stage (using existing transformed data)")

            # Stage 3: Summary
            self._print_summary()

            self.end_time = datetime.now()
            self._print_footer()

        except KeyboardInterrupt:
            self.logger.warning("\n⚠️  Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

    def _run_fetch_stage(self):
        """Run the fetch stage."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("STAGE 1: DATA FETCH")
        self.logger.info("=" * 70)

        # Setup fetch logging
        setup_fetch_logging(self.path_manager.get_fetch_log(), self.config.log_level)

        # Run fetcher
        fetcher = Fetcher(self.config, self.path_manager)
        fetcher.fetch_all()

        self.logger.info("✅ Fetch stage complete")

    def _run_transform_stage(self):
        """Run the transform stage."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("STAGE 2: DATA TRANSFORMATION")
        self.logger.info("=" * 70)

        # Setup transform logging
        setup_transform_logging(self.path_manager.get_transform_log(), self.config.log_level)

        # Run transformer
        transformer = Transformer(self.path_manager)
        results = transformer.transform_all()
        transformer.combine_all(results)

        self.logger.info("✅ Transform stage complete")

    def _print_header(self):
        """Print pipeline header."""
        eastern_now = datetime.now(ZoneInfo("US/Eastern"))

        print("\n" + "=" * 70)
        print("DATA PIPELINE")
        print("=" * 70)
        print(f"Started: {eastern_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Pipeline Run: {self.path_manager.date}")
        print("=" * 70)

    def _print_config_summary(self):
        """Print configuration summary."""
        print(self.config.summary())

    def _print_summary(self):
        """Print pipeline execution summary."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 70)

        # Check what was fetched
        self.logger.info("\nRaw Data Files:")
        raw_dir = self.path_manager.get_raw_dir()
        if raw_dir.exists():
            for source_dir in sorted(raw_dir.iterdir()):
                if source_dir.is_dir():
                    files = list(source_dir.glob("*.csv")) + list(source_dir.glob("*.xls"))
                    if files:
                        self.logger.info(f"  {source_dir.name.upper()}:")
                        for file in sorted(files):
                            size_kb = file.stat().st_size / 1024
                            self.logger.info(f"    - {file.name} ({size_kb:.1f} KB)")

        # Check what was transformed
        self.logger.info("\nTransformed Data Files:")
        transformed_dir = self.path_manager.get_transformed_dir()
        if transformed_dir.exists():
            files = list(transformed_dir.glob("*.csv"))
            if files:
                for file in sorted(files):
                    size_kb = file.stat().st_size / 1024

                    # Try to count rows
                    try:
                        import pandas as pd
                        df = pd.read_csv(file)
                        row_count = len(df)
                        self.logger.info(f"  - {file.name} ({size_kb:.1f} KB, {row_count:,} rows)")
                    except:
                        self.logger.info(f"  - {file.name} ({size_kb:.1f} KB)")
            else:
                self.logger.info("  No transformed files found")

        # Log locations
        self.logger.info("\nOutput Locations:")
        self.logger.info(f"  Base Directory: {self.path_manager.get_date_dir()}")
        self.logger.info(f"  Raw Data: {self.path_manager.get_raw_dir()}")
        self.logger.info(f"  Transformed: {self.path_manager.get_transformed_dir()}")
        self.logger.info(f"  Logs: {self.path_manager.get_logs_dir()}")

    def _print_footer(self):
        """Print pipeline footer."""
        eastern_now = datetime.now(ZoneInfo("US/Eastern"))

        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "unknown"

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Finished: {eastern_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Duration: {duration_str}")
        print(f"Status: ✅ SUCCESS")
        print("=" * 70 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Pipeline - Fetch and Transform Call Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default config
  python main.py --config custom.yaml      # Use custom config file
  python main.py --date 2026-01-10         # Override date
  python main.py --skip-fetch              # Skip fetch, only transform
  python main.py --skip-transform          # Only fetch, skip transform
  python main.py --fetch-only              # Same as --skip-transform
  python main.py --transform-only          # Same as --skip-fetch
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Override date for pipeline run (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetch stage (use existing raw data)'
    )

    parser.add_argument(
        '--skip-transform',
        action='store_true',
        help='Skip transform stage (only fetch)'
    )

    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Only run fetch stage (alias for --skip-transform)'
    )

    parser.add_argument(
        '--transform-only',
        action='store_true',
        help='Only run transform stage (alias for --skip-fetch)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Override log level from config'
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found: {args.config}")
        print(f"   Create a config.yaml file or specify a different config with --config")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Override date if provided
    if args.date:
        try:
            # Validate date format
            datetime.strptime(args.date, "%Y-%m-%d")
            config.start_date = args.date
            config.end_date = args.date
        except ValueError:
            print(f"❌ Error: Invalid date format: {args.date}")
            print(f"   Use YYYY-MM-DD format (e.g., 2026-01-14)")
            sys.exit(1)

    # Override log level if provided
    if args.log_level:
        config.log_level = args.log_level

    # Setup path manager
    path_manager = PathManager(config.base_output_dir, config.start_date)

    # Setup main logging
    setup_main_logging(path_manager.get_main_log(), config.log_level)

    # Determine which stages to skip
    skip_fetch = args.skip_fetch or args.transform_only
    skip_transform = args.skip_transform or args.fetch_only

    # Validate that we're not skipping everything
    if skip_fetch and skip_transform:
        print("❌ Error: Cannot skip both fetch and transform stages")
        print("   Remove conflicting flags or run without any skip flags")
        sys.exit(1)

    # Run pipeline
    pipeline = Pipeline(config, path_manager)
    pipeline.run(skip_fetch=skip_fetch, skip_transform=skip_transform)


if __name__ == "__main__":
    main()