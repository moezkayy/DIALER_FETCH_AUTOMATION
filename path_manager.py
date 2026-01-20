#!/usr/bin/env python3
"""
Path Manager - Centralized Path Control
========================================
Responsible only for constructing and managing file paths.
No business logic. No data processing. Just paths.
made by moez khan

"""

from pathlib import Path
from typing import Optional


class PathManager:
    """
    Manages all file paths for the pipeline.

    Directory Structure:
    {base_output_dir}/{date}/
    ├── raw/
    │   ├── workdialers/
    │   ├── cl1/
    │   ├── cl2/
    │   ├── jt/
    │   ├── threeway/
    │   └── omni/
    ├── transformed/
    └── logs/
    """

    def __init__(self, base_output_dir: str, date: str):
        """
        Initialize path manager.

        Args:
            base_output_dir: Base directory for all pipeline runs
            date: Date string (YYYY-MM-DD)
        """
        self.base_dir = Path(base_output_dir)
        self.date = date
        self.date_dir = self.base_dir / date

        # Create directory structure
        self._create_structure()

    def _create_structure(self):
        """Create the directory structure."""
        directories = [
            self.get_raw_dir(),
            self.get_raw_workdialers_dir(),
            self.get_raw_cl1_dir(),
            self.get_raw_cl2_dir(),
            self.get_raw_jt_dir(),
            self.get_raw_threeway_dir(),
            self.get_raw_omni_dir(),
            self.get_transformed_dir(),
            self.get_logs_dir(),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # ROOT DIRECTORIES
    # ========================================================================

    def get_date_dir(self) -> Path:
        """Get date-specific base directory."""
        return self.date_dir

    def get_raw_dir(self) -> Path:
        """Get raw data directory."""
        return self.date_dir / "raw"

    def get_transformed_dir(self) -> Path:
        """Get transformed data directory."""
        return self.date_dir / "transformed"

    def get_logs_dir(self) -> Path:
        """Get logs directory."""
        return self.date_dir / "logs"

    # ========================================================================
    # RAW DATA PATHS
    # ========================================================================

    def get_raw_workdialers_dir(self) -> Path:
        """Get raw workdialers directory."""
        return self.get_raw_dir() / "workdialers"

    def get_raw_cl1_dir(self) -> Path:
        """Get raw CL1 directory."""
        return self.get_raw_dir() / "cl1"

    def get_raw_cl2_dir(self) -> Path:
        """Get raw CL2 directory."""
        return self.get_raw_dir() / "cl2"

    def get_raw_jt_dir(self) -> Path:
        """Get raw JT directory."""
        return self.get_raw_dir() / "jt"

    def get_raw_threeway_dir(self) -> Path:
        """Get raw ThreeWay directory."""
        return self.get_raw_dir() / "threeway"

    def get_raw_omni_dir(self) -> Path:
        """Get raw Omni directory."""
        return self.get_raw_dir() / "omni"

    # ========================================================================
    # SPECIFIC FILE PATHS - RAW
    # ========================================================================

    def get_raw_workdialer_call_report(self, dialer_number: str, suffix: str = "") -> Path:
        """
        Get path for workdialer call report.

        Args:
            dialer_number: Dialer number (e.g., "1", "2", "7")
            suffix: Optional suffix (e.g., "_MERGED", "_campaign_X")

        Returns:
            Path to call report file
        """
        if suffix:
            filename = f"dialer{dialer_number}_call_report{suffix}.csv"
        else:
            filename = f"dialer{dialer_number}_call_report.csv"

        return self.get_raw_workdialers_dir() / filename

    def get_raw_cl1_file(self) -> Path:
        """Get path for raw CL1 file."""
        return self.get_raw_cl1_dir() / f"cl1_{self.date}.csv"

    def get_raw_cl2_file(self) -> Path:
        """Get path for raw CL2 file."""
        return self.get_raw_cl2_dir() / f"cl2_{self.date}.csv"

    def get_raw_jt_file(self) -> Path:
        """Get path for raw JT file."""
        return self.get_raw_jt_dir() / f"jt_{self.date}.csv"

    def get_raw_threeway_file(self) -> Path:
        """Get path for raw ThreeWay file."""
        return self.get_raw_threeway_dir() / f"threeway_raw_{self.date}.csv"

    def get_parsed_threeway_file(self) -> Path:
        """Get path for parsed ThreeWay file."""
        return self.get_raw_threeway_dir() / f"threeway_parsed_{self.date}.csv"

    def get_raw_omni_file(self) -> Path:
        """Get path for raw Omni file."""
        return self.get_raw_omni_dir() / f"omni_{self.date}.csv"

    def get_omni_download_dir(self) -> Path:
        """Get Omni download directory."""
        return self.get_raw_omni_dir() / "downloads"

    # ========================================================================
    # SPECIFIC FILE PATHS - TRANSFORMED
    # ========================================================================

    def get_transformed_workdialers_file(self) -> Path:
        """Get path for transformed workdialers data."""
        return self.get_transformed_dir() / "workdialers_transformed.csv"

    def get_transformed_cl1_file(self) -> Path:
        """Get path for transformed CL1 data."""
        return self.get_transformed_dir() / "cl1_transformed.csv"

    def get_transformed_cl2_file(self) -> Path:
        """Get path for transformed CL2 data."""
        return self.get_transformed_dir() / "cl2_transformed.csv"

    def get_transformed_jt_file(self) -> Path:
        """Get path for transformed JT data."""
        return self.get_transformed_dir() / "jt_transformed.csv"

    def get_transformed_threeway_file(self) -> Path:
        """Get path for transformed ThreeWay data."""
        return self.get_transformed_dir() / "threeway_transformed.csv"

    def get_transformed_omni_file(self) -> Path:
        """Get path for transformed Omni data."""
        return self.get_transformed_dir() / "omni_transformed.csv"

    def get_combined_transformed_file(self) -> Path:
        """Get path for combined transformed data."""
        return self.get_transformed_dir() / "all_sources_combined.csv"

    # ========================================================================
    # LOG PATHS
    # ========================================================================

    def get_fetch_log(self) -> Path:
        """Get path for fetch log."""
        return self.get_logs_dir() / "fetch.log"

    def get_transform_log(self) -> Path:
        """Get path for transform log."""
        return self.get_logs_dir() / "transform.log"

    def get_main_log(self) -> Path:
        """Get path for main pipeline log."""
        return self.get_logs_dir() / "pipeline.log"

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def list_raw_workdialer_files(self) -> list[Path]:
        """List all raw workdialer call report files."""
        workdialers_dir = self.get_raw_workdialers_dir()
        return sorted(workdialers_dir.glob("dialer*_call_report*.csv"))

    def get_structure_summary(self) -> str:
        """Get a summary of the directory structure."""
        lines = [
            "=" * 70,
            f"PATH STRUCTURE FOR {self.date}",
            "=" * 70,
            f"Base: {self.base_dir}",
            f"Date: {self.date_dir}",
            "",
            "Raw Data:",
            f"  - Workdialers: {self.get_raw_workdialers_dir().relative_to(self.base_dir)}",
            f"  - CL1:         {self.get_raw_cl1_dir().relative_to(self.base_dir)}",
            f"  - CL2:         {self.get_raw_cl2_dir().relative_to(self.base_dir)}",
            f"  - JT:          {self.get_raw_jt_dir().relative_to(self.base_dir)}",
            f"  - ThreeWay:    {self.get_raw_threeway_dir().relative_to(self.base_dir)}",
            f"  - Omni:        {self.get_raw_omni_dir().relative_to(self.base_dir)}",
            "",
            "Transformed Data:",
            f"  - Directory:   {self.get_transformed_dir().relative_to(self.base_dir)}",
            "",
            "Logs:",
            f"  - Directory:   {self.get_logs_dir().relative_to(self.base_dir)}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def print_structure(self):
        """Print the directory structure summary."""
        print(self.get_structure_summary())


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    pm = PathManager(base_output_dir="pipeline_runs", date="2025-01-09")
    pm.print_structure()

    # Example: Get specific paths
    print("\nExample paths:")
    print(f"Raw workdialer 1: {pm.get_raw_workdialer_call_report('1')}")
    print(f"Raw CL1:          {pm.get_raw_cl1_file()}")
    print(f"Raw CL2:          {pm.get_raw_cl2_file()}")
    print(f"Raw Omni:         {pm.get_raw_omni_file()}")
    print(f"Transformed:      {pm.get_transformed_workdialers_file()}")
    print(f"Fetch log:        {pm.get_fetch_log()}")