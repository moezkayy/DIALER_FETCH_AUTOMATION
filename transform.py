#!/usr/bin/env python3
"""
Transform Module - Data Cleaning and Transformation Only
=========================================================
Responsible ONLY for cleaning and transforming raw data.
No fetching. No configuration values. No business logic.
Just take raw data and make it clean and standardized.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import Config
from path_manager import PathManager


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )


# ============================================================================
# BASIC CLEANING UTILITIES
# ============================================================================

class DataCleaner:
    """Basic data cleaning utilities."""

    @staticmethod
    def clean_phone_number(phone: Any) -> str:
        """
        Clean and normalize phone number.

        Args:
            phone: Phone number (any type)

        Returns:
            Cleaned phone number string
        """
        if pd.isna(phone):
            return ""

        phone_str = str(phone)
        phone_str = phone_str.strip()
        phone_str = phone_str.replace("\u00A0", "")  # Non-breaking spaces
        phone_str = phone_str.replace(" ", "")
        phone_str = phone_str.replace("-", "")
        phone_str = phone_str.replace("+", "")
        phone_str = phone_str.split('.')[0]  # Remove .0 from floats

        return phone_str

    @staticmethod
    def normalize_string(s: Any) -> str:
        """
        Normalize string (strip and uppercase).

        Args:
            s: String to normalize

        Returns:
            Normalized string
        """
        if pd.isna(s):
            return ""
        return str(s).strip().upper()

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names (strip whitespace and quotes).

        Args:
            df: DataFrame to clean

        Returns:
            DataFrame with cleaned column names
        """
        df.columns = df.columns.str.strip('"').str.strip()
        return df

    @staticmethod
    def convert_to_int_safe(value: Any, default: int = 0) -> int:
        """
        Safely convert value to integer.

        Args:
            value: Value to convert
            default: Default value if conversion fails

        Returns:
            Integer value
        """
        try:
            return int(pd.to_numeric(value, errors='coerce'))
        except:
            return default

    @staticmethod
    def convert_to_datetime_safe(value: Any) -> Optional[pd.Timestamp]:
        """
        Safely convert value to datetime.

        Args:
            value: Value to convert

        Returns:
            Datetime or None
        """
        try:
            return pd.to_datetime(value, errors='coerce')
        except:
            return None


# ============================================================================
# WORKDIALERS TRANSFORMER
# ============================================================================

class WorkdialersTransformer:
    """Transform raw workdialers data."""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def transform(self) -> Optional[pd.DataFrame]:
        """
        Transform all raw workdialers files.

        Returns:
            Combined and cleaned DataFrame, or None if no files
        """
        raw_files = self.path_manager.list_raw_workdialer_files()

        if not raw_files:
            self.logger.warning("No raw workdialers files found")
            return None

        self.logger.info(f"Transforming {len(raw_files)} workdialers file(s)")

        all_dfs = []

        for file_path in raw_files:
            try:
                # Extract dialer number from filename
                match = re.search(r'dialer(\d+)', file_path.name)
                dialer_number = match.group(1) if match else "unknown"

                # Read file
                df = pd.read_csv(file_path, sep="\t", engine="python")
                df = df.dropna(axis=1, how="all")

                # Add dialer identifier
                df['dialer'] = dialer_number

                # Clean and standardize
                df = self._clean_dataframe(df)

                all_dfs.append(df)
                self.logger.info(f"   ✅ Processed: {file_path.name} ({len(df)} rows)")

            except Exception as e:
                self.logger.error(f"   ❌ Failed to process {file_path.name}: {e}")
                continue

        if not all_dfs:
            self.logger.error("No workdialers files processed successfully")
            return None

        # Combine all dataframes
        combined = pd.concat(all_dfs, ignore_index=True)

        # Save transformed data
        output_file = self.path_manager.get_transformed_workdialers_file()
        combined.to_csv(output_file, index=False)

        self.logger.info(f"   💾 Saved: {output_file.name} ({len(combined)} total rows)")

        return combined

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize a single workdialers dataframe."""
        # Clean column names
        df = DataCleaner.clean_column_names(df)

        # Clean phone numbers if column exists
        if 'phone_number' in df.columns:
            df['clean_phone_number'] = df['phone_number'].apply(DataCleaner.clean_phone_number)

        # Normalize status if column exists
        if 'status' in df.columns:
            df['status'] = df['status'].apply(DataCleaner.normalize_string)

        # Normalize user if column exists
        if 'user' in df.columns:
            df['user'] = df['user'].astype(str).str.strip()

        # Convert length_in_sec to int if column exists
        if 'length_in_sec' in df.columns:
            df['length_in_sec'] = df['length_in_sec'].apply(
                lambda x: DataCleaner.convert_to_int_safe(x, 0)
            )

        # Convert call_date to datetime if column exists
        if 'call_date' in df.columns:
            df['call_date'] = df['call_date'].apply(DataCleaner.convert_to_datetime_safe)

        return df


# ============================================================================
# CL1 TRANSFORMER
# ============================================================================

class CL1Transformer:
    """Transform raw CL1 data."""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def transform(self) -> Optional[pd.DataFrame]:
        """
        Transform raw CL1 file.

        Returns:
            Cleaned DataFrame, or None if file not found
        """
        raw_file = self.path_manager.get_raw_cl1_file()

        if not raw_file.exists():
            self.logger.warning("Raw CL1 file not found")
            return None

        self.logger.info("Transforming CL1 data")

        try:
            df = pd.read_csv(raw_file)
            df = self._clean_dataframe(df)

            # Save transformed data
            output_file = self.path_manager.get_transformed_cl1_file()
            df.to_csv(output_file, index=False)

            self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")

            return df

        except Exception as e:
            self.logger.error(f"   ❌ Failed to transform CL1: {e}")
            return None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize CL1 dataframe."""
        # Clean column names
        df = DataCleaner.clean_column_names(df)

        # Clean phone numbers
        if 'phone_number' in df.columns:
            df['clean_phone_number'] = df['phone_number'].apply(DataCleaner.clean_phone_number)

        # Normalize status
        if 'status' in df.columns:
            df['status'] = df['status'].apply(DataCleaner.normalize_string)

        # Convert length_in_sec to int
        if 'length_in_sec' in df.columns:
            df['length_in_sec'] = df['length_in_sec'].apply(
                lambda x: DataCleaner.convert_to_int_safe(x, 0)
            )

        # Convert call_date to datetime
        if 'call_date' in df.columns:
            df['call_date'] = df['call_date'].apply(DataCleaner.convert_to_datetime_safe)

        return df


# ============================================================================
# JT TRANSFORMER
# ============================================================================

class JTTransformer:
    """Transform raw JT data."""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def transform(self) -> Optional[pd.DataFrame]:
        """
        Transform raw JT file.

        Returns:
            Cleaned DataFrame, or None if file not found
        """
        raw_file = self.path_manager.get_raw_jt_file()

        if not raw_file.exists():
            self.logger.warning("Raw JT file not found")
            return None

        self.logger.info("Transforming JT data")

        try:
            df = pd.read_csv(raw_file)
            df = self._clean_dataframe(df)

            # Save transformed data
            output_file = self.path_manager.get_transformed_jt_file()
            df.to_csv(output_file, index=False)

            self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")

            return df

        except Exception as e:
            self.logger.error(f"   ❌ Failed to transform JT: {e}")
            return None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize JT dataframe."""
        # Clean column names
        df = DataCleaner.clean_column_names(df)

        # Clean phone numbers
        if 'phone_number' in df.columns:
            df['clean_phone_number'] = df['phone_number'].apply(DataCleaner.clean_phone_number)

        # Normalize status
        if 'status' in df.columns:
            df['status'] = df['status'].apply(DataCleaner.normalize_string)

        # Convert length_in_sec to int
        if 'length_in_sec' in df.columns:
            df['length_in_sec'] = df['length_in_sec'].apply(
                lambda x: DataCleaner.convert_to_int_safe(x, 0)
            )

        # Convert call_date to datetime
        if 'call_date' in df.columns:
            df['call_date'] = df['call_date'].apply(DataCleaner.convert_to_datetime_safe)

        return df


# ============================================================================
# THREEWAY TRANSFORMER
# ============================================================================

class ThreeWayTransformer:
    """Transform raw ThreeWay data."""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def transform(self) -> Optional[pd.DataFrame]:
        """
        Transform parsed ThreeWay file.

        Returns:
            Cleaned DataFrame, or None if file not found
        """
        parsed_file = self.path_manager.get_parsed_threeway_file()

        if not parsed_file.exists():
            self.logger.warning("Parsed ThreeWay file not found")
            return None

        self.logger.info("Transforming ThreeWay data")

        try:
            df = pd.read_csv(parsed_file)
            df = self._clean_dataframe(df)

            # Save transformed data
            output_file = self.path_manager.get_transformed_threeway_file()
            df.to_csv(output_file, index=False)

            self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")

            return df

        except Exception as e:
            self.logger.error(f"   ❌ Failed to transform ThreeWay: {e}")
            return None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize ThreeWay dataframe."""
        if df.empty:
            return df

        # Basic cleanup
        if 'PHONE' in df.columns:
            df = df[df['PHONE'] != 'Restricted']
            df = df.dropna(subset=['PHONE'])
            df['PHONE'] = pd.to_numeric(df['PHONE'], errors='coerce').fillna(0).astype(int)
            df['clean_phone_number'] = df['PHONE'].apply(DataCleaner.clean_phone_number)

        # Convert Length to int
        if 'Length' in df.columns:
            df['Length'] = pd.to_numeric(df['Length'], errors='coerce').fillna(0).astype(int)

        # Convert Transfer Time to datetime
        if 'Transfer Time' in df.columns:
            df['Transfer Time'] = df['Transfer Time'].apply(DataCleaner.convert_to_datetime_safe)

        # Normalize Preset Name
        if 'Preset Name' in df.columns:
            df['Preset Name'] = df['Preset Name'].apply(DataCleaner.normalize_string)

        # Sort by length descending
        if 'Length' in df.columns:
            df = df.sort_values('Length', ascending=False).reset_index(drop=True)

        return df


# ============================================================================
# MAIN TRANSFORM ORCHESTRATOR
# ============================================================================

class Transformer:
    """Main orchestrator for all data transformation."""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def transform_all(self) -> dict[str, Optional[pd.DataFrame]]:
        """
        Transform all raw data sources.

        Returns:
            Dictionary of source name to transformed DataFrame
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING DATA TRANSFORMATION")
        self.logger.info("=" * 70)

        results = {}

        # Transform workdialers
        transformer = WorkdialersTransformer(self.path_manager)
        results['workdialers'] = transformer.transform()

        # Transform CL1
        transformer = CL1Transformer(self.path_manager)
        results['cl1'] = transformer.transform()

        # Transform JT
        transformer = JTTransformer(self.path_manager)
        results['jt'] = transformer.transform()

        # Transform ThreeWay
        transformer = ThreeWayTransformer(self.path_manager)
        results['threeway'] = transformer.transform()

        self.logger.info("=" * 70)
        self.logger.info("TRANSFORMATION COMPLETE")
        self.logger.info("=" * 70)

        # Print summary
        for source, df in results.items():
            if df is not None:
                self.logger.info(f"  ✓ {source.upper()}: {len(df):,} rows")
            else:
                self.logger.info(f"  ✗ {source.upper()}: No data")

        return results

    def combine_all(self, results: dict[str, Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
        """
        Combine all transformed data into single DataFrame.

        Args:
            results: Dictionary of transformed DataFrames

        Returns:
            Combined DataFrame, or None if no data
        """
        self.logger.info("Combining all transformed data")

        valid_dfs = [df for df in results.values() if df is not None and not df.empty]

        if not valid_dfs:
            self.logger.warning("No data to combine")
            return None

        # For now, just save each source separately
        # You can implement your own combining logic here

        self.logger.info(f"   ✅ {len(valid_dfs)} source(s) available for combining")

        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    from config import load_config

    config = load_config()
    path_manager = PathManager(config.base_output_dir, config.start_date)

    setup_logging(path_manager.get_transform_log(), config.log_level)

    transformer = Transformer(path_manager)
    results = transformer.transform_all()
    transformer.combine_all(results)