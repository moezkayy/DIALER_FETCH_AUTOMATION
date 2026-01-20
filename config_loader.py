#!/usr/bin/env python3
"""
Configuration Loader
====================
Loads configuration from YAML files.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    enabled: bool
    username: Optional[str] = None
    password: Optional[str] = None
    url: Optional[str] = None


@dataclass
class WorkdialersConfig:
    """Configuration for workdialers fetching."""
    enabled: bool
    max_playwright_attempts: int = 3
    max_requests_attempts: int = 3
    retry_wait_seconds: int = 10
    skip_playwright_for_dialers: list = None

    def __post_init__(self):
        if self.skip_playwright_for_dialers is None:
            self.skip_playwright_for_dialers = []


@dataclass
class OmniConfig:
    """Configuration for Omni fetching."""
    enabled: bool
    username: Optional[str] = None
    password: Optional[str] = None
    url: Optional[str] = None
    download_path: Optional[str] = None
    max_attempts: int = 6


@dataclass
class Config:
    """Master configuration for the entire pipeline."""

    # Date range
    start_date: str
    end_date: str

    # Paths
    base_output_dir: str
    workdialers_config_file: str

    # Sources
    workdialers: WorkdialersConfig
    cl1: SourceConfig
    cl2: SourceConfig
    jt: SourceConfig
    threeway: SourceConfig
    omni: OmniConfig

    # Behavior
    headless_browser: bool
    playwright_timeout_ms: int
    download_timeout_ms: int
    request_timeout_seconds: int
    min_content_lines: int
    log_level: str

    def get_enabled_sources(self) -> list:
        """Get list of enabled source names."""
        sources = []
        if self.workdialers.enabled:
            sources.append("workdialers")
        if self.cl1.enabled:
            sources.append("cl1")
        if self.cl2.enabled:
            sources.append("cl2")
        if self.jt.enabled:
            sources.append("jt")
        if self.threeway.enabled:
            sources.append("threeway")
        if self.omni.enabled:
            sources.append("omni")
        return sources

    def summary(self) -> str:
        """Get configuration summary."""
        lines = [
            "=" * 70,
            "PIPELINE CONFIGURATION",
            "=" * 70,
            f"Date Range: {self.start_date} to {self.end_date}",
            f"Output Directory: {self.base_output_dir}",
            "",
            "Enabled Sources:",
        ]

        for source in self.get_enabled_sources():
            lines.append(f"  ✓ {source.upper()}")

        lines.extend([
            "",
            "Behavior:",
            f"  - Headless Browser: {self.headless_browser}",
            f"  - Playwright Attempts: {self.workdialers.max_playwright_attempts}",
            f"  - Request Attempts: {self.workdialers.max_requests_attempts}",
            f"  - Log Level: {self.log_level}",
            "=" * 70,
        ])

        return "\n".join(lines)


def load_config(config_file: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Config instance

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If required configuration missing
    """
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    # Extract configuration sections
    dates = yaml_config.get('dates', {})
    paths = yaml_config.get('paths', {})
    workdialers_cfg = yaml_config.get('workdialers', {})
    cl1_cfg = yaml_config.get('cl1', {})
    cl2_cfg = yaml_config.get('cl2', {})
    jt_cfg = yaml_config.get('jt', {})
    threeway_cfg = yaml_config.get('threeway', {})
    omni_cfg = yaml_config.get('omni', {})
    browser_cfg = yaml_config.get('browser', {})
    requests_cfg = yaml_config.get('requests', {})
    validation_cfg = yaml_config.get('validation', {})
    logging_cfg = yaml_config.get('logging', {})

    eastern_today = datetime.now(ZoneInfo("US/Eastern")).date().isoformat()

    start_date = dates.get('start_date') or eastern_today
    end_date = dates.get('end_date') or eastern_today

    # Build configuration
    config = Config(
        # Dates
        start_date=start_date,
        end_date=end_date,

        # Paths
        base_output_dir=paths.get('base_output_dir', 'pipeline_runs'),
        workdialers_config_file=paths.get('workdialers_config', 'workdialers.yaml'),

        # Workdialers
        workdialers=WorkdialersConfig(
            enabled=workdialers_cfg.get('enabled', True),
            max_playwright_attempts=workdialers_cfg.get('max_playwright_attempts', 3),
            max_requests_attempts=workdialers_cfg.get('max_requests_attempts', 3),
            retry_wait_seconds=workdialers_cfg.get('retry_wait_seconds', 10),
            skip_playwright_for_dialers=workdialers_cfg.get('skip_playwright_for_dialers', [])
        ),

        # CL1
        cl1=SourceConfig(
            enabled=cl1_cfg.get('enabled', False),
            username=cl1_cfg.get('username'),
            password=cl1_cfg.get('password'),
            url=cl1_cfg.get('url')
        ),

        # CL2
        cl2=SourceConfig(
            enabled=cl2_cfg.get('enabled', False),
            username=cl2_cfg.get('username'),
            password=cl2_cfg.get('password'),
            url=cl2_cfg.get('url')
        ),

        # JT
        jt=SourceConfig(
            enabled=jt_cfg.get('enabled', False),
            username=jt_cfg.get('username'),
            password=jt_cfg.get('password'),
            url=jt_cfg.get('url')
        ),

        # ThreeWay
        threeway=SourceConfig(
            enabled=threeway_cfg.get('enabled', False),
            username=threeway_cfg.get('username'),
            password=threeway_cfg.get('password'),
            url=threeway_cfg.get('url')
        ),

        # Omni
        omni=OmniConfig(
            enabled=omni_cfg.get('enabled', False),
            username=omni_cfg.get('username'),
            password=omni_cfg.get('password'),
            url=omni_cfg.get('url'),
            download_path=omni_cfg.get('download_path'),
            max_attempts=omni_cfg.get('max_attempts', 6)
        ),

        # Browser
        headless_browser=browser_cfg.get('headless', True),
        playwright_timeout_ms=browser_cfg.get('playwright_timeout_ms', 20000),
        download_timeout_ms=browser_cfg.get('download_timeout_ms', 30000),

        # Requests
        request_timeout_seconds=requests_cfg.get('timeout_seconds', 60),

        # Validation
        min_content_lines=validation_cfg.get('min_content_lines', 2),

        # Logging
        log_level=logging_cfg.get('level', 'INFO')
    )

    return config


if __name__ == "__main__":
    config = load_config()
    print(config.summary())