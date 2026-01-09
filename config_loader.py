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
    jt: SourceConfig
    threeway: SourceConfig

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
        if self.jt.enabled:
            sources.append("jt")
        if self.threeway.enabled:
            sources.append("threeway")
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
    jt_cfg = yaml_config.get('jt', {})
    threeway_cfg = yaml_config.get('threeway', {})
    browser_cfg = yaml_config.get('browser', {})
    requests_cfg = yaml_config.get('requests', {})
    validation_cfg = yaml_config.get('validation', {})
    logging_cfg = yaml_config.get('logging', {})

    # Build configuration
    config = Config(
        # Dates
        start_date=dates.get('start_date', '2025-01-09'),
        end_date=dates.get('end_date', '2025-01-09'),

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