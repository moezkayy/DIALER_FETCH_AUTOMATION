#!/usr/bin/env python3
"""
Fetch Module - Data Retrieval Only
===================================
Responsible ONLY for fetching raw data from sources.
No transformations. No validations. No business logic.
Just get the data and save it to disk.
"""

import time
import logging
import re
import io
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo
from datetime import datetime

import yaml
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from playwright.sync_api import sync_playwright, Page, Download
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from config_loader import Config, load_config
from path_manager import PathManager


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Configure logging with proper Unicode support."""
    import sys

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create console handler with UTF-8 encoding for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Try to set UTF-8 encoding on Windows
    try:
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

    handlers = [console_handler]

    if log_file:
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )

    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def should_search_archived_data(target_date_str: str) -> bool:
    """
    Determine if archived data should be searched based on target date.

    Args:
        target_date_str: Target date in YYYY-MM-DD format

    Returns:
        True if target date is not today
    """
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        eastern_now = datetime.now(ZoneInfo("America/New_York"))
        eastern_today = eastern_now.date()
        return target_date != eastern_today
    except ValueError:
        return True


def check_no_data_response(content: bytes) -> bool:
    """
    Check if response indicates no data available.

    Args:
        content: Downloaded content bytes

    Returns:
        True if this is a "no data" response
    """
    try:
        text = content.decode('utf-8', errors='ignore').strip()

        no_data_indicators = [
            "There are no inbound calls during this time period",
            "There are no records",
            "No records found",
            "no calls during this time period"
        ]

        text_lower = text.lower()
        for indicator in no_data_indicators:
            if indicator.lower() in text_lower:
                return True

        # Check if it's empty HTML
        if '<body>' in text_lower and '</body>' in text_lower:
            body_match = re.search(r'<body[^>]*>(.*?)</body>', text, re.DOTALL | re.IGNORECASE)
            if body_match:
                body_content = body_match.group(1).strip()
                body_text = re.sub(r'<[^>]+>', '', body_content).strip()
                if len(body_text) < 100:
                    return True

        return False
    except Exception:
        return False


def validate_content(content: bytes, min_lines: int = 2) -> Tuple[bool, str]:
    """
    Validate that downloaded content is valid data.

    Args:
        content: Downloaded content bytes
        min_lines: Minimum number of lines expected

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        text = content.decode('utf-8', errors='ignore')

        if check_no_data_response(content):
            return False, 'no_data'

        if '<html' in text.lower() or '<!doctype' in text.lower():
            return False, 'html_response'

        lines = text.strip().split('\n')
        if len(lines) < min_lines:
            return False, 'insufficient_lines'

        return True, 'valid'
    except Exception:
        return False, 'validation_error'


# ============================================================================
# WORKDIALERS FETCHER
# ============================================================================

class WorkdialersFetcher:
    """Fetches call reports from workdialers using Playwright and requests fallback."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dialers = self._load_dialers()

    def _load_dialers(self) -> dict:
        """Load dialer configurations from YAML file."""
        config_file = Path(self.config.workdialers_config_file)

        if not config_file.exists():
            self.logger.error(f"Workdialers config file not found: {config_file}")
            return {}

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse workdialers YAML: {e}")
            return {}

    def fetch_all(self):
        """Fetch call reports from all enabled workdialers."""
        active_dialers = {k: v for k, v in self.dialers.items()
                          if isinstance(v, dict) and v.get("do_run", False)}

        if not active_dialers:
            self.logger.warning("No dialers enabled in workdialers config")
            return

        self.logger.info(f"Processing {len(active_dialers)} active dialer(s)")

        for key, dialer_config in active_dialers.items():
            self._fetch_dialer(key, dialer_config)

    def _fetch_dialer(self, dialer_key: str, dialer_config: dict):
        """Fetch call report for a single dialer."""
        name = dialer_config.get("name", dialer_key)
        work_number = dialer_config["work_number"]
        username = dialer_config["username"].strip()
        password = dialer_config["password"].strip()
        server_code = dialer_config["server_code"].strip()

        base_url = f"https://work{work_number}.dialerhosting.com/{server_code}"

        self.logger.info(f"⬇️  Fetching: {name} (work{work_number})")

        # Try Playwright first (unless skipped)
        skip_playwright = work_number in self.config.workdialers.skip_playwright_for_dialers

        if not skip_playwright:
            success = self._try_playwright(work_number, base_url, username, password)
            if success:
                return
        else:
            self.logger.info(f"   ⏭️  Skipping Playwright for dialer {work_number}")

        # Fallback to requests
        self._try_requests(work_number, base_url, username, password)

    def _try_playwright(self, work_number: str, base_url: str,
                        username: str, password: str) -> bool:
        """Try fetching using Playwright."""
        max_attempts = self.config.workdialers.max_playwright_attempts

        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"   🎭 Playwright attempt {attempt}/{max_attempts}")

            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self.config.headless_browser)
                    context = browser.new_context(
                        http_credentials={"username": username, "password": password},
                        accept_downloads=True
                    )
                    page = context.new_page()

                    url = f"{base_url}/call_report_export.php"
                    page.goto(url, timeout=self.config.playwright_timeout_ms)
                    time.sleep(2)

                    # Fill form and download
                    download = self._fill_and_submit_form(page)

                    if download:
                        temp_path = self.path_manager.get_raw_workdialer_call_report(
                            str(work_number), "_temp"
                        )
                        download.save_as(temp_path)

                        content = temp_path.read_bytes()
                        is_valid, reason = validate_content(content, self.config.min_content_lines)

                        if is_valid:
                            final_path = self.path_manager.get_raw_workdialer_call_report(
                                str(work_number)
                            )
                            if final_path.exists():
                                final_path.unlink()
                            temp_path.rename(final_path)

                            size_kb = final_path.stat().st_size / 1024
                            self.logger.info(f"   ✅ Saved: {final_path.name} ({size_kb:.1f} KB)")

                            page.close()
                            context.close()
                            browser.close()
                            return True
                        else:
                            temp_path.unlink()
                            if reason == 'no_data':
                                self.logger.info(f"   ℹ️  No data available")
                                page.close()
                                context.close()
                                browser.close()
                                return False

                    page.close()
                    context.close()
                    browser.close()

            except Exception as e:
                self.logger.warning(f"   ⚠️  Playwright error: {e}")

            if attempt < max_attempts:
                time.sleep(self.config.workdialers.retry_wait_seconds)

        return False

    def _fill_and_submit_form(self, page: Page) -> Optional[Download]:
        """Fill form and submit to get download."""
        try:
            # Fill dates
            page.fill('input[name="query_date"]', self.config.start_date)
            page.fill('input[name="end_date"]', self.config.end_date)

            # Select options
            page.select_option('select[name="date_field"]', 'call_date')
            page.select_option('select[name="header_row"]', 'YES')
            page.select_option('select[name="export_fields"]', 'STANDARD')

            # Check archived data if needed
            if should_search_archived_data(self.config.start_date):
                try:
                    archive_checkbox = page.locator('input[name="search_archived_data"]')
                    if archive_checkbox.count() > 0:
                        archive_checkbox.check()
                        self.logger.info(f"   📦 Checked 'search_archived_data'")
                except Exception:
                    pass

            # Select all campaigns and groups
            page.locator('select[name="campaign[]"]').select_option("---ALL---")

            # Select all user groups
            user_group_select = page.locator('select[name="user_group[]"]')
            options = user_group_select.locator("option")
            option_values = []
            for i in range(options.count()):
                value = options.nth(i).get_attribute("value")
                if value and value != "---ALL---":
                    option_values.append(value)

            if option_values:
                user_group_select.select_option(option_values)
                self.logger.info(f"   👥 Selected {len(option_values)} user groups")

            # Select all for other fields
            for field in ["group[]", "list_id[]", "status[]"]:
                page.locator(f'select[name="{field}"]').select_option("---ALL---")

            # Submit and wait for download
            with page.expect_download(timeout=self.config.download_timeout_ms) as download_info:
                page.click('input[type="submit"][name="SUBMIT"]')

            return download_info.value

        except PlaywrightTimeoutError:
            # Check if page has no data message
            try:
                page_content = page.content()
                if check_no_data_response(page_content.encode('utf-8')):
                    self.logger.info(f"   ℹ️  No data message detected")
            except:
                pass
            return None
        except Exception as e:
            self.logger.error(f"   ❌ Form submission failed: {e}")
            return None

    def _try_requests(self, work_number: str, base_url: str,
                      username: str, password: str):
        """Try fetching using requests library."""
        max_attempts = self.config.workdialers.max_requests_attempts
        url = f"{base_url}/call_report_export.php"

        params = {
            'DB': '',
            'run_export': '1',
            'query_date': self.config.start_date,
            'end_date': self.config.end_date,
            'date_field': 'call_date',
            'header_row': 'YES',
            'export_fields': 'STANDARD',
            'group[]': '---ALL---',
            'list_id[]': '---ALL---',
            'status[]': '---ALL---',
            'user_group[]': '---ALL---',
            'campaign[]': '---ALL---',
            'SUBMIT': 'SUBMIT'
        }

        if should_search_archived_data(self.config.start_date):
            params['search_archived_data'] = 'checked'
            self.logger.info(f"   📦 Including 'search_archived_data' parameter")

        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"   📤 Requests attempt {attempt}/{max_attempts}")

            try:
                response = requests.get(
                    url,
                    params=params,
                    auth=HTTPBasicAuth(username, password),
                    timeout=self.config.request_timeout_seconds
                )
                response.raise_for_status()

                content = response.content
                is_valid, reason = validate_content(content, self.config.min_content_lines)

                if is_valid:
                    filepath = self.path_manager.get_raw_workdialer_call_report(str(work_number))
                    filepath.write_bytes(content)
                    size_kb = len(content) / 1024
                    self.logger.info(f"   ✅ Saved: {filepath.name} ({size_kb:.1f} KB)")
                    return
                elif reason == 'no_data':
                    self.logger.info(f"   ℹ️  No data available")
                    return
                else:
                    self.logger.warning(f"   ⚠️  Invalid content: {reason}")

            except Exception as e:
                self.logger.warning(f"   ⚠️  Requests error: {e}")

            if attempt < max_attempts:
                time.sleep(self.config.workdialers.retry_wait_seconds)


# ============================================================================
# CL1 FETCHER
# ============================================================================

class CL1Fetcher:
    """Fetches call reports from CL1 dialer."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch(self):
        """Fetch CL1 data."""
        if not self.config.cl1.enabled:
            self.logger.info("CL1 fetching disabled")
            return

        self.logger.info("⬇️  Fetching CL1 data")

        export_link = (
            f'{self.config.cl1.url}/call_report_export.php?'
            f'DB=&run_export=1&ivr_export=&query_date={self.config.start_date}'
            f'&end_date={self.config.end_date}'
            f'&date_field=call_date&header_row=YES&rec_fields=NONE&custom_fields=NO&call_notes=NO'
            f'&export_fields=STANDARD&campaign%5B%5D=---ALL---&group%5B%5D=---ALL---'
            f'&list_id%5B%5D=---ALL---&status%5B%5D=---ALL---&user_group%5B%5D=---ALL---&SUBMIT=SUBMIT'
        )

        try:
            response = requests.get(
                export_link,
                auth=(self.config.cl1.username, self.config.cl1.password),
                timeout=self.config.request_timeout_seconds
            )
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text), sep='\t')

            output_file = self.path_manager.get_raw_cl1_file()
            df.to_csv(output_file, index=False)

            self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")

        except Exception as e:
            self.logger.error(f"   ❌ Failed to fetch CL1: {e}")


# ============================================================================
# JT FETCHER
# ============================================================================

class JTFetcher:
    """Fetches call reports from JT dialer."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch(self):
        """Fetch JT data."""
        if not self.config.jt.enabled:
            self.logger.info("JT fetching disabled")
            return

        self.logger.info("⬇️  Fetching JT data")

        # Extract hostname from URL
        url_base = self.config.jt.url.replace("https://", "").replace("http://", "")

        export_link = (
            f'https://{self.config.jt.username}:{self.config.jt.password}@'
            f'{url_base}/call_report_export.php?'
            f'DB=&run_export=1&ivr_export=&query_date={self.config.start_date}'
            f'&end_date={self.config.end_date}'
            f'&date_field=call_date&header_row=YES&rec_fields=NONE&call_notes=NO'
            f'&export_fields=STANDARD&campaign%5B%5D=---ALL---&group%5B%5D=---ALL---'
            f'&list_id%5B%5D=---ALL---&status%5B%5D=---ALL---&user_group%5B%5D=---ALL---&SUBMIT=SUBMIT'
        )

        try:
            response = requests.get(export_link, timeout=self.config.request_timeout_seconds)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text), sep='\t')

            output_file = self.path_manager.get_raw_jt_file()
            df.to_csv(output_file, index=False)

            self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")

        except Exception as e:
            self.logger.error(f"   ❌ Failed to fetch JT: {e}")


# ============================================================================
# THREEWAY FETCHER
# ============================================================================

class ThreeWayFetcher:
    """Fetches ThreeWay call reports."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch(self):
        """Fetch ThreeWay data."""
        if not self.config.threeway.enabled:
            self.logger.info("ThreeWay fetching disabled")
            return

        self.logger.info("⬇️  Fetching ThreeWay data")

        url = (
            f"{self.config.threeway.url}"
            f"?begin_query_time={self.config.start_date}+00:00:00"
            f"&end_query_time={self.config.start_date}+23:59:00"
        )

        try:
            response = requests.get(
                url,
                auth=HTTPBasicAuth(self.config.threeway.username, self.config.threeway.password),
                verify=False,
                timeout=self.config.request_timeout_seconds
            )
            response.raise_for_status()

            # Parse HTML tables
            tables = pd.read_html(io.StringIO(response.text), flavor="bs4")

            df = None
            for t in tables:
                if all(col in t.columns for col in ["Lead ID", "Phone Number"]):
                    df = t
                    break

            if df is None or df.empty:
                self.logger.info("   ℹ️  No ThreeWay data found")
                return

            # Save raw data
            output_file = self.path_manager.get_raw_threeway_file()
            df.to_csv(output_file, index=False)

            self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")

            # Parse structure
            parsed_df = self._parse_structure(df)
            if not parsed_df.empty:
                parsed_file = self.path_manager.get_parsed_threeway_file()
                parsed_df.to_csv(parsed_file, index=False)
                self.logger.info(f"   ✅ Parsed: {parsed_file.name} ({len(parsed_df)} rows)")

        except Exception as e:
            self.logger.error(f"   ❌ Failed to fetch ThreeWay: {e}")

    def _parse_structure(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Parse the VICIdial 3-way structure to extract call lengths."""
        if df_raw.empty:
            return df_raw

        records = []
        current_lead = None

        for idx, row in df_raw.iterrows():
            if str(row.get('#')).strip() not in ["", "nan"]:
                if current_lead is not None and current_lead.get('call_lengths'):
                    max_length = max(current_lead['call_lengths'])
                    current_lead['Length'] = max_length
                    records.append(current_lead)

                current_lead = {
                    'Lead ID': row.get('Lead ID'),
                    'Campaign': row.get('Campaign'),
                    'Agent': row.get('Agent'),
                    'PHONE': row.get('Phone Number'),
                    'Preset Name': row.get('Preset Phone Number'),
                    'Transfer Time': row.get('Transfer Time'),
                    'call_lengths': []
                }

            elif current_lead is not None:
                row_str = row.astype(str)

                if row_str.str.contains("Call Length:", case=False).any():
                    for i in range(len(row_str) - 1):
                        if "Call Length:" in row_str.iloc[i]:
                            length_val = row_str.iloc[i + 1]
                            try:
                                current_lead["call_lengths"].append(int(length_val))
                            except:
                                pass

        if current_lead is not None and current_lead.get('call_lengths'):
            max_length = max(current_lead['call_lengths'])
            current_lead['Length'] = max_length
            records.append(current_lead)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.drop('call_lengths', axis=1, errors='ignore')

        return df


# ============================================================================
# MAIN FETCH ORCHESTRATOR
# ============================================================================

class Fetcher:
    """Main orchestrator for all data fetching."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_all(self):
        """Fetch data from all enabled sources."""
        self.logger.info("=" * 70)
        self.logger.info("STARTING DATA FETCH")
        self.logger.info("=" * 70)

        enabled = self.config.get_enabled_sources()
        self.logger.info(f"Enabled sources: {', '.join(enabled)}")

        # Fetch workdialers
        if self.config.workdialers.enabled:
            fetcher = WorkdialersFetcher(self.config, self.path_manager)
            fetcher.fetch_all()

        # Fetch CL1
        if self.config.cl1.enabled:
            fetcher = CL1Fetcher(self.config, self.path_manager)
            fetcher.fetch()

        # Fetch JT
        if self.config.jt.enabled:
            fetcher = JTFetcher(self.config, self.path_manager)
            fetcher.fetch()

        # Fetch ThreeWay
        if self.config.threeway.enabled:
            fetcher = ThreeWayFetcher(self.config, self.path_manager)
            fetcher.fetch()

        self.logger.info("=" * 70)
        self.logger.info("FETCH COMPLETE")
        self.logger.info("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = load_config()
    path_manager = PathManager(config.base_output_dir, config.start_date)

    setup_logging(path_manager.get_fetch_log(), config.log_level)

    fetcher = Fetcher(config, path_manager)
    # fetcher.fetch_all()
    cl1_fetch = CL1Fetcher(config, path_manager)
    cl1_fetch.fetch()