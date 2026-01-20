#!/usr/bin/env python3
"""
Fetch Module - Data Retrieval Only
===================================
Responsible ONLY for fetching raw data from sources.
No transformations. No validations. No business logic.
Just get the data and save it to disk.
made by moez khan
"""

import time
import logging
import re
import io
import glob
import asyncio
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo
from datetime import datetime
from random import randint

import yaml
import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
from playwright.sync_api import sync_playwright, Page, Download
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

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


def time_to_seconds(duration):
    """Convert time string to seconds"""
    try:
        parts = duration.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        else:
            return int(duration)
    except:
        return 0


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

        # Check if Playwright should be skipped for this dialer
        skip_playwright = work_number in self.config.workdialers.skip_playwright_for_dialers

        # Alternating retry pattern
        max_attempts = max(
            self.config.workdialers.max_playwright_attempts,
            self.config.workdialers.max_requests_attempts
        )

        for attempt in range(1, max_attempts + 1):
            # Try Playwright on odd attempts (1, 3, 5...)
            if attempt % 2 == 1 and not skip_playwright:
                playwright_attempt_num = (attempt + 1) // 2
                if playwright_attempt_num <= self.config.workdialers.max_playwright_attempts:
                    self.logger.info(
                        f"   🎭 Playwright attempt {playwright_attempt_num}/{self.config.workdialers.max_playwright_attempts}")
                    result = self._try_playwright_single(work_number, base_url, username, password)

                    if result == 'success':
                        return  # Data downloaded successfully
                    elif result == 'no_data':
                        self.logger.info(f"   ⏭️  No data for this dialer, moving to next")
                        return  # No data exists, stop trying
                    # If result == 'error', continue to next attempt

            # Try requests on even attempts (2, 4, 6...)
            elif attempt % 2 == 0:
                requests_attempt_num = attempt // 2
                if requests_attempt_num <= self.config.workdialers.max_requests_attempts:
                    self.logger.info(
                        f"   📤 Requests attempt {requests_attempt_num}/{self.config.workdialers.max_requests_attempts}")
                    result = self._try_requests_single(work_number, base_url, username, password)

                    if result == 'success':
                        return  # Data downloaded successfully
                    elif result == 'no_data':
                        self.logger.info(f"   ⏭️  No data for this dialer, moving to next")
                        return  # No data exists, stop trying
                    # If result == 'error', continue to next attempt

            # Wait before next attempt (except on last attempt)
            if attempt < max_attempts:
                time.sleep(self.config.workdialers.retry_wait_seconds)

        self.logger.warning(f"   ❌ All attempts failed for {name}")

    def _try_playwright_single(self, work_number: str, base_url: str,
                               username: str, password: str) -> str:
        """
        Single Playwright attempt.

        Returns:
            'success' - Data downloaded successfully
            'no_data' - Confirmed no data exists for this date
            'error' - Attempt failed, should retry
        """
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
                        return 'success'
                    else:
                        temp_path.unlink()
                        if reason == 'no_data':
                            self.logger.info(f"   ℹ️  No data available")
                            page.close()
                            context.close()
                            browser.close()
                            return 'no_data'  # Confirmed no data

                page.close()
                context.close()
                browser.close()

        except Exception as e:
            self.logger.warning(f"   ⚠️  Playwright error: {e}")

        return 'error'  # Failed, should retry

    def _try_requests_single(self, work_number: str, base_url: str,
                             username: str, password: str) -> str:
        """
        Single requests attempt.

        Returns:
            'success' - Data downloaded successfully
            'no_data' - Confirmed no data exists for this date
            'error' - Attempt failed, should retry
        """
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
                return 'success'
            elif reason == 'no_data':
                self.logger.info(f"   ℹ️  No data available")
                return 'no_data'  # Confirmed no data
            else:
                self.logger.warning(f"   ⚠️  Invalid content: {reason}")
                return 'error'  # Invalid response, should retry

        except Exception as e:
            self.logger.warning(f"   ⚠️  Requests error: {e}")
            return 'error'  # Failed, should retry

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
# CL2 FETCHER
# ============================================================================

class CL2Fetcher:
    """Fetches call reports from CL2 dialer."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch(self):
        """Fetch CL2 data."""
        if not self.config.cl2.enabled:
            self.logger.info("CL2 fetching disabled")
            return

        self.logger.info("⬇️  Fetching CL2 data")

        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"   📤 Attempt {attempt}/{max_attempts}")

                # Extract hostname from URL
                url_base = self.config.cl2.url.replace("https://", "").replace("http://", "")

                export_link = (
                    f"https://{self.config.cl2.username}:{self.config.cl2.password}@"
                    f"{self.config.cl2.url.lstrip('https://').lstrip('http://')}"
                    f"/call_report_export.php?DB=&run_export=1"
                    f"&ivr_export=&query_date={self.config.start_date}"
                    f"&end_date={self.config.start_date}"
                    f"&date_field=call_date&header_row=YES&rec_fields=NONE"
                    f"&call_notes=NO&export_fields=STANDARD"
                    f"&campaign%5B%5D=---ALL---&group%5B%5D=---NONE---"
                    f"&list_id%5B%5D=---ALL---&status%5B%5D=---ALL---"
                    f"&user_group%5B%5D=---ALL---&SUBMIT=SUBMIT"
                )

                response = requests.get(
                    export_link,
                    timeout=self.config.request_timeout_seconds
                )
                response.raise_for_status()

                # Parse and clean the data
                df = pd.read_csv(io.StringIO(response.text), delimiter="\t")
                df = df[df["user"] != "VDAD"]
                df["user"] = df["user"].str.split("_").str[0]
                df = df.sort_values(by="length_in_sec", ascending=False)

                # Save the cleaned data
                output_file = self.path_manager.get_raw_cl2_file()
                df.to_csv(output_file, index=False)

                self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")
                return

            except Exception as e:
                self.logger.warning(f"   ⚠️  Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    time.sleep(5)
                else:
                    self.logger.error(f"   ❌ Failed to fetch CL2 after {max_attempts} attempts")


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
# OMNI FETCHER
# ============================================================================

class OmniFetcher:
    """Fetches call reports from Omni using async Playwright."""

    def __init__(self, config: Config, path_manager: PathManager):
        self.config = config
        self.path_manager = path_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch(self):
        """Fetch Omni data (synchronous wrapper)."""
        if not self.config.omni.enabled:
            self.logger.info("Omni fetching disabled")
            return

        self.logger.info("⬇️  Fetching Omni data")

        # Run async fetch in event loop
        try:
            asyncio.run(self._fetch_async())
        except Exception as e:
            self.logger.error(f"   ❌ Failed to fetch Omni: {e}")

    async def _fetch_async(self):
        """Async method to fetch Omni data."""
        max_attempts = self.config.omni.max_attempts

        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"   🎭 Attempt {attempt}/{max_attempts}")

                # Setup download directory
                download_dir = self.path_manager.get_omni_download_dir()
                download_dir.mkdir(parents=True, exist_ok=True)

                # Download the file
                result = await self._download_agent_performance(
                    str(download_dir),
                    self._convert_date_format(self.config.start_date)
                )

                if result == 'File Downloaded':
                    # Clean and process the data
                    fetch_pattern = str(download_dir / "*.xls")
                    df = self._clean_omni_data(fetch_pattern)

                    if df is not None and not df.empty:
                        output_file = self.path_manager.get_raw_omni_file()
                        df.to_csv(output_file, index=False)
                        self.logger.info(f"   ✅ Saved: {output_file.name} ({len(df)} rows)")
                        return
                    else:
                        self.logger.warning(f"   ⚠️  No data extracted from download")

            except Exception as e:
                self.logger.warning(f"   ⚠️  Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(5)
                else:
                    self.logger.error(f"   ❌ Failed after {max_attempts} attempts")

    def _convert_date_format(self, date_str: str) -> str:
        """Convert YYYY-MM-DD to MM/DD/YYYY format."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y")
        except:
            return date_str

    async def _download_agent_performance(self, d_path: str, date_to_find: str):
        """Download agent performance data using async Playwright."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.config.headless_browser,
                downloads_path=d_path
            )

            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()

            try:
                # Navigate to login page
                await page.goto(self.config.omni.url)
                await asyncio.sleep(randint(3, 5))

                # Login
                await page.fill('input#Usuario', self.config.omni.username)
                await page.fill('input#Contrasenya', self.config.omni.password)
                await asyncio.sleep(1)

                # Wait for navigation after login
                async with page.expect_navigation():
                    await page.click('button#conectar')
                await asyncio.sleep(randint(4, 6))

                # Wait for page to be fully loaded
                await page.wait_for_load_state('networkidle')

                # Navigate to Reports page
                base_url = self.config.omni.url.replace('/Manager/Home/Login', '')
                await page.goto(f'{base_url}/Manager/Reports/ReportList/100000001')
                await page.wait_for_load_state('networkidle')
                await asyncio.sleep(randint(2, 3))

                # Click on "Transactions list" report
                await page.click('tr#Report100000077 a')
                await asyncio.sleep(1)

                # Click agent name button
                await page.click('input[name="NomAgent"]')
                await asyncio.sleep(1)

                # Click first input in titles table
                await page.click('table#tblTitles input')
                await asyncio.sleep(1)

                # Click first input in fields table
                await page.click('table#tblFieldsToSelect input')
                await asyncio.sleep(1)

                # Click Accept button
                await page.wait_for_selector('span.ui-button-text:has-text("Accept")', state='visible')
                await page.locator('span.ui-button-text', has_text="Accept").click()
                await asyncio.sleep(1)

                # Fill date fields
                await page.fill('input[name="fDesde"]', '')
                await page.fill('input[name="fDesde"]', date_to_find)
                await page.fill('input[name="fHasta"]', '')
                await page.fill('input[name="fHasta"]', date_to_find)
                await asyncio.sleep(1)

                # Scroll to download button and click
                download_btn = page.locator('a#btnAbrirExcel')
                await download_btn.scroll_into_view_if_needed()
                await asyncio.sleep(1)

                # Wait for download
                async with page.expect_download() as download_info:
                    await download_btn.click()
                download = await download_info.value

                # Save download
                import os
                await download.save_as(os.path.join(d_path, download.suggested_filename))
                await asyncio.sleep(5)

                self.logger.info(f"   📥 Downloaded: {download.suggested_filename}")
                return 'File Downloaded'

            finally:
                await browser.close()

    def _clean_omni_data(self, fetch_path: str) -> Optional[pd.DataFrame]:
        """Clean and process downloaded Omni data."""
        try:
            files = glob.glob(fetch_path)
            if not files:
                self.logger.warning(f"   ⚠️  No files found matching: {fetch_path}")
                return None

            max_file = max(files, key=lambda f: Path(f).stat().st_ctime)
            self.logger.info(f"   🧹 Processing: {Path(max_file).name}")

            with open(max_file, "rb") as my_file:
                bs_obj = BeautifulSoup(my_file, 'lxml')
                table = bs_obj.find('table').find_all('tr')

                records = []
                for tr in table[1:]:
                    td = tr.find_all('td')
                    if len(td) < 17:
                        continue

                    id_ = td[0].text.strip()
                    camp = td[1].text.strip()
                    date_time = td[2].text.strip()
                    dis = td[4].text.strip()
                    phone = td[12].text.strip()
                    closer = td[14].text.strip()
                    audio = td[16].text.strip()
                    duration = td[9].text.strip()
                    source = td[12].text.strip()
                    did = td[13].text.strip()

                    if source in ['nan', '', 'None']:
                        source = 'avoid'

                    a_duration = time_to_seconds(duration)

                    records.append({
                        'id': id_,
                        'camp': camp,
                        'date/time': date_time,
                        'status': dis,
                        'phone': phone,
                        'closer': closer,
                        'audio': audio,
                        'duration': a_duration,
                        'source': source,
                        'did': did
                    })

                return pd.DataFrame(records)

        except Exception as e:
            self.logger.error(f"   ❌ Failed to clean Omni data: {e}")
            return None


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

        # Fetch CL2
        if self.config.cl2.enabled:
            fetcher = CL2Fetcher(self.config, self.path_manager)
            fetcher.fetch()

        # Fetch JT
        if self.config.jt.enabled:
            fetcher = JTFetcher(self.config, self.path_manager)
            fetcher.fetch()

        # Fetch ThreeWay
        if self.config.threeway.enabled:
            fetcher = ThreeWayFetcher(self.config, self.path_manager)
            fetcher.fetch()

        # Fetch Omni
        if self.config.omni.enabled:
            fetcher = OmniFetcher(self.config, self.path_manager)
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
    cl2_fetcher = CL2Fetcher(config, path_manager)
    cl2_fetcher.fetch()
    jt_fetcher = JTFetcher(config, path_manager)
    jt_fetcher.fetch()