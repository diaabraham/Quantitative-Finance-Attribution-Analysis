"""Market and SEC data access for QuantFAA."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv

from data.providers import (
    PROVIDER_NAMES,
    PriceProvider,
    ProviderError,
    get_provider,
    list_available_providers,
    resolve_provider,
)
from data.providers.yfinance_provider import _extract_close


# Backward-compatible alias used by older tests/imports
def _adjusted_close_series(df: pd.DataFrame, symbol: str) -> pd.Series:
    return _extract_close(df, symbol)


class DataManager:
    def __init__(
        self,
        data_dir: str = "data/raw",
        cache_dir: str = "data/cache",
        provider: str | PriceProvider | None = "auto",
        require_sec: bool = False,
    ):
        """
        Initialize the Data Manager.

        Parameters
        ----------
        data_dir, cache_dir :
            Local storage paths.
        provider :
            Market-data provider name (``yfinance``, ``stooq``, ``massive``,
            ``polygon``, ``databento``), a ``PriceProvider`` instance, or
            ``auto`` to pick the first available provider.
        require_sec :
            When True, raise if ``SEC_API_KEY`` is missing. Price fetching
            does not require SEC credentials.
        """
        load_dotenv()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.ensure_directories()

        self.provider = resolve_provider(provider, fallback=True)
        self.provider_name = getattr(self.provider, "name", str(provider))

        self.sec_api_key = os.getenv("SEC_API_KEY")
        if self.sec_api_key and self.sec_api_key.strip() in {
            "your_api_key_here",
            "YOUR_API_KEY",
            "changeme",
        }:
            self.sec_api_key = None

        self.extractor_api = None
        self.query_api = None
        if self.sec_api_key:
            from sec_api import ExtractorApi, QueryApi

            self.extractor_api = ExtractorApi(self.sec_api_key)
            self.query_api = QueryApi(self.sec_api_key)
        elif require_sec:
            raise ValueError("SEC_API_KEY not found in environment variables")

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "sec"), exist_ok=True)

    @staticmethod
    def available_providers() -> list[str]:
        return [p.name for p in list_available_providers()]

    @staticmethod
    def known_providers() -> tuple[str, ...]:
        return PROVIDER_NAMES

    def set_provider(self, provider: str | PriceProvider, *, fallback: bool = False):
        """Switch the active market-data provider."""
        self.provider = resolve_provider(provider, fallback=fallback)
        self.provider_name = self.provider.name
        return self.provider

    def get_sector_etfs(self, enrich_sec: bool = False):
        """
        Get sector ETF information.

        SEC enrichment is optional and skipped when no SEC API key is configured.
        """
        sector_etfs = {
            "Technology": {"symbol": "XLK", "cik": "0001067983"},
            "Financials": {"symbol": "XLF", "cik": "0001067983"},
            "Healthcare": {"symbol": "XLV", "cik": "0001067983"},
            "Consumer Discretionary": {"symbol": "XLY", "cik": "0001067983"},
            "Consumer Staples": {"symbol": "XLP", "cik": "0001067983"},
            "Energy": {"symbol": "XLE", "cik": "0001067983"},
            "Industrials": {"symbol": "XLI", "cik": "0001067983"},
            "Materials": {"symbol": "XLB", "cik": "0001067983"},
            "Utilities": {"symbol": "XLU", "cik": "0001067983"},
            "Real Estate": {"symbol": "XLRE", "cik": "0001067983"},
        }

        if enrich_sec and self.query_api is not None:
            for sector, info in sector_etfs.items():
                try:
                    filing = self.fetch_sec_filings(info["cik"], "10-K", limit=1)
                    if filing:
                        info["latest_filing"] = filing[0]["filingDate"]
                        info["total_assets"] = self.extract_financial_metric(
                            filing[0]["linkToHtml"], "totalAssets"
                        )
                        info["expense_ratio"] = self.extract_financial_metric(
                            filing[0]["linkToHtml"], "expenseRatio"
                        )
                except Exception as e:  # noqa: BLE001
                    print(f"Error fetching SEC data for {sector}: {e}")

        return sector_etfs

    def fetch_sec_filings(
        self,
        cik,
        filing_type="10-K",
        start_date=None,
        end_date=None,
        limit=10,
    ):
        if self.query_api is None:
            print("SEC API not configured; skipping filings fetch")
            return []

        query = {
            "query": {
                "query_string": {
                    "query": f'cik:{cik} AND formType:"{filing_type}"'
                }
            },
            "from": 0,
            "size": limit,
            "sort": [{"filingDate": {"order": "desc"}}],
        }

        if start_date:
            date_range = f"[{start_date} TO {end_date or '*'}]"
            query["query"]["query_string"]["query"] += f" AND filingDate:{date_range}"

        try:
            response = self.query_api.get_filings(query)
            return response["filings"]
        except Exception as e:  # noqa: BLE001
            print(f"Error fetching SEC filings: {e}")
            return []

    def extract_financial_metric(self, filing_url, metric_name):
        if self.extractor_api is None:
            return None
        try:
            section = self.extractor_api.get_section(filing_url, metric_name, "text")
            if metric_name in ["totalAssets", "totalLiabilities", "totalEquity"]:
                return self._parse_balance_sheet_metric(section)
            if metric_name in ["revenue", "netIncome", "operatingIncome"]:
                return self._parse_income_statement_metric(section)
            if metric_name in [
                "operatingCashFlow",
                "investingCashFlow",
                "financingCashFlow",
            ]:
                return self._parse_cash_flow_metric(section)
            if metric_name == "expenseRatio":
                return self._parse_expense_ratio(section)
            return self._parse_general_metric(section)
        except Exception as e:  # noqa: BLE001
            print(f"Error extracting metric {metric_name}: {e}")
            return None

    def _parse_balance_sheet_metric(self, text):
        try:
            tables = re.findall(r"<table.*?>(.*?)</table>", text, re.DOTALL)
            for table in tables:
                rows = re.findall(r"<tr.*?>(.*?)</tr>", table, re.DOTALL)
                for row in rows:
                    cells = re.findall(r"<td.*?>(.*?)</td>", row, re.DOTALL)
                    if len(cells) >= 2:
                        metric_name = re.sub(r"<.*?>", "", cells[0]).strip()
                        if any(
                            keyword in metric_name.lower()
                            for keyword in [
                                "total assets",
                                "total liabilities",
                                "total equity",
                            ]
                        ):
                            value = re.sub(r"[^\d.-]", "", cells[1])
                            return float(value) if value else None
            return None
        except Exception as e:  # noqa: BLE001
            print(f"Error parsing balance sheet metric: {e}")
            return None

    def _parse_income_statement_metric(self, text):
        try:
            tables = re.findall(r"<table.*?>(.*?)</table>", text, re.DOTALL)
            for table in tables:
                rows = re.findall(r"<tr.*?>(.*?)</tr>", table, re.DOTALL)
                for row in rows:
                    cells = re.findall(r"<td.*?>(.*?)</td>", row, re.DOTALL)
                    if len(cells) >= 2:
                        metric_name = re.sub(r"<.*?>", "", cells[0]).strip()
                        if any(
                            keyword in metric_name.lower()
                            for keyword in [
                                "revenue",
                                "net income",
                                "operating income",
                            ]
                        ):
                            value = re.sub(r"[^\d.-]", "", cells[1])
                            return float(value) if value else None
            return None
        except Exception as e:  # noqa: BLE001
            print(f"Error parsing income statement metric: {e}")
            return None

    def _parse_cash_flow_metric(self, text):
        try:
            tables = re.findall(r"<table.*?>(.*?)</table>", text, re.DOTALL)
            for table in tables:
                rows = re.findall(r"<tr.*?>(.*?)</tr>", table, re.DOTALL)
                for row in rows:
                    cells = re.findall(r"<td.*?>(.*?)</td>", row, re.DOTALL)
                    if len(cells) >= 2:
                        metric_name = re.sub(r"<.*?>", "", cells[0]).strip()
                        if any(
                            keyword in metric_name.lower()
                            for keyword in [
                                "operating cash",
                                "investing cash",
                                "financing cash",
                            ]
                        ):
                            value = re.sub(r"[^\d.-]", "", cells[1])
                            return float(value) if value else None
            return None
        except Exception as e:  # noqa: BLE001
            print(f"Error parsing cash flow metric: {e}")
            return None

    def _parse_expense_ratio(self, text):
        try:
            pattern = r"expense\s*ratio\s*[:=]?\s*([\d.]+)%?"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100
            return None
        except Exception as e:  # noqa: BLE001
            print(f"Error parsing expense ratio: {e}")
            return None

    def _parse_general_metric(self, text):
        try:
            numbers = re.findall(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?", text)
            if numbers:
                number = numbers[0].replace("$", "").replace(",", "").replace("%", "")
                return float(number)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"Error parsing general metric: {e}")
            return None

    def _cache_path(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        provider_name: str | None = None,
    ) -> str:
        provider_name = provider_name or self.provider_name
        sym_key = "_".join(sorted(symbols))
        # Keep filenames portable
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{provider_name}_{sym_key}_{start_date}_{end_date}")
        return os.path.join(self.cache_dir, f"price_data_{safe}.csv")

    def fetch_historical_data(
        self,
        symbols,
        start_date,
        end_date,
        *,
        provider: str | PriceProvider | None = None,
        use_cache: bool = True,
    ):
        """
        Fetch historical price data via the configured provider and optionally cache it.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start >= end:
            raise ValueError("start_date must be before end_date")

        symbol_list = list(symbols)
        active = (
            resolve_provider(provider, fallback=False)
            if provider is not None
            else self.provider
        )
        provider_name = active.name
        start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        cache_file = self._cache_path(symbol_list, start_s, end_s, provider_name)

        if use_cache and os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not cached.empty and all(s in cached.columns for s in symbol_list):
                cached.index = pd.DatetimeIndex(pd.to_datetime(cached.index)).astype(
                    "datetime64[ns]"
                )
                return cached[symbol_list]

        try:
            df = active.fetch_closes(symbol_list, start_s, end_s)
        except ProviderError as exc:
            raise ValueError(str(exc)) from exc

        if df.empty or df.columns.empty:
            raise ValueError(
                "No price data retrieved for symbols: " + ", ".join(symbol_list)
            )

        df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).astype("datetime64[ns]")
        if use_cache:
            df.to_csv(cache_file)
        return df

    def get_risk_free_rate(self):
        """Fetch risk-free rate data (e.g., 10-year Treasury yield)."""
        try:
            series = self.provider.fetch_series(
                "^TNX",
                (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            )
            return float(series.iloc[-1]) / 100
        except Exception as e:  # noqa: BLE001
            # Stooq uses a different treasury symbol; try yfinance explicitly
            try:
                yf_provider = get_provider("yfinance")
                series = yf_provider.fetch_series(
                    "^TNX",
                    (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d"),
                )
                return float(series.iloc[-1]) / 100
            except Exception:
                print(f"Error fetching risk-free rate: {e}")
                return 0.02

    def get_market_data(self, symbol="^GSPC", start_date=None, end_date=None):
        """Fetch market benchmark data."""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        try:
            return self.provider.fetch_series(symbol, start_s, end_s)
        except Exception as e:  # noqa: BLE001
            # Index symbols differ across vendors; fall back to yfinance
            if self.provider_name != "yfinance":
                try:
                    return get_provider("yfinance").fetch_series(symbol, start_s, end_s)
                except Exception:
                    pass
            print(f"Error fetching market data: {e}")
            return None
