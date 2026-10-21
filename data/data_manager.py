import os
import re
from datetime import datetime, timedelta
from typing import Iterable, Optional, Union

import pandas as pd
from dotenv import load_dotenv

from data.providers import MarketDataProvider, build_provider
from data.providers.errors import MarketDataError

load_dotenv()


class DataManager:
    def __init__(
        self,
        data_dir: str = "data/raw",
        cache_dir: str = "data/cache",
        provider: Optional[Union[str, MarketDataProvider]] = None,
        require_sec: bool = False,
    ):
        """
        Initialize the Data Manager.

        Parameters
        ----------
        data_dir : str
            Directory to store raw data
        cache_dir : str
            Directory to store processed/cached data
        provider : str | MarketDataProvider | None
            Market-data provider name or instance. Defaults to MARKET_DATA_PROVIDER
            (with MARKET_DATA_FALLBACKS).
        require_sec : bool
            If True, raise when SEC_API_KEY is missing. SEC is optional for price paths.
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.ensure_directories()

        self.sec_api_key = os.getenv("SEC_API_KEY")
        self.extractor_api = None
        self.query_api = None
        if self.sec_api_key and self.sec_api_key != "your_api_key_here":
            try:
                from sec_api import ExtractorApi, QueryApi

                self.extractor_api = ExtractorApi(self.sec_api_key)
                self.query_api = QueryApi(self.sec_api_key)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: SEC API client init failed: {exc}")
        elif require_sec:
            raise ValueError("SEC_API_KEY not found in environment variables")

        if isinstance(provider, MarketDataProvider):
            self.provider = provider
        else:
            self.provider = build_provider(provider)

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "sec"), exist_ok=True)

    def get_sector_etfs(self):
        """
        Get sector ETF information. Uses hardcoded Select Sector SPDRs; optionally
        enriches from SEC when a real API key is configured.
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

        if not self.query_api:
            return sector_etfs

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
        self, cik, filing_type="10-K", start_date=None, end_date=None, limit=10
    ):
        if not self.query_api:
            print("SEC API unavailable (missing/invalid SEC_API_KEY); skipping filings")
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
        if not self.extractor_api:
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
                            k in metric_name.lower()
                            for k in ["total assets", "total liabilities", "total equity"]
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
                            k in metric_name.lower()
                            for k in ["revenue", "net income", "operating income"]
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
                            k in metric_name.lower()
                            for k in ["operating cash", "investing cash", "financing cash"]
                        ):
                            value = re.sub(r"[^\d.-]", "", cells[1])
                            return float(value) if value else None
            return None
        except Exception as e:  # noqa: BLE001
            print(f"Error parsing cash flow metric: {e}")
            return None

    def _parse_expense_ratio(self, text):
        try:
            match = re.search(
                r"expense\s*ratio\s*[:=]?\s*([\d.]+)%?", text, re.IGNORECASE
            )
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

    def _cache_path(self, symbols: Iterable[str], start_date: str, end_date: str) -> str:
        provider_name = getattr(self.provider, "name", "unknown")
        sym_key = "-".join(sorted(symbols))
        safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{provider_name}_{sym_key}_{start_date}_{end_date}")
        return os.path.join(self.cache_dir, f"price_data_{safe}.csv")

    def fetch_historical_data(self, symbols, start_date, end_date, use_cache: bool = True):
        """
        Fetch historical price data via the configured market-data provider.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start >= end:
            raise ValueError("start_date must be before end_date")

        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")
        symbols = list(symbols)
        cache_file = self._cache_path(symbols, start_s, end_s)

        if use_cache and os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not cached.empty:
                cached.index = pd.DatetimeIndex(pd.to_datetime(cached.index)).astype(
                    "datetime64[ns]"
                )
                return cached

        try:
            df = self.provider.fetch_closes(symbols, start_s, end_s)
        except MarketDataError:
            raise
        except Exception as e:  # noqa: BLE001
            raise MarketDataError(f"Provider {self.provider.name} failed: {e}") from e

        if df is None or df.empty or df.columns.empty:
            raise ValueError(
                "No price data retrieved for symbols: " + ", ".join(symbols)
            )

        df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).astype("datetime64[ns]")
        df.to_csv(cache_file)
        return df

    def get_risk_free_rate(self):
        """Fetch risk-free rate data (e.g., 10-year Treasury yield)."""
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        try:
            s = self.provider.fetch_series("^TNX", start, end)
            if s is None or s.empty:
                return 0.02
            last = float(s.iloc[-1])
            # Yields may be percent (4.2) or decimal (0.042)
            return last / 100.0 if last > 1.0 else last
        except Exception as e:  # noqa: BLE001
            print(f"Error fetching risk-free rate: {e}")
            return 0.02

    def get_market_data(self, symbol="^GSPC", start_date=None, end_date=None):
        """Fetch market benchmark data"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        try:
            s = self.provider.fetch_series(symbol, start_s, end_s)
            if s is None or s.empty:
                return None
            return s.astype(float)
        except Exception as e:  # noqa: BLE001
            print(f"Error fetching market data: {e}")
            return None
