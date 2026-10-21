"""Databento historical market-data adapter."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd

from .base import MarketDataProvider, normalize_dates, to_naive_daily_index
from .errors import ProviderAuthError, ProviderEmptyDataError

# Databento EQUS.SUMMARY uses raw equity symbols; map Yahoo indices to liquid proxies.
_SYMBOL_MAP = {
    "^GSPC": "SPY",
    "^SPX": "SPY",
    "SPX": "SPY",
    "^DJI": "DIA",
    "^IXIC": "QQQ",
    "^TNX": "IEF",  # duration proxy when true yield unavailable on EQUS.SUMMARY
}


class DatabentoProvider(MarketDataProvider):
    name = "databento"

    def __init__(
        self,
        api_key: Optional[str] = None,
        dataset: Optional[str] = None,
        schema: str = "ohlcv-1d",
    ):
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise ProviderAuthError(
                "databento requires DATABENTO_API_KEY in the environment."
            )
        self.dataset = dataset or os.getenv("DATABENTO_DATASET", "EQUS.SUMMARY")
        self.schema = schema
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            try:
                import databento as db
            except ImportError as exc:
                raise ProviderAuthError(
                    "databento package is not installed. pip install databento"
                ) from exc
            self._client = db.Historical(self.api_key)
        return self._client

    def normalize_symbol(self, symbol: str) -> str:
        if symbol in _SYMBOL_MAP:
            return _SYMBOL_MAP[symbol]
        return symbol.lstrip("^").upper()

    def fetch_closes(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        start, end = normalize_dates(start_date, end_date)
        symbols = list(symbols)
        mapped = [self.normalize_symbol(s) for s in symbols]
        # Preserve reverse map for original column names (first wins on collisions)
        rev = {}
        for orig, m in zip(symbols, mapped):
            rev.setdefault(m, orig)

        client = self._client_or_create()
        try:
            store = client.timeseries.get_range(
                dataset=self.dataset,
                symbols=list(dict.fromkeys(mapped)),
                schema=self.schema,
                start=start,
                end=end,
                stype_in="raw_symbol",
            )
            raw = store.to_df()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "auth" in msg or "401" in msg or "403" in msg or "key" in msg:
                raise ProviderAuthError(f"databento auth failed: {exc}") from exc
            raise ProviderEmptyDataError(f"databento request failed: {exc}") from exc

        if raw is None or raw.empty:
            raise ProviderEmptyDataError(
                f"databento returned no rows for {symbols} on {self.dataset}"
            )

        if "symbol" not in raw.columns:
            raise ProviderEmptyDataError("databento frame missing 'symbol' column")

        # Index is typically ts_event
        frame = raw.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            if "ts_event" in frame.columns:
                frame = frame.set_index("ts_event")
            else:
                frame.index = pd.to_datetime(frame.index)

        close_col = "close" if "close" in frame.columns else "Close"
        if close_col not in frame.columns:
            raise ProviderEmptyDataError(
                f"databento frame missing close column; have {list(frame.columns)}"
            )

        series_map = {}
        for mapped_sym, group in frame.groupby("symbol"):
            orig = rev.get(str(mapped_sym), str(mapped_sym))
            s = group[close_col].astype(float)
            s.index = to_naive_daily_index(s.index)
            series_map[orig] = s[~s.index.duplicated(keep="last")]

        if not series_map:
            raise ProviderEmptyDataError("databento produced no close series")

        df = pd.DataFrame(series_map)
        df.index = to_naive_daily_index(df.index)
        return df.sort_index()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "schema": self.schema,
            "has_api_key": bool(self.api_key),
        }
