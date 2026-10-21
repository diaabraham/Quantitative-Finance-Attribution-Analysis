"""Abstract market-data provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import pandas as pd


class MarketDataProvider(ABC):
    """Fetch adjusted/close prices for equities and common benchmarks."""

    name: str = "base"

    @abstractmethod
    def fetch_closes(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of adjusted (or close) prices.

        Index: DatetimeIndex (naive UTC/calendar dates)
        Columns: requested symbols (original symbols, not provider-mapped)
        """

    def fetch_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Fetch a single symbol as a Series."""
        df = self.fetch_closes([symbol], start_date, end_date)
        if symbol not in df.columns or df[symbol].dropna().empty:
            return pd.Series(dtype=float, name=symbol)
        return df[symbol].dropna().astype(float)

    def normalize_symbol(self, symbol: str) -> str:
        """Map a canonical symbol to the provider's ticker format."""
        return symbol

    def supports_symbol(self, symbol: str) -> bool:
        """Whether this provider can resolve the symbol."""
        return True

    def describe(self) -> dict:
        """Return non-secret provider metadata for diagnostics."""
        return {"name": self.name}


def normalize_dates(start_date, end_date) -> tuple[str, str]:
    """Coerce start/end to YYYY-MM-DD strings and validate order."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start >= end:
        raise ValueError("start_date must be before end_date")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def to_naive_daily_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert any datetime index to naive calendar dates (ns resolution)."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx.normalize().astype("datetime64[ns]"))
