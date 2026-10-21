"""Shared interfaces for historical price providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import pandas as pd


class ProviderError(RuntimeError):
    """Raised when a provider cannot return usable market data."""


def normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    if isinstance(symbols, str):
        return [symbols]
    out = [str(s).strip() for s in symbols if str(s).strip()]
    if not out:
        raise ValueError("symbols must be a non-empty string or sequence")
    return out


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a timezone-naive DatetimeIndex of daily closes."""
    if df is None or df.empty:
        raise ProviderError("provider returned empty price frame")
    frame = df.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    # Drop all-NaN columns but keep partial coverage for diagnostics upstream
    if frame.isna().all().all():
        raise ProviderError("provider returned only NaN prices")
    return frame.astype(float)


class PriceProvider(ABC):
    """Fetch daily close/adjusted-close series for one or more symbols."""

    name: str = "base"
    requires_api_key: bool = False

    @abstractmethod
    def available(self) -> bool:
        """Return True when this provider can be used in the current environment."""

    @abstractmethod
    def fetch_closes(
        self,
        symbols: str | Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of daily closes indexed by date with one column per symbol.

        Prefer split-/dividend-adjusted closes when the vendor provides them.
        """

    def fetch_series(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        frame = self.fetch_closes([symbol], start_date, end_date)
        if symbol not in frame.columns:
            raise ProviderError(f"{self.name}: missing series for {symbol}")
        series = frame[symbol].dropna()
        if series.empty:
            raise ProviderError(f"{self.name}: empty series for {symbol}")
        return series

    def unavailable_reason(self) -> str | None:
        if self.available():
            return None
        if self.requires_api_key:
            return f"{self.name}: missing API key"
        return f"{self.name}: unavailable"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} available={self.available()}>"
