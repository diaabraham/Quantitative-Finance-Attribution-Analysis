"""Yahoo Finance market-data adapter via yfinance."""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import yfinance as yf

from .base import MarketDataProvider, normalize_dates, to_naive_daily_index
from .errors import ProviderEmptyDataError, ProviderRateLimitError


def _adjusted_close_series(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Extract adjusted close from yfinance output, including modern MultiIndex frames
    where adjusted prices are in 'Close' (no separate 'Adj Close' column).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        adj = ("Adj Close", symbol)
        close = ("Close", symbol)
        if adj in df.columns:
            return df[adj].squeeze()
        if close in df.columns:
            return df[close].squeeze()
        tickers = df.columns.get_level_values(-1).unique()
        if len(tickers) == 1:
            sym = tickers[0]
            if ("Adj Close", sym) in df.columns:
                return df[("Adj Close", sym)].squeeze()
            return df[("Close", sym)].squeeze()
        raise KeyError(f"Could not resolve price columns for {symbol}")
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    return df["Close"]


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def __init__(self, max_retries: int = 3, pause: float = 0.35):
        self.max_retries = max_retries
        self.pause = pause

    def fetch_closes(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        start, end = normalize_dates(start_date, end_date)
        symbols = list(symbols)
        frames = {}
        errors = []

        for symbol in symbols:
            series = None
            last_err = None
            for attempt in range(self.max_retries):
                try:
                    raw = yf.download(
                        symbol,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                    )
                    series = _adjusted_close_series(raw, symbol)
                    if series is not None and not series.empty:
                        break
                    # Fallback to Ticker.history for stubborn MultiIndex cases
                    hist = yf.Ticker(symbol).history(
                        start=start, end=end, auto_adjust=True
                    )
                    if hist is not None and not hist.empty and "Close" in hist.columns:
                        series = hist["Close"]
                        break
                    last_err = ProviderEmptyDataError(
                        f"yfinance returned no rows for {symbol}"
                    )
                except Exception as exc:  # noqa: BLE001 - surface provider errors
                    last_err = exc
                    msg = str(exc).lower()
                    if "rate" in msg or "too many" in msg or "429" in msg:
                        if attempt + 1 >= self.max_retries:
                            raise ProviderRateLimitError(
                                f"yfinance rate-limited for {symbol}: {exc}"
                            ) from exc
                        time.sleep(self.pause * (2**attempt))
                        continue
                    if attempt + 1 >= self.max_retries:
                        break
                    time.sleep(self.pause * (attempt + 1))
            if series is None or series.empty:
                errors.append(f"{symbol}: {last_err}")
                continue
            s = pd.Series(series.values, index=to_naive_daily_index(series.index), name=symbol)
            frames[symbol] = s.astype(float)

        if not frames:
            raise ProviderEmptyDataError(
                "No price data retrieved from yfinance for symbols: "
                + ", ".join(symbols)
                + (f" ({'; '.join(errors)})" if errors else "")
            )

        df = pd.DataFrame(frames)
        df.index = to_naive_daily_index(df.index)
        return df.sort_index()
