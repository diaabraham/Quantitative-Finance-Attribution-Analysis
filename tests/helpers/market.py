"""OHLC helpers — live yfinance path + synthetic constructors for unit tests."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_ohlc_yfinance(
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Live daily OHLCV via yfinance (auto_adjust). Empty frame on failure."""
    hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.rename(columns=str.title)
    need = ["Open", "High", "Low", "Close"]
    if any(c not in hist.columns for c in need):
        return pd.DataFrame()
    out = hist[need].copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    return out.dropna()


def make_ohlc_from_close(close: pd.Series, range_frac: float = 0.01) -> pd.DataFrame:
    c = close.astype(float)
    return pd.DataFrame(
        {
            "Open": c,
            "High": c * (1.0 + range_frac),
            "Low": c * (1.0 - range_frac),
            "Close": c,
        },
        index=c.index,
    )


def synthetic_ohlc(
    n: int = 120,
    start: str = "2023-01-01",
    seed: Optional[int] = 0,
    level: float = 100.0,
    vol: float = 0.012,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B")
    rets = rng.normal(0.0003, vol, size=n)
    close = level * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.002, 0.015, size=n))
    low = close * (1.0 - rng.uniform(0.002, 0.015, size=n))
    open_ = close * (1.0 + rng.normal(0, 0.002, size=n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=idx,
    )
