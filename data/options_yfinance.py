"""
Live option-chain helpers via yfinance.

Isolated from the modeled ``OptionsCreditPutPath`` (Black–Scholes on underlying).
Use this for real bid/ask/strike tape when Yahoo quotes are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class OptionChainSnapshot:
    underlying: str
    expiry: str
    spot: float
    calls: pd.DataFrame
    puts: pd.DataFrame

    @property
    def n_contracts(self) -> int:
        return int(len(self.calls) + len(self.puts))


def list_expiries(symbol: str) -> List[str]:
    t = yf.Ticker(symbol)
    opts = list(t.options or [])
    return opts


def fetch_option_chain(
    symbol: str,
    expiry: Optional[str] = None,
) -> OptionChainSnapshot:
    """
    Fetch one expiry's call/put chain with bid/ask when Yahoo provides them.
    """
    t = yf.Ticker(symbol)
    expiries = list(t.options or [])
    if not expiries:
        raise RuntimeError(f"No option expiries for {symbol}")
    exp = expiry or expiries[0]
    if exp not in expiries:
        raise ValueError(f"Expiry {exp} not in {expiries[:5]}…")

    spot = float("nan")
    try:
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            spot = float(getattr(fi, "last_price", None) or getattr(fi, "lastPrice", float("nan")))
    except Exception:  # noqa: BLE001
        spot = float("nan")
    if not (spot == spot) or spot <= 0:  # NaN check
        hist = t.history(period="5d", auto_adjust=True)
        if hist is None or hist.empty:
            raise RuntimeError(f"Could not resolve spot for {symbol}")
        spot = float(hist["Close"].iloc[-1])

    chain = t.option_chain(exp)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    for df in (calls, puts):
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
            df["spread"] = (df["ask"].astype(float) - df["bid"].astype(float)).clip(lower=0)
            df["spread_pct"] = df["spread"] / df["mid"].replace(0, pd.NA)
    return OptionChainSnapshot(symbol, exp, spot, calls, puts)


def near_atm_puts(
    snap: OptionChainSnapshot,
    n: int = 5,
) -> pd.DataFrame:
    """Return *n* puts closest to spot by strike."""
    puts = snap.puts.dropna(subset=["strike"]).copy()
    if puts.empty:
        return puts
    puts["dist"] = (puts["strike"].astype(float) - snap.spot).abs()
    return puts.sort_values("dist").head(n)


def quote_quality_ok(row: pd.Series, max_spread_pct: float = 0.5) -> bool:
    """True when bid/ask mid is usable (finite, positive, not absurdly wide)."""
    bid = float(row.get("bid", float("nan")))
    ask = float(row.get("ask", float("nan")))
    if not (bid == bid and ask == ask) or bid < 0 or ask <= 0 or ask < bid:
        return False
    mid = 0.5 * (bid + ask)
    if mid <= 0:
        return False
    return ((ask - bid) / mid) <= max_spread_pct


def summarize_chain(snap: OptionChainSnapshot) -> Tuple[int, int, int]:
    """Return (n_calls, n_puts, n_puts_with_usable_quotes)."""
    n_ok = 0
    if "bid" in snap.puts.columns and "ask" in snap.puts.columns:
        for _, row in snap.puts.iterrows():
            if quote_quality_ok(row):
                n_ok += 1
    return len(snap.calls), len(snap.puts), n_ok
