"""Stooq daily CSV historical prices (free / optional API key)."""

from __future__ import annotations

import io
import os
from typing import Sequence

import pandas as pd
import requests

from data.providers.base import PriceProvider, ProviderError, normalize_frame, normalize_symbols

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Common Yahoo-style index/ticker aliases → Stooq symbols
_STOOQ_ALIASES = {
    "^GSPC": "^spx",
    "^SPX": "^spx",
    "^DJI": "^dji",
    "^IXIC": "^ndq",
    "^TNX": "10yty",
}


def to_stooq_symbol(symbol: str) -> str:
    sym = symbol.strip()
    upper = sym.upper()
    if upper in _STOOQ_ALIASES:
        return _STOOQ_ALIASES[upper]
    if sym.startswith("^"):
        return sym.lower()
    if "." in sym:
        # already exchange-qualified (e.g. XLK.US)
        return sym.lower()
    return f"{sym.lower()}.us"


class StooqProvider(PriceProvider):
    name = "stooq"
    requires_api_key = False  # key optional; required by Stooq for some IPs since 2026
    base_url = "https://stooq.com/q/d/l/"

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or os.getenv("STOOQ_API_KEY") or None
        self.session = session or requests.Session()

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def _download_one(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        stooq_symbol = to_stooq_symbol(symbol)
        params = {
            "s": stooq_symbol,
            "i": "d",
            "d1": pd.Timestamp(start_date).strftime("%Y%m%d"),
            "d2": pd.Timestamp(end_date).strftime("%Y%m%d"),
        }
        if self.api_key:
            params["apikey"] = self.api_key

        response = self.session.get(
            self.base_url,
            params=params,
            headers=_HEADERS,
            timeout=30,
        )
        text = (response.text or "").strip()
        if response.status_code != 200:
            raise ProviderError(
                f"stooq: HTTP {response.status_code} for {symbol} ({stooq_symbol})"
            )
        if "Exceeded the daily hits limit" in text:
            raise ProviderError("stooq: daily request quota exceeded")
        if "Get your apikey" in text or "get_apikey" in text:
            raise ProviderError(
                "stooq: API key required (set STOOQ_API_KEY from stooq.com captcha link)"
            )
        # Stooq bot / captcha interstitial (common from datacenter IPs)
        lowered = text.lower()
        if (
            text.startswith("<!DOCTYPE html")
            or text.startswith("<html")
            or "noscript" in lowered
            or "cf-browser-verification" in lowered
            or "just a moment" in lowered
        ):
            raise ProviderError(
                "stooq: blocked by bot/captcha challenge "
                "(set STOOQ_API_KEY from https://stooq.com/q/d/?s=AAPL.US&get_apikey)"
            )
        if not text.startswith("Date,"):
            raise ProviderError(
                f"stooq: unexpected payload for {symbol} ({stooq_symbol}): {text[:120]!r}"
            )

        frame = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
        if frame.empty or "Close" not in frame.columns:
            raise ProviderError(f"stooq: empty CSV for {symbol}")
        series = frame["Close"].sort_index().astype(float)
        series.name = symbol
        return series

    def fetch_closes(
        self,
        symbols: str | Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        tickers = normalize_symbols(symbols)
        data: dict[str, pd.Series] = {}
        errors: list[str] = []
        for symbol in tickers:
            try:
                data[symbol] = self._download_one(symbol, start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: {exc}")
        if not data:
            detail = "; ".join(errors) if errors else "no rows"
            raise ProviderError(f"stooq: no price data retrieved ({detail})")
        return normalize_frame(pd.DataFrame(data))
