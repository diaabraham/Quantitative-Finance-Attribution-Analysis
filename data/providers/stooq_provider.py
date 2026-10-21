"""Stooq CSV historical market-data adapter."""

from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import pandas as pd
import requests

from .base import MarketDataProvider, normalize_dates, to_naive_daily_index
from .errors import (
    ProviderAuthError,
    ProviderEmptyDataError,
    ProviderRateLimitError,
)

# Canonical -> Stooq ticker
_SYMBOL_MAP = {
    "^GSPC": "^spx",
    "^SPX": "^spx",
    "SPX": "^spx",
    "^TNX": "10yty.b",  # US 10Y yield (approx)
    "^IRX": "13wty.b",
}

_INDEX_LIKE = {"^spx", "^dji", "^ndx", "^rut"}


class StooqProvider(MarketDataProvider):
    name = "stooq"
    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        max_retries: int = 3,
        pause: float = 0.4,
        country: str = "us",
    ):
        self.api_key = api_key or os.getenv("STOOQ_API_KEY")
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.pause = pause
        self.country = country.lower()
        # Stooq bot-challenges the default python-requests UA; override it.
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    def normalize_symbol(self, symbol: str) -> str:
        if symbol in _SYMBOL_MAP:
            return _SYMBOL_MAP[symbol]
        if symbol.startswith("^"):
            return symbol.lower()
        # Already exchange-qualified (e.g. xlk.us)
        if "." in symbol:
            return symbol.lower()
        return f"{symbol.lower()}.{self.country}"

    def fetch_closes(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        start, end = normalize_dates(start_date, end_date)
        d1 = start.replace("-", "")
        d2 = end.replace("-", "")
        frames = {}
        errors = []

        for symbol in symbols:
            mapped = self.normalize_symbol(symbol)
            try:
                series = self._download_close(mapped, d1, d2)
            except (ProviderAuthError, ProviderRateLimitError):
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: {exc}")
                continue
            if series.empty:
                errors.append(f"{symbol}: empty")
                continue
            frames[symbol] = series

        if not frames:
            raise ProviderEmptyDataError(
                "No price data retrieved from Stooq for symbols: "
                + ", ".join(symbols)
                + (f" ({'; '.join(errors)})" if errors else "")
            )
        df = pd.DataFrame(frames)
        df.index = to_naive_daily_index(df.index)
        return df.sort_index()

    def _download_close(self, stooq_symbol: str, d1: str, d2: str) -> pd.Series:
        params = {"s": stooq_symbol, "d1": d1, "d2": d2, "i": "d"}
        if self.api_key:
            params["apikey"] = self.api_key

        last_err = None
        for attempt in range(self.max_retries):
            # Stooq is picky about ^ in URLs; encode path-style query manually if needed
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(self.pause * (attempt + 1))
                continue

            text = (resp.text or "").strip()
            lower = text.lower()

            if resp.status_code == 401 or "apikey" in lower and "get your apikey" in lower:
                raise ProviderAuthError(
                    "Stooq requires STOOQ_API_KEY. Obtain one via "
                    "https://stooq.com/q/d/?s=aapl.us&get_apikey and set STOOQ_API_KEY."
                )
            if resp.status_code == 429 or "exceeded the daily hits limit" in lower:
                if attempt + 1 >= self.max_retries:
                    raise ProviderRateLimitError("Stooq daily hit limit exceeded")
                time.sleep(self.pause * (2**attempt))
                continue
            if resp.status_code >= 400:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {text[:200]}")
                time.sleep(self.pause * (attempt + 1))
                continue
            if "date" not in lower.split("\n", 1)[0]:
                # JS challenge / HTML interstitial
                if "<html" in lower or "verify your browser" in lower:
                    last_err = ProviderAuthError(
                        "Stooq returned a browser/JS challenge. "
                        "Set STOOQ_API_KEY from https://stooq.com/q/d/?s=aapl.us&get_apikey"
                    )
                    break
                last_err = RuntimeError(f"Unexpected Stooq payload: {text[:200]}")
                break

            from io import StringIO

            frame = pd.read_csv(StringIO(text))
            if "Date" not in frame.columns or "Close" not in frame.columns:
                last_err = RuntimeError(f"Stooq CSV missing Date/Close: {frame.columns}")
                break
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame = frame.set_index("Date").sort_index()
            series = frame["Close"].astype(float)
            series.index = to_naive_daily_index(series.index)
            return series

        if isinstance(last_err, ProviderAuthError):
            raise last_err
        raise ProviderEmptyDataError(
            f"Stooq failed for {stooq_symbol}: {last_err or 'unknown error'}"
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "has_api_key": bool(self.api_key),
            "country": self.country,
        }
