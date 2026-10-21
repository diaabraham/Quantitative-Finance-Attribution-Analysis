"""Shared REST client for Polygon.io / Massive.com aggregates API."""

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

# Yahoo-style / canonical symbols -> Polygon/Massive tickers
_SYMBOL_MAP = {
    "^GSPC": "I:SPX",
    "^SPX": "I:SPX",
    "SPX": "I:SPX",
    "^DJI": "I:DJI",
    "^IXIC": "I:COMP",
    "^TNX": "I:TNX",
}


class PolygonCompatibleProvider(MarketDataProvider):
    """
    Daily OHLC via /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}.

    Works against either api.polygon.io or api.massive.com (same schema).
    """

    name = "polygon"
    base_url = "https://api.polygon.io"
    api_key_env = "POLYGON_API_KEY"
    alt_api_key_envs = ("MASSIVE_API_KEY",)

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        max_retries: int = 4,
        pause: float = 0.25,
        adjusted: bool = True,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or self._resolve_api_key()
        if not self.api_key:
            raise ProviderAuthError(
                f"{self.name} requires an API key. Set {self.api_key_env}"
                + (
                    f" (or {' / '.join(self.alt_api_key_envs)})"
                    if self.alt_api_key_envs
                    else ""
                )
                + "."
            )
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.pause = pause
        self.adjusted = adjusted
        if base_url:
            self.base_url = base_url.rstrip("/")

    def _resolve_api_key(self) -> Optional[str]:
        key = os.getenv(self.api_key_env)
        if key:
            return key
        for env in self.alt_api_key_envs:
            key = os.getenv(env)
            if key:
                return key
        return None

    def normalize_symbol(self, symbol: str) -> str:
        if symbol in _SYMBOL_MAP:
            return _SYMBOL_MAP[symbol]
        if symbol.startswith("^"):
            # Best-effort: treat remaining as index ticker
            return f"I:{symbol.lstrip('^').upper()}"
        return symbol.upper()

    def fetch_closes(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        start, end = normalize_dates(start_date, end_date)
        frames = {}
        errors = []
        for symbol in symbols:
            try:
                series = self._fetch_symbol_close(symbol, start, end)
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
                f"No price data retrieved from {self.name} for symbols: "
                + ", ".join(symbols)
                + (f" ({'; '.join(errors)})" if errors else "")
            )
        df = pd.DataFrame(frames)
        df.index = to_naive_daily_index(df.index)
        return df.sort_index()

    def _fetch_symbol_close(self, symbol: str, start: str, end: str) -> pd.Series:
        ticker = self.normalize_symbol(symbol)
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {
            "adjusted": str(self.adjusted).lower(),
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        results = []
        next_url = None

        for attempt in range(self.max_retries):
            try:
                if next_url:
                    resp = self.session.get(
                        next_url,
                        params={"apiKey": self.api_key},
                        timeout=30,
                    )
                else:
                    resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                if attempt + 1 >= self.max_retries:
                    raise
                time.sleep(self.pause * (2**attempt))
                continue

            if resp.status_code in (401, 403):
                raise ProviderAuthError(
                    f"{self.name} auth failed ({resp.status_code}): {resp.text[:200]}"
                )
            if resp.status_code == 429:
                if attempt + 1 >= self.max_retries:
                    raise ProviderRateLimitError(f"{self.name} rate limit exceeded")
                time.sleep(self.pause * (2**attempt) + 1.0)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.name} HTTP {resp.status_code}: {resp.text[:300]}"
                )

            payload = resp.json()
            status = str(payload.get("status", "")).upper()
            if status in {"ERROR", "NOT_AUTHORIZED"}:
                raise ProviderAuthError(f"{self.name} error: {payload}")

            chunk = payload.get("results") or []
            results.extend(chunk)
            next_url = payload.get("next_url")
            if next_url:
                # Continue pagination without burning the attempt counter
                attempt = 0
                continue
            break
        else:
            raise ProviderEmptyDataError(
                f"{self.name} exhausted retries for {symbol} ({ticker})"
            )

        if not results:
            return pd.Series(dtype=float, name=symbol)

        frame = pd.DataFrame(results)
        # Prefer close; Polygon/Massive 'c' is split-adjusted when adjusted=true
        frame["date"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
        series = pd.Series(
            frame["c"].astype(float).values,
            index=to_naive_daily_index(frame["date"]),
            name=symbol,
        )
        return series[~series.index.duplicated(keep="last")].sort_index()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "adjusted": self.adjusted,
        }


class PolygonProvider(PolygonCompatibleProvider):
    name = "polygon"
    base_url = "https://api.polygon.io"
    api_key_env = "POLYGON_API_KEY"
    alt_api_key_envs = ("MASSIVE_API_KEY",)


class MassiveProvider(PolygonCompatibleProvider):
    """Massive.com (formerly Polygon.io) — same API, new default host."""

    name = "massive"
    base_url = "https://api.massive.com"
    api_key_env = "MASSIVE_API_KEY"
    alt_api_key_envs = ("POLYGON_API_KEY",)
