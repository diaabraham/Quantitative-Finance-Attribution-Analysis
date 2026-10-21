"""Shared aggregates client for Massive.com / Polygon.io daily bars."""

from __future__ import annotations

import os
from typing import Sequence

import pandas as pd

from data.providers.base import PriceProvider, ProviderError, normalize_frame, normalize_symbols


class PolygonCompatibleProvider(PriceProvider):
    """
    Daily adjusted closes via the Massive/Polygon aggregates API.

    Massive is the Polygon.io rebrand (Oct 2025). Keys remain interchangeable;
    only the default REST host differs.
    """

    name = "polygon_compatible"
    requires_api_key = True
    default_base: str = "https://api.polygon.io"
    env_keys: tuple[str, ...] = ("POLYGON_API_KEY", "MASSIVE_API_KEY")
    sdk_import: str = "polygon"  # or "massive"

    def __init__(self, api_key: str | None = None, base: str | None = None):
        self.api_key = api_key or self._resolve_api_key()
        self.base = base or self.default_base
        self._client = None

    def _resolve_api_key(self) -> str | None:
        for key in self.env_keys:
            value = os.getenv(key)
            if value and value.strip() and value.strip() not in {
                "your_api_key_here",
                "YOUR_API_KEY",
                "changeme",
            }:
                return value.strip()
        return None

    def available(self) -> bool:
        return bool(self.api_key)

    def _build_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderError(f"{self.name}: missing API key")

        if self.sdk_import == "massive":
            from massive import RESTClient
        else:
            from polygon import RESTClient

        self._client = RESTClient(api_key=self.api_key, base=self.base)
        return self._client

    def _aggs_to_series(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        client = self._build_client()
        # Prefer list_aggs for pagination; fall back to get_aggs.
        bars = []
        try:
            if hasattr(client, "list_aggs"):
                bars = list(
                    client.list_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="day",
                        from_=start_date,
                        to=end_date,
                        adjusted=True,
                        sort="asc",
                        limit=50000,
                    )
                )
            else:
                bars = list(
                    client.get_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="day",
                        from_=start_date,
                        to=end_date,
                        adjusted=True,
                        sort="asc",
                        limit=50000,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.name}: aggregates failed for {symbol}: {exc}") from exc

        if not bars:
            raise ProviderError(f"{self.name}: no bars for {symbol}")

        index = []
        closes = []
        for bar in bars:
            # SDK objects expose .timestamp / .close; dicts use t/c
            ts = getattr(bar, "timestamp", None)
            if ts is None and isinstance(bar, dict):
                ts = bar.get("t") or bar.get("timestamp")
            close = getattr(bar, "close", None)
            if close is None and isinstance(bar, dict):
                close = bar.get("c") or bar.get("close")
            if ts is None or close is None:
                continue
            index.append(pd.to_datetime(ts, unit="ms", utc=True).tz_convert(None).normalize())
            closes.append(float(close))

        if not closes:
            raise ProviderError(f"{self.name}: could not parse bars for {symbol}")

        series = pd.Series(closes, index=pd.DatetimeIndex(index), name=symbol)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series

    def fetch_closes(
        self,
        symbols: str | Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not self.available():
            raise ProviderError(self.unavailable_reason() or f"{self.name}: unavailable")

        tickers = normalize_symbols(symbols)
        data: dict[str, pd.Series] = {}
        errors: list[str] = []
        for symbol in tickers:
            try:
                data[symbol] = self._aggs_to_series(symbol, start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: {exc}")
        if not data:
            detail = "; ".join(errors) if errors else "no rows"
            raise ProviderError(f"{self.name}: no price data retrieved ({detail})")
        return normalize_frame(pd.DataFrame(data))


class MassiveProvider(PolygonCompatibleProvider):
    name = "massive"
    default_base = "https://api.massive.com"
    env_keys = ("MASSIVE_API_KEY", "POLYGON_API_KEY")
    sdk_import = "massive"


class PolygonProvider(PolygonCompatibleProvider):
    name = "polygon"
    default_base = "https://api.polygon.io"
    env_keys = ("POLYGON_API_KEY", "MASSIVE_API_KEY")
    sdk_import = "polygon"
