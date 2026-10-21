"""Databento historical OHLCV-1d equities closes."""

from __future__ import annotations

import os
from typing import Sequence

import pandas as pd

from data.providers.base import PriceProvider, ProviderError, normalize_frame, normalize_symbols


class DatabentoProvider(PriceProvider):
    name = "databento"
    requires_api_key = True
    # Official consolidated US equities end-of-day summary
    default_dataset = "EQUS.SUMMARY"

    def __init__(
        self,
        api_key: str | None = None,
        dataset: str | None = None,
    ):
        raw_key = api_key or os.getenv("DATABENTO_API_KEY") or None
        if raw_key and raw_key.strip() in {
            "your_api_key_here",
            "YOUR_API_KEY",
            "changeme",
        }:
            raw_key = None
        self.api_key = raw_key.strip() if raw_key else None
        self.dataset = dataset or os.getenv("DATABENTO_DATASET") or self.default_dataset
        self._client = None

    def available(self) -> bool:
        return bool(self.api_key)

    def _build_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderError("databento: missing DATABENTO_API_KEY")
        import databento as db

        self._client = db.Historical(key=self.api_key)
        return self._client

    def fetch_closes(
        self,
        symbols: str | Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not self.available():
            raise ProviderError(self.unavailable_reason() or "databento: unavailable")

        tickers = normalize_symbols(symbols)
        # Databento end is exclusive; bump by one day for inclusive end semantics
        end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            client = self._build_client()
            store = client.timeseries.get_range(
                dataset=self.dataset,
                symbols=tickers,
                schema="ohlcv-1d",
                start=start_date,
                end=end_exclusive,
                stype_in="raw_symbol",
                stype_out="raw_symbol",
            )
            raw = store.to_df()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"databento: get_range failed: {exc}") from exc

        if raw is None or raw.empty:
            raise ProviderError("databento: empty response")

        frame = raw.copy()
        if "symbol" not in frame.columns or "close" not in frame.columns:
            raise ProviderError(f"databento: unexpected columns {list(frame.columns)}")

        if not isinstance(frame.index, pd.DatetimeIndex):
            if "ts_event" in frame.columns:
                frame = frame.set_index("ts_event")
            else:
                raise ProviderError("databento: missing datetime index")

        dates = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        frame = frame.assign(_date=dates)
        wide = (
            frame.groupby(["_date", "symbol"], sort=True)["close"]
            .last()
            .unstack("symbol")
        )
        wide.index = pd.DatetimeIndex(wide.index)

        ordered = [s for s in tickers if s in wide.columns]
        if not ordered:
            raise ProviderError(
                f"databento: none of the requested symbols returned ({tickers})"
            )
        return normalize_frame(wide[ordered])
