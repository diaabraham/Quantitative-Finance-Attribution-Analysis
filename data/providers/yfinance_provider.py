"""Yahoo Finance historical prices via yfinance."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import yfinance as yf

from data.providers.base import PriceProvider, ProviderError, normalize_frame, normalize_symbols


def _extract_close(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Extract adjusted/close series from modern yfinance MultiIndex frames."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        for key in (("Adj Close", symbol), ("Close", symbol)):
            if key in df.columns:
                return df[key].squeeze()
        tickers = df.columns.get_level_values(-1).unique()
        if len(tickers) == 1:
            sym = tickers[0]
            if ("Adj Close", sym) in df.columns:
                return df[("Adj Close", sym)].squeeze()
            if ("Close", sym) in df.columns:
                return df[("Close", sym)].squeeze()
        raise ProviderError(f"yfinance: could not resolve close column for {symbol}")

    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if "Close" in df.columns:
        return df["Close"]
    raise ProviderError(f"yfinance: no Close/Adj Close columns for {symbol}")


class YFinanceProvider(PriceProvider):
    name = "yfinance"
    requires_api_key = False

    def available(self) -> bool:
        return True

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
                raw = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                series = _extract_close(raw, symbol).dropna()
                if series.empty:
                    errors.append(f"{symbol}: empty")
                    continue
                data[symbol] = series
            except Exception as exc:  # noqa: BLE001 - surface per-symbol failures
                errors.append(f"{symbol}: {exc}")

        if not data:
            detail = "; ".join(errors) if errors else "no rows"
            raise ProviderError(f"yfinance: no price data retrieved ({detail})")

        return normalize_frame(pd.DataFrame(data))
