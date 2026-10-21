"""
Live smoke tests against free / optionally keyed market-data providers.

Paid providers (massive, polygon, databento) are skipped unless API keys exist.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from data.providers import get_provider
from data.providers.base import ProviderError
from main import SectorRotationAlpha

SYMBOLS = ["XLK", "XLF", "XLV"]
START = "2024-06-03"
END = "2024-08-30"


def _has_real_key(*names: str) -> bool:
    placeholders = {"", "your_api_key_here", "YOUR_API_KEY", "changeme"}
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value and value not in placeholders:
            return True
    return False


@pytest.mark.live
@pytest.mark.parametrize("provider_name", ["yfinance", "stooq"])
def test_free_provider_fetch(provider_name):
    provider = get_provider(provider_name)
    try:
        frame = provider.fetch_closes(SYMBOLS, START, END)
    except ProviderError as exc:
        msg = str(exc)
        if provider_name == "stooq" and (
            "API key required" in msg
            or "bot/captcha" in msg
            or "daily request quota" in msg
        ):
            pytest.skip(msg)
        raise
    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert set(SYMBOLS).issubset(set(frame.columns))
    assert frame.index.is_monotonic_increasing
    # sanity: ETF prices are positive and not constant
    assert (frame > 0).all().all()
    assert frame.std().min() > 0


@pytest.mark.live
@pytest.mark.parametrize(
    "provider_name,env_vars",
    [
        ("massive", ("MASSIVE_API_KEY", "POLYGON_API_KEY")),
        ("polygon", ("POLYGON_API_KEY", "MASSIVE_API_KEY")),
        ("databento", ("DATABENTO_API_KEY",)),
    ],
)
def test_paid_provider_fetch(provider_name, env_vars):
    if not _has_real_key(*env_vars):
        pytest.skip(f"no API key for {provider_name}")
    provider = get_provider(provider_name)
    assert provider.available()
    frame = provider.fetch_closes(SYMBOLS, START, END)
    assert not frame.empty
    assert set(SYMBOLS).issubset(set(frame.columns))


@pytest.mark.live
def test_yfinance_backtest_smoke():
    strategy = SectorRotationAlpha(provider="yfinance")
    strategy.fetch_data(START, END, symbols=SYMBOLS, use_cache=False)
    strategy.calculate_momentum_indicators()
    signals = strategy.generate_signals(top_n=2)
    portfolio_value, strategy_returns, weights = strategy.backtest(
        signals, weighting_method="risk_parity"
    )
    assert len(portfolio_value.dropna()) > 20
    assert weights.shape[1] == len(SYMBOLS)
    assert strategy.performance_metrics
    assert "Sharpe Ratio" in strategy.performance_metrics
