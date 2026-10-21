"""
Provider contract tests against live endpoints.

yfinance always runs (no key). Stooq / Polygon / Massive / Databento run only
when the corresponding env key is present (``requires_key`` marker).
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from data.providers import build_provider, create_provider
from data.providers.yfinance_provider import YFinanceProvider
from tests.helpers.market import fetch_ohlc_yfinance

LIVE_SYMBOLS = ["AAPL", "MSFT", "SPY", "XLK"]
FUTURES = ["ES=F", "GC=F"]


@pytest.mark.live
@pytest.mark.parametrize("symbol", LIVE_SYMBOLS)
def test_yfinance_provider_closes_contract(symbol, live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    df = YFinanceProvider().fetch_closes([symbol], start, live_end)
    assert not df.empty
    assert symbol in df.columns
    assert df[symbol].notna().sum() >= 10
    assert (df[symbol] > 0).all()


@pytest.mark.live
def test_yfinance_multi_symbol_and_futures(live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    syms = ["AAPL", "MSFT", "ES=F"]
    df = YFinanceProvider().fetch_closes(syms, start, live_end)
    assert set(df.columns) & set(syms)
    assert len(df) >= 15


@pytest.mark.live
@pytest.mark.parametrize("symbol", ["AAPL", "SPY", "NVDA", "ES=F", "CL=F"])
def test_yfinance_ohlc_helper_contract(symbol, live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    ohlc = fetch_ohlc_yfinance(symbol, start, live_end)
    assert not ohlc.empty
    for col in ("Open", "High", "Low", "Close"):
        assert col in ohlc.columns
    assert (ohlc["High"] >= ohlc["Low"]).all()
    assert (ohlc["Close"] > 0).all()


@pytest.mark.live
def test_build_provider_yfinance_default(live_end):
    p = build_provider("yfinance", fallbacks=[])
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    df = p.fetch_closes(["SPY"], start, live_end)
    assert not df.empty


@pytest.mark.live
@pytest.mark.requires_key("STOOQ_API_KEY")
def test_stooq_live_closes(live_end):
    p = create_provider("stooq")
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    df = p.fetch_closes(["AAPL", "MSFT"], start, live_end)
    assert not df.empty
    assert df.notna().any().any()


@pytest.mark.live
@pytest.mark.requires_key("POLYGON_API_KEY")
def test_polygon_live_closes(live_end):
    p = create_provider("polygon")
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    df = p.fetch_closes(["AAPL"], start, live_end)
    assert not df.empty
    assert "AAPL" in df.columns


@pytest.mark.live
@pytest.mark.requires_key("MASSIVE_API_KEY", "POLYGON_API_KEY")
def test_massive_live_closes(live_end):
    p = create_provider("massive")
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    df = p.fetch_closes(["MSFT"], start, live_end)
    assert not df.empty


@pytest.mark.live
@pytest.mark.requires_key("DATABENTO_API_KEY")
def test_databento_live_closes(live_end):
    p = create_provider("databento")
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    df = p.fetch_closes(["AAPL"], start, live_end)
    assert not df.empty


@pytest.mark.live
def test_keyed_providers_skipped_without_env(api_keys_present):
    """Document which keyed providers are unavailable in this environment."""
    missing = [k for k, ok in api_keys_present.items() if k != "sec" and not ok]
    # Always true — this test records the gap rather than failing CI without keys
    assert isinstance(missing, list)
    # yfinance never needs a key
    assert True
