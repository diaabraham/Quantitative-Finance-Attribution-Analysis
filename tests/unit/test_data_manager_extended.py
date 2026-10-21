"""Additional DataManager coverage (provider wiring, caching, sectors)."""

from __future__ import annotations

import pandas as pd
import pytest

from data.data_manager import DataManager


def test_sector_etf_list_nonempty():
    dm = DataManager(provider="yfinance")
    etfs = dm.get_sector_etfs()
    assert isinstance(etfs, (list, tuple, dict))
    assert len(etfs) >= 5


@pytest.mark.live
def test_fetch_historical_live_and_cache(tmp_path, live_end):
    cache = tmp_path / "cache"
    cache.mkdir()
    dm = DataManager(provider="yfinance", cache_dir=str(cache))
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    a = dm.fetch_historical_data(["AAPL"], start, live_end, use_cache=True)
    b = dm.fetch_historical_data(["AAPL"], start, live_end, use_cache=True)
    assert not a.empty and not b.empty
    assert list(a.columns) == list(b.columns)


@pytest.mark.live
def test_risk_free_rate_available():
    dm = DataManager(provider="yfinance")
    r = dm.get_risk_free_rate()
    assert isinstance(r, (int, float)) and -0.05 < r < 0.25
