"""
Integration: strategies + providers on live yfinance OHLC / closes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.data_manager import DataManager
from data.providers.yfinance_provider import YFinanceProvider
from strategies.high_probability import (
    HighProbabilityMeanReversion,
    OptionsCreditPutPath,
    summarize_trades,
)
from strategies.recursive import RecursiveStrategy
from tests.helpers.market import fetch_ohlc_yfinance


@pytest.mark.live
def test_data_manager_live_sector_slice(live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    dm = DataManager(provider="yfinance")
    px = dm.fetch_historical_data(["XLK", "XLF", "XLV"], start, live_end)
    assert not px.empty
    assert px.shape[1] >= 2
    assert len(px) >= 40


@pytest.mark.live
def test_mean_reversion_on_live_equities(live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    strat = HighProbabilityMeanReversion(
        z_entry=2.0, max_spread_bps=40, trend_sma=0, rsi_low=30, rsi_high=70
    )
    all_trades = []
    for sym in ("AAPL", "MSFT", "JPM", "XOM"):
        ohlc = fetch_ohlc_yfinance(sym, start, live_end)
        assert not ohlc.empty
        all_trades.extend(strat.run(ohlc, sym, "stocks", "1y"))
    # May be selective; ensure API works and any trades have sane fields
    for t in all_trades:
        assert t.asset_class == "stocks"
        assert t.entry_price > 0 and t.exit_price > 0


@pytest.mark.live
def test_mean_reversion_on_live_futures(live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    strat = HighProbabilityMeanReversion(
        z_entry=2.0, max_spread_bps=40, trend_sma=0, rsi_low=30, rsi_high=70
    )
    trades = []
    for sym in ("ES=F", "GC=F", "CL=F"):
        ohlc = fetch_ohlc_yfinance(sym, start, live_end)
        if ohlc.empty:
            continue
        trades.extend(strat.run(ohlc, sym, "futures", "1y"))
    for t in trades:
        assert t.asset_class == "futures"


@pytest.mark.live
def test_modeled_credit_spread_on_live_underlyings(live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    opt = OptionsCreditPutPath(hold_days=7, rsi_entry=40, short_otm_pct=0.06)
    trades = []
    for sym in ("SPY", "QQQ", "IWM"):
        ohlc = fetch_ohlc_yfinance(sym, start, live_end)
        assert not ohlc.empty
        trades.extend(opt.run(ohlc, sym, "1y"))
    assert len(trades) >= 1
    summary = summarize_trades(trades)
    assert 0.0 <= summary.win_rate <= 1.0
    assert summary.n_trades == len(trades)


@pytest.mark.live
def test_recursive_strategy_on_live_etfs(live_end):
    start = (pd.Timestamp(live_end) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    px = YFinanceProvider().fetch_closes(
        ["XLK", "XLF", "XLV", "XLY", "XLE"], start, live_end
    )
    assert px.shape[1] >= 3
    w = RecursiveStrategy.multi_horizon().generate_weights(px)
    assert w.shape == px.shape
    assert (w.sum(axis=1) <= 1.0 + 1e-5).all()
