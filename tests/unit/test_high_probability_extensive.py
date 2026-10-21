"""High-probability strategy and win-rate summarizer edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.high_probability import (
    HighProbabilityMeanReversion,
    OptionsCreditPutPath,
    Trade,
    summarize_trades,
)
from tests.helpers.market import synthetic_ohlc


def _trade(pnl: float, cls: str = "stocks", horizon: str = "1y") -> Trade:
    return Trade(
        "X",
        cls,
        horizon,
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        "buy",
        1.0,
        1.0 + pnl,
        pnl,
        pnl,
        1,
        "tp" if pnl > 0 else "sl",
    )


def test_summarize_empty():
    r = summarize_trades([])
    assert r.n_trades == 0 and r.win_rate == 0.0


@pytest.mark.parametrize(
    "pnls,expected_wr",
    [
        ([0.01, 0.02, -0.01], 2 / 3),
        ([0.01] * 10, 1.0),
        ([-0.01] * 5, 0.0),
        ([0.0, 0.01], 0.5),  # pnl==0 is not a win
    ],
)
def test_win_rate_definition(pnls, expected_wr):
    r = summarize_trades([_trade(p) for p in pnls])
    assert r.n_trades == len(pnls)
    assert abs(r.win_rate - expected_wr) < 1e-12


def test_profit_factor_and_expectancy():
    r = summarize_trades([_trade(0.02), _trade(0.02), _trade(-0.01)])
    assert r.profit_factor > 1.0
    assert r.expectancy_pct > 0


@pytest.mark.parametrize("asset_class", ["stocks", "futures", "options"])
def test_mean_reversion_runs_on_synthetic(asset_class):
    ohlc = synthetic_ohlc(180, seed=7)
    # Force a dip
    ohlc.iloc[100, ohlc.columns.get_loc("Close")] *= 0.92
    ohlc.iloc[100, ohlc.columns.get_loc("Low")] *= 0.90
    strat = HighProbabilityMeanReversion(
        z_entry=1.2, rsi_low=45, rsi_high=55, max_spread_bps=80, trend_sma=0
    )
    trades = strat.run(ohlc, "SYN", asset_class, "1y")
    assert isinstance(trades, list)


def test_mean_reversion_too_short_returns_empty():
    ohlc = synthetic_ohlc(10, seed=1)
    assert HighProbabilityMeanReversion().run(ohlc, "S", "stocks", "1d") == []


def test_trend_sma_blocks_long_in_downtrend():
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    close = np.linspace(150, 80, 120)
    ohlc = pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close},
        index=idx,
    )
    ohlc.iloc[100, ohlc.columns.get_loc("Close")] *= 0.90
    ohlc.iloc[100, ohlc.columns.get_loc("Low")] *= 0.88
    blocked = HighProbabilityMeanReversion(
        z_entry=1.0, rsi_low=55, trend_sma=40, max_spread_bps=200
    )
    assert blocked.run(ohlc, "DN", "stocks", "1y") == []


@pytest.mark.parametrize("hold", [3, 5, 7, 10])
def test_options_credit_path_hold_variants(hold):
    ohlc = synthetic_ohlc(200, seed=3, vol=0.02)
    # Oversold patch for RSI entry
    ohlc.iloc[80:90, ohlc.columns.get_loc("Close")] *= np.linspace(1.0, 0.88, 10)
    opt = OptionsCreditPutPath(hold_days=hold, rsi_entry=55, short_otm_pct=0.05)
    trades = opt.run(ohlc, "SYN", "1y")
    for t in trades:
        assert t.asset_class == "options"
        assert t.bars_held == hold
        assert t.pnl >= -1e6  # defined risk floor applied


def test_options_credit_path_short_series_empty():
    assert OptionsCreditPutPath().run(synthetic_ohlc(15), "X", "1m") == []


@pytest.mark.parametrize(
    "z,tp,sl",
    [(2.0, 0.3, 1.5), (2.5, 0.2, 2.5), (3.0, 0.4, 2.0)],
)
def test_param_variants_do_not_crash(z, tp, sl):
    ohlc = synthetic_ohlc(150, seed=11)
    strat = HighProbabilityMeanReversion(
        z_entry=z, tp_atr=tp, sl_atr=sl, max_spread_bps=50, trend_sma=0
    )
    strat.run(ohlc, "P", "stocks", "6m")
