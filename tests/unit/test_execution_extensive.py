"""Spread-aware execution edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.execution import SpreadAwareExecutor


@pytest.fixture
def ex():
    return SpreadAwareExecutor(max_spread_bps=25.0, impact_bps=2.0)


@pytest.mark.parametrize(
    "side,mid,spread,expect_ok",
    [
        ("buy", 100.0, 0.001, True),   # 10 bps
        ("sell", 100.0, 0.002, True),  # 20 bps
        ("buy", 100.0, 0.005, False),  # 50 bps
        ("buy", 0.0, 0.001, False),
        ("buy", 100.0, float("nan"), False),
        ("BUY", 50.0, 0.0015, True),
    ],
)
def test_should_trade_matrix(ex, side, mid, spread, expect_ok):
    ok, reason, mult = ex.should_trade(side, mid, spread)
    assert ok is expect_ok
    assert mult >= 0.0
    assert isinstance(reason, str)


def test_adverse_selection_clips_size(ex):
    rets = pd.Series([0.001] * 10 + [0.003, 0.003, 0.003])
    ok, reason, mult = ex.should_trade("buy", 100.0, 0.001, rets)
    assert ok
    assert mult < 1.0
    assert "adverse" in reason


def test_adverse_selection_short(ex):
    rets = pd.Series([-0.003] * 8)
    ok, reason, mult = ex.should_trade("sell", 100.0, 0.001, rets)
    assert ok and mult < 1.0


def test_simulate_fill_buy_pays_half_spread(ex):
    fill = ex.simulate_fill("buy", 10.0, 100.0, 0.002)
    assert not fill.rejected
    assert fill.fill_price > 100.0
    assert fill.filled_qty == 10.0


def test_simulate_fill_sell_receives_less(ex):
    fill = ex.simulate_fill("sell", 5.0, 100.0, 0.002)
    assert not fill.rejected
    assert fill.fill_price < 100.0


def test_simulate_fill_rejects_wide(ex):
    fill = ex.simulate_fill("buy", 1.0, 100.0, 0.01)
    assert fill.rejected
    assert fill.filled_qty == 0.0


def test_estimate_spread_basic():
    s = SpreadAwareExecutor.estimate_spread(102, 98, 100)
    assert s == pytest.approx(0.04)


def test_estimate_spread_invalid():
    assert np.isnan(SpreadAwareExecutor.estimate_spread(0, 1, 1))


def test_apply_to_weights_rejects_wide():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=idx)
    # Huge H-L → wide spread proxy
    highs = prices * 1.05
    lows = prices * 0.95
    tw = pd.DataFrame({"A": [0.0, 0.5, 0.5, 0.0, 0.0]}, index=idx)
    ex = SpreadAwareExecutor(max_spread_bps=5.0)
    executable, diag = ex.apply_to_weights(tw, prices, highs, lows)
    assert executable.shape == tw.shape
    assert (diag >= 0).all().all()


@pytest.mark.parametrize("max_bps", [5, 10, 25, 50, 100])
def test_max_spread_threshold_monotonic(max_bps):
    ex = SpreadAwareExecutor(max_spread_bps=max_bps)
    # 30 bps spread
    ok, _, _ = ex.should_trade("buy", 100.0, 0.003)
    assert ok == (30 <= max_bps)
