"""Recursive, ML, swaps, Norbert’s Gambit unit coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.ml_strategy import MLMomentumStrategy
from strategies.norberts_gambit import NorbertsGambit
from strategies.recursive import RecursiveStrategy, default_momentum_signals
from strategies.swaps import CrossChainSwap, OptionsSwap
from strategies.swaps.options_swap import OptionsSwapSpec


@pytest.fixture
def prices():
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    rng = np.random.default_rng(42)
    cols = {}
    for name in list("ABCDEFGH"):
        rets = rng.normal(0.0005, 0.015, size=len(idx))
        cols[name] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(cols, index=idx)


def test_default_momentum_signals_shape(prices):
    sig = default_momentum_signals(prices, top_n=3)
    assert sig.shape == prices.shape
    sums = sig.sum(axis=1)
    assert (sums >= -1e-9).all()
    assert (sums <= 1.0 + 1e-6).all()


@pytest.mark.parametrize("top_n", [1, 2, 3, 4])
def test_recursive_multi_horizon(prices, top_n):
    w = RecursiveStrategy.multi_horizon(top_n=top_n).generate_weights(prices)
    assert w.shape == prices.shape
    assert (w.sum(axis=1) <= 1.0 + 1e-5).all()


def test_recursive_emptyish():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": range(5)}, index=idx).astype(float)
    w = RecursiveStrategy.multi_horizon().generate_weights(prices)
    assert w.shape[0] == 5


@pytest.mark.parametrize("top_n", [1, 2])
def test_ml_momentum_variants(prices, top_n):
    w = MLMomentumStrategy(min_train=40, train_window=60, top_n=top_n).generate_weights(
        prices.iloc[:, :4]
    )
    assert w.shape[1] == 4


@pytest.mark.parametrize("journal_days", [1, 2, 5])
def test_norberts_gambit_journal(journal_days):
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    usd = pd.Series(np.linspace(100, 110, 40), index=idx)
    fx = pd.Series(1.35 + 0.01 * np.sin(np.linspace(0, 3, 40)), index=idx)
    cad = usd * fx
    res = NorbertsGambit(journal_days=journal_days).convert_cad_to_usd(
        cad, usd, 1 / fx, notional_cad=25000
    )
    assert not res.aborted
    assert res.usd_received > 0


@pytest.mark.parametrize("qty", [10, 100, 1000])
def test_cross_chain_path(qty):
    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    mid = pd.Series(np.linspace(1.0, 1.05, 25), index=idx)
    path = CrossChainSwap().backtest_path(mid, qty_src=qty)
    assert len(path) == 25


@pytest.mark.parametrize("vol", [0.15, 0.25, 0.4])
def test_options_swap_mtm(vol):
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    spots = pd.Series(100.0, index=idx)
    vols = pd.Series(vol, index=idx)
    o = OptionsSwap(OptionsSwapSpec(notional=1e5, fixed_rate=0.2, strike=100.0))
    mtm = o.path_mtm(spots, vols)
    assert "mtm" in mtm.columns
    assert len(mtm) == 30
