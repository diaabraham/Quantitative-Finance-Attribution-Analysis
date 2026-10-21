"""Ensure position weighting returns a dated weight matrix for backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk.risk_manager import RiskManager


def test_calculate_position_weights_returns_dataframe():
    idx = pd.bdate_range("2024-01-02", periods=60)
    cols = ["XLK", "XLF", "XLV"]
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, size=(len(idx), 3)), index=idx, columns=cols)
    signals = pd.DataFrame(0, index=idx, columns=cols)
    signals.iloc[20:, 0] = 1
    signals.iloc[20:, 1] = 1

    rm = RiskManager(max_position_size=0.5, max_sector_exposure=0.6, lookback_period=40)
    weights = rm.calculate_position_weights(signals, returns, method="risk_parity")

    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == signals.shape
    assert (weights.iloc[:20].sum(axis=1) == 0).all()
    active = weights.iloc[25:]
    assert np.allclose(active.sum(axis=1), 1.0, atol=1e-6)
    assert (active[["XLK", "XLF"]].sum(axis=1) > 0.99).all()
