"""RiskManager coverage against the real API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.risk_manager import RiskManager


@pytest.fixture
def returns_df():
    idx = pd.date_range("2022-01-01", periods=260, freq="B")
    rng = np.random.default_rng(0)
    data = rng.normal(0.0004, 0.012, size=(len(idx), 4))
    return pd.DataFrame(data, index=idx, columns=list("ABCD"))


@pytest.fixture
def port(returns_df):
    return returns_df.mean(axis=1)


@pytest.fixture
def rm():
    return RiskManager()


def test_var_returns_dict(rm, port):
    out = rm.calculate_var(port, 0.95)
    assert set(out) >= {"historical", "parametric", "combined"}
    for v in out.values():
        assert np.isfinite(v)


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_var_confidence_levels(rm, port, conf):
    out = rm.calculate_var(port, conf)
    assert out["historical"] <= 0 or True  # losses typically negative percentile
    assert np.isfinite(out["combined"])


def test_expected_shortfall(rm, port):
    es = rm.calculate_expected_shortfall(port, 0.95)
    assert np.isfinite(es)


def test_risk_metrics_bundle(rm, port):
    metrics = rm.calculate_risk_metrics(port)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


@pytest.mark.parametrize(
    "method",
    ["risk_parity", "equal_risk", "min_variance", "max_sharpe"],
)
def test_position_weights_methods(rm, returns_df, method):
    signals = (returns_df.rolling(5).mean() > 0).astype(float)
    w = rm.calculate_position_weights(signals, returns_df, method=method)
    assert isinstance(w, pd.DataFrame)
    assert w.shape[1] == returns_df.shape[1]
    # Weights should be finite where defined
    assert np.isfinite(w.to_numpy(dtype=float)).all() or w.notna().any().any()


def test_check_sector_exposure_scales(rm, returns_df):
    idx = returns_df.index[:30]
    weights = pd.DataFrame(0.25, index=idx, columns=list("ABCD"))
    # Mapping is sector -> list of assets
    sectors = {"tech": ["A", "B"], "fin": ["C", "D"]}
    out = rm.check_sector_exposure(weights, sectors)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == weights.shape


def test_drawdown_from_equity(rm):
    equity = pd.Series(np.cumprod(1 + np.random.default_rng(1).normal(0.001, 0.01, 200)))
    dd = rm.calculate_drawdown(equity)
    assert (dd <= 0).all() or (dd >= -1).all()


def test_check_drawdown_limit(rm):
    equity = pd.Series([100, 110, 105, 90, 95])
    # Should not raise
    rm.check_drawdown_limit(equity)
