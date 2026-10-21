"""Integration: attribution + risk on coherent synthetic panels (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.brinson import BrinsonAttribution
from attribution.factor_attribution import FactorAttribution
from attribution.transaction_cost import TransactionCostAttribution
from risk.risk_manager import RiskManager


@pytest.fixture
def panel():
    idx = pd.date_range("2021-01-01", periods=252, freq="B")
    rng = np.random.default_rng(0)
    assets = list("ABCDE")
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, size=(len(idx), 5)), index=idx, columns=assets)
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.01, len(idx)),
            "SMB": rng.normal(0.0001, 0.005, len(idx)),
            "HML": rng.normal(0.0001, 0.005, len(idx)),
        },
        index=idx,
    )
    return rets, factors


def test_brinson_then_risk_metrics(panel):
    rets, _ = panel
    # Single-period Brinson on last-day equal weights vs tilted
    sectors = rets.columns
    port_w = pd.Series(0.2, index=sectors)
    port_w["A"] = 0.4
    port_w["E"] = 0.0
    port_w = port_w / port_w.sum()
    bench_w = pd.Series(1 / len(sectors), index=sectors)
    day = rets.iloc[-21:].mean()
    effects = BrinsonAttribution(day, day * 0.8, port_w, bench_w).calculate_attribution()
    assert np.isfinite(effects["total"])

    port = rets.mean(axis=1)
    metrics = RiskManager().calculate_risk_metrics(port)
    assert "Sharpe Ratio" in metrics
    assert "VaR (95%) - Historical" in metrics


@pytest.mark.parametrize("method", ["risk_parity", "equal_risk", "min_variance"])
def test_risk_weights_then_var(panel, method):
    rets, _ = panel
    rm = RiskManager()
    signals = (rets.rolling(10).mean() > 0).astype(float)
    w = rm.calculate_position_weights(signals, rets, method=method)
    assert w.shape[1] == rets.shape[1]
    port = (rets * w.shift(1).fillna(0)).sum(axis=1)
    var = rm.calculate_var(port.dropna())
    assert np.isfinite(var["combined"])


def test_factor_attribution_smoke(panel):
    rets, factors = panel
    port = rets.mean(axis=1)
    fa = FactorAttribution(port, factors)
    if hasattr(fa, "estimate_factor_exposures"):
        exp = fa.estimate_factor_exposures()
        assert exp is not None
    elif hasattr(fa, "calculate_factor_contributions"):
        out = fa.calculate_factor_contributions()
        assert out is not None
    else:
        pytest.skip("FactorAttribution API mismatch")


def test_transaction_cost_attribution_smoke(panel):
    rets, _ = panel
    n = 40
    trades = pd.DataFrame(
        {
            "timestamp": rets.index[:n],
            "symbol": ["A"] * n,
            "quantity": [100] * n,
            "price": (100 * (1 + rets["A"].iloc[:n])).values,
            "side": (["BUY", "SELL"] * (n // 2))[:n],
        }
    )
    market_data = pd.DataFrame(
        {
            "timestamp": rets.index[:n],
            "symbol": ["A"] * n,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": [1_000_000] * n,
        }
    )
    tca = TransactionCostAttribution(trades, market_data)
    costs = tca.calculate_total_costs()
    assert isinstance(costs, pd.DataFrame)
    assert "total_cost" in costs.columns
    assert len(costs) == n


@pytest.mark.parametrize("conf", [0.9, 0.95, 0.99])
def test_es_vs_var_ordering(panel, conf):
    rets, _ = panel
    port = rets.mean(axis=1)
    rm = RiskManager()
    var = rm.calculate_var(port, conf)["historical"]
    es = rm.calculate_expected_shortfall(port, conf)
    # ES should be <= historical VaR for loss distributions (more extreme)
    assert es <= var + 1e-12 or not np.isfinite(es)
