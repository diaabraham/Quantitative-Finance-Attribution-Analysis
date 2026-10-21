import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.providers import available_providers, build_provider, create_provider
from data.providers.errors import ProviderAuthError, ProviderEmptyDataError
from data.providers.yfinance_provider import YFinanceProvider
from data.providers.polygon_rest import PolygonProvider, MassiveProvider
from data.providers.stooq_provider import StooqProvider
from pricing.black_scholes import BlackScholes, OptionContract, implied_volatility
from strategies.execution import SpreadAwareExecutor
from strategies.ml_strategy import MLMomentumStrategy
from strategies.norberts_gambit import NorbertsGambit
from strategies.recursive import RecursiveStrategy
from strategies.swaps import CrossChainSwap, OptionsSwap
from strategies.swaps.options_swap import OptionsSwapSpec


def test_available_providers():
    names = available_providers()
    for required in ("yfinance", "stooq", "polygon", "massive", "databento"):
        assert required in names


def test_yfinance_live_small_window():
    p = YFinanceProvider()
    df = p.fetch_closes(["XLK"], "2024-01-02", "2024-01-15")
    assert not df.empty
    assert "XLK" in df.columns


def test_polygon_requires_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError):
        PolygonProvider()


def test_massive_requires_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError):
        MassiveProvider()


def test_databento_requires_key(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError):
        create_provider("databento")


def test_stooq_symbol_mapping():
    s = StooqProvider(api_key="dummy")
    assert s.normalize_symbol("XLK") == "xlk.us"
    assert s.normalize_symbol("^GSPC") == "^spx"


def test_polygon_symbol_mapping(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    p = PolygonProvider(api_key="test-key")
    assert p.normalize_symbol("^GSPC") == "I:SPX"
    assert p.normalize_symbol("XLK") == "XLK"


def test_polygon_fetch_mocked(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    p = PolygonProvider(api_key="test-key")
    payload = {
        "status": "OK",
        "results": [
            {"t": 1704067200000, "c": 100.0},
            {"t": 1704153600000, "c": 101.0},
        ],
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = payload
    with patch.object(p.session, "get", return_value=mock_resp):
        df = p.fetch_closes(["XLK"], "2024-01-01", "2024-01-05")
    assert list(df.columns) == ["XLK"]
    assert len(df) == 2


def test_black_scholes_call_atm():
    bs = BlackScholes(
        OptionContract(100, 100, 0.05, 0.0, 0.2, 1.0, "call")
    )
    price = bs.price()
    assert 5 < price < 15
    g = bs.greeks()
    assert 0.4 < g["delta"] < 0.7
    iv = implied_volatility(price, 100, 100, 0.05, 0.0, 1.0, "call")
    assert iv is not None and abs(iv - 0.2) < 0.01


def test_spread_aware_rejects_wide():
    ex = SpreadAwareExecutor(max_spread_bps=10)
    ok, reason, _ = ex.should_trade("buy", 100.0, 0.05)  # 500 bps
    assert not ok
    assert "spread_too_wide" in reason


def test_recursive_strategy_weights():
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    prices = pd.DataFrame(
        {
            "A": range(100, 180),
            "B": range(180, 100, -1),
            "C": [100 + (i % 7) for i in range(80)],
        },
        index=idx,
    ).astype(float)
    w = RecursiveStrategy.multi_horizon().generate_weights(prices)
    assert w.shape == prices.shape
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all()


def test_ml_strategy_runs():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = pd.Series(range(100), index=idx).astype(float)
    prices = pd.DataFrame({"A": rng, "B": rng[::-1].values, "C": rng * 1.01})
    w = MLMomentumStrategy(min_train=30, train_window=50, top_n=1).generate_weights(prices)
    assert w.shape == prices.shape


def test_norberts_gambit_synthetic():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    usd = pd.Series(100.0, index=idx)
    fx = pd.Series(1.35, index=idx)
    cad = usd * fx
    res = NorbertsGambit(journal_days=2).convert_cad_to_usd(
        cad, usd, 1 / fx, notional_cad=10000
    )
    assert not res.aborted
    assert res.usd_received > 0
    assert res.savings_vs_retail > 0


def test_cross_chain_and_options_swap():
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    mid = pd.Series(1.0, index=idx)
    xc = CrossChainSwap().backtest_path(mid, qty_src=100)
    assert len(xc) == 20
    o = OptionsSwap(
        OptionsSwapSpec(notional=1e5, fixed_rate=0.2, strike=100.0)
    )
    spots = pd.Series(100.0, index=idx)
    vols = pd.Series(0.25, index=idx)
    mtm = o.path_mtm(spots, vols)
    assert "mtm" in mtm.columns


def test_build_provider_fallback_skips_unauthed(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    p = build_provider("yfinance", fallbacks=["polygon", "databento"])
    # polygon/databento skipped due to missing keys → plain yfinance or fallback chain without them
    assert "yfinance" in p.name


def test_high_probability_and_credit_spread_smoke():
    from strategies.high_probability import (
        HighProbabilityMeanReversion,
        OptionsCreditPutPath,
        summarize_trades,
    )

    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    rng = np.linspace(100, 110, 120) + np.sin(np.linspace(0, 12, 120)) * 3
    ohlc = pd.DataFrame(
        {
            "Open": rng,
            "High": rng + 1.5,
            "Low": rng - 1.5,
            "Close": rng,
        },
        index=idx,
    )
    # Inject a sharp drop + bounce so mean-reversion can fire
    ohlc.iloc[80, ohlc.columns.get_loc("Close")] *= 0.93
    ohlc.iloc[80, ohlc.columns.get_loc("Low")] *= 0.92
    eq = HighProbabilityMeanReversion(z_entry=1.5, max_spread_bps=50)
    opt = OptionsCreditPutPath(hold_days=5, rsi_entry=50)
    t1 = eq.run(ohlc, "TEST", "stocks", "1y")
    t2 = opt.run(ohlc, "TEST", "1y")
    assert isinstance(summarize_trades(t1 + t2).n_trades, int)
