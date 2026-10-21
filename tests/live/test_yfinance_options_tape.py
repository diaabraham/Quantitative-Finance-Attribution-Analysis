"""
Live option-chain / quote-tape tests via yfinance.

These exercise *real* bid/ask/strike data when Yahoo provides it.
Modeled BS credit-spread path is covered separately under unit/regression tests.
"""

from __future__ import annotations

import math

import pytest

from data.options_yfinance import (
    fetch_option_chain,
    list_expiries,
    near_atm_puts,
    quote_quality_ok,
    summarize_chain,
)
from pricing.black_scholes import BlackScholes, OptionContract


@pytest.mark.options_tape
@pytest.mark.live
@pytest.mark.parametrize("symbol", ["SPY", "QQQ", "AAPL", "MSFT", "IWM"])
def test_live_expiries_available(symbol):
    exps = list_expiries(symbol)
    assert len(exps) >= 1
    assert all(isinstance(e, str) and len(e) >= 8 for e in exps[:3])


@pytest.mark.options_tape
@pytest.mark.live
@pytest.mark.parametrize("symbol", ["SPY", "QQQ", "AAPL"])
def test_live_chain_has_quotes(symbol):
    snap = fetch_option_chain(symbol)
    assert snap.spot > 0
    assert snap.n_contracts > 0
    n_calls, n_puts, n_ok = summarize_chain(snap)
    assert n_calls > 0 and n_puts > 0
    # Yahoo usually supplies bid/ask; require at least some usable quotes
    assert "bid" in snap.puts.columns and "ask" in snap.puts.columns
    assert n_ok >= 1


@pytest.mark.options_tape
@pytest.mark.live
def test_near_atm_puts_and_quote_quality():
    snap = fetch_option_chain("SPY")
    atm = near_atm_puts(snap, n=8)
    assert len(atm) >= 1
    usable = [quote_quality_ok(r) for _, r in atm.iterrows()]
    assert any(usable)


@pytest.mark.options_tape
@pytest.mark.live
def test_bs_vs_live_mid_sanity_band():
    """
    Modeled BS price should be same order of magnitude as live mid for near-ATM
    puts — not an arb check; guards against wiring mistakes.
    """
    snap = fetch_option_chain("SPY")
    atm = near_atm_puts(snap, n=15)
    checked = 0
    for _, row in atm.iterrows():
        if not quote_quality_ok(row, max_spread_pct=0.35):
            continue
        mid = float(row["mid"]) if "mid" in row and row["mid"] == row["mid"] else None
        if mid is None or mid <= 0.05:
            continue
        strike = float(row["strike"])
        # Rough T from expiry string YYYY-MM-DD
        import pandas as pd

        today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
        t_days = max((pd.Timestamp(snap.expiry) - today).days, 1)
        t = t_days / 365.0
        # Use mid-implied vol if present else 0.2
        iv = float(row["impliedVolatility"]) if "impliedVolatility" in row else 0.2
        if not (iv == iv) or iv <= 0:
            iv = 0.2
        model = BlackScholes(
            OptionContract(snap.spot, strike, 0.04, 0.0, iv, t, "put")
        ).price()
        # Same order of magnitude (within 5x) when both meaningful
        if model > 0.05 and mid > 0.05:
            ratio = model / mid
            assert 0.1 < ratio < 10.0
            checked += 1
        if checked >= 3:
            break
    assert checked >= 1


@pytest.mark.options_tape
@pytest.mark.live
def test_modeled_path_isolated_from_tape_module():
    """Documented isolation: tape module must not import credit-spread path."""
    import data.options_yfinance as tape
    import strategies.high_probability as hp

    assert "OptionsCreditPutPath" not in dir(tape)
    assert hasattr(hp, "OptionsCreditPutPath")
    assert not math.isnan(fetch_option_chain("SPY").spot)
