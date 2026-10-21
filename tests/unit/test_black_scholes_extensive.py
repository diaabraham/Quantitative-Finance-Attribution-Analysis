"""Extensive Black–Scholes unit coverage (no network)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricing.black_scholes import BlackScholes, OptionContract, implied_volatility


@pytest.mark.parametrize(
    "spot,strike,rate,q,vol,t,opt,lo,hi",
    [
        (100, 100, 0.05, 0.0, 0.20, 1.0, "call", 5.0, 15.0),
        (100, 100, 0.05, 0.0, 0.20, 1.0, "put", 5.0, 15.0),
        (100, 90, 0.01, 0.0, 0.25, 0.5, "call", 8.0, 25.0),
        (100, 110, 0.01, 0.0, 0.25, 0.5, "put", 8.0, 25.0),
        (50, 50, 0.0, 0.0, 0.30, 0.25, "call", 1.0, 8.0),
        (200, 180, 0.03, 0.01, 0.15, 2.0, "call", 20.0, 60.0),
    ],
)
def test_price_bands(spot, strike, rate, q, vol, t, opt, lo, hi):
    px = BlackScholes(OptionContract(spot, strike, rate, q, vol, t, opt)).price()
    assert lo < px < hi


@pytest.mark.parametrize("opt", ["call", "put"])
def test_put_call_parity(opt):
    spot, k, r, q, vol, t = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0
    c = BlackScholes(OptionContract(spot, k, r, q, vol, t, "call")).price()
    p = BlackScholes(OptionContract(spot, k, r, q, vol, t, "put")).price()
    lhs = c - p
    rhs = spot * math.exp(-q * t) - k * math.exp(-r * t)
    assert abs(lhs - rhs) < 1e-6


@pytest.mark.parametrize("opt", ["call", "put"])
def test_expiry_intrinsic(opt):
    spot, k = 105.0, 100.0
    px = BlackScholes(OptionContract(spot, k, 0.01, 0.0, 0.2, 0.0, opt)).price()
    if opt == "call":
        assert abs(px - 5.0) < 1e-9
    else:
        assert abs(px - 0.0) < 1e-9


@pytest.mark.parametrize("vol", [0.05, 0.1, 0.2, 0.4, 0.8])
def test_iv_roundtrip_call(vol):
    spot, k, r, q, t = 100.0, 100.0, 0.03, 0.0, 0.75
    px = BlackScholes(OptionContract(spot, k, r, q, vol, t, "call")).price()
    iv = implied_volatility(px, spot, k, r, q, t, "call")
    assert iv is not None
    assert abs(iv - vol) < 1e-3


@pytest.mark.parametrize("moneyness", [0.8, 0.9, 1.0, 1.1, 1.2])
def test_delta_bounds_call(moneyness):
    spot = 100.0
    k = spot / moneyness
    g = BlackScholes(OptionContract(spot, k, 0.02, 0.0, 0.25, 0.5, "call")).greeks()
    assert 0.0 <= g["delta"] <= 1.0
    assert g["gamma"] >= 0.0
    assert g["vega"] >= 0.0


@pytest.mark.parametrize("bad", [
    dict(spot=0, strike=100, rate=0.01, dividend_yield=0, volatility=0.2, time_to_expiry=1),
    dict(spot=100, strike=-1, rate=0.01, dividend_yield=0, volatility=0.2, time_to_expiry=1),
    dict(spot=100, strike=100, rate=0.01, dividend_yield=0, volatility=-0.1, time_to_expiry=1),
    dict(spot=100, strike=100, rate=0.01, dividend_yield=0, volatility=0.2, time_to_expiry=-0.1),
])
def test_invalid_contracts_raise(bad):
    with pytest.raises(ValueError):
        BlackScholes(OptionContract(**bad, option_type="call"))


def test_higher_vol_higher_price():
    low = BlackScholes(OptionContract(100, 100, 0.01, 0.0, 0.1, 1.0, "call")).price()
    high = BlackScholes(OptionContract(100, 100, 0.01, 0.0, 0.4, 1.0, "call")).price()
    assert high > low


def test_iv_fails_gracefully_on_impossible_price():
    iv = implied_volatility(1e6, 100, 100, 0.01, 0.0, 0.1, "call")
    # May return None or a huge vol depending on solver — must not crash
    assert iv is None or iv > 0
