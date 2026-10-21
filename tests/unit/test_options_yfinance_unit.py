"""Unit tests for options tape helpers (mocked; no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.options_yfinance import (
    OptionChainSnapshot,
    near_atm_puts,
    quote_quality_ok,
    summarize_chain,
)


def test_quote_quality_matrix():
    assert quote_quality_ok(pd.Series({"bid": 1.0, "ask": 1.1}))
    assert not quote_quality_ok(pd.Series({"bid": 1.0, "ask": 3.0}))  # wide
    assert not quote_quality_ok(pd.Series({"bid": -0.1, "ask": 1.0}))
    assert not quote_quality_ok(pd.Series({"bid": 2.0, "ask": 1.0}))  # inverted


def test_near_atm_and_summarize():
    calls = pd.DataFrame({"strike": [100], "bid": [1], "ask": [1.1]})
    puts = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "bid": [2.0, 1.0, 0.5],
            "ask": [2.2, 1.1, 0.6],
        }
    )
    snap = OptionChainSnapshot("SPY", "2026-12-18", 100.0, calls, puts)
    atm = near_atm_puts(snap, n=2)
    assert len(atm) == 2
    assert atm.iloc[0]["strike"] == 100
    n_c, n_p, n_ok = summarize_chain(snap)
    assert (n_c, n_p) == (1, 3)
    assert n_ok >= 1


@patch("data.options_yfinance.yf.Ticker")
def test_fetch_option_chain_mocked(mock_ticker):
    from data.options_yfinance import fetch_option_chain

    t = MagicMock()
    t.options = ["2026-12-18"]
    t.fast_info = MagicMock(last_price=500.0)
    calls = pd.DataFrame({"strike": [500], "bid": [5], "ask": [5.5]})
    puts = pd.DataFrame({"strike": [495], "bid": [4], "ask": [4.4]})
    chain = MagicMock()
    chain.calls = calls
    chain.puts = puts
    t.option_chain.return_value = chain
    mock_ticker.return_value = t
    snap = fetch_option_chain("SPY")
    assert snap.spot == 500.0
    assert "mid" in snap.puts.columns
