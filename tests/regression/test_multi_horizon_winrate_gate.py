"""
Automated per-horizon win-rate gate (≥87% on each of 1d…5y).

Uses **live yfinance** OHLC (not synthetic stubs). Options path remains the
documented modeled bull-put credit spread on real underlyings; equity/futures
use real daily bars.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_multi_horizon_backtest import (
    FUTURES,
    HORIZONS_DAYS,
    OPTIONS_UNDERLYINGS,
    STOCKS,
    default_equity_params,
    default_options_params,
    fetch_ohlc,
    run_horizon,
)
from strategies.high_probability import summarize_trades

TARGET = 0.87
MIN_TRADES_TOTAL = 40
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "multi_horizon_backtest.json"


@pytest.fixture(scope="module")
def live_backtest():
    end = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    start = (end - pd.Timedelta(days=365 * 5 + 90)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    prices = {}
    errors = {}
    for sym in sorted(set(STOCKS + FUTURES + OPTIONS_UNDERLYINGS)):
        try:
            df = fetch_ohlc(sym, start, end_s)
            if df.empty:
                errors[sym] = "empty"
            else:
                prices[sym] = df
        except Exception as exc:  # noqa: BLE001
            errors[sym] = str(exc)[:160]
    if len(prices) < 10:
        pytest.fail(f"Insufficient live symbols fetched: {len(prices)} errors={errors}")

    params = default_equity_params()
    per = {}
    all_trades = []
    for h in HORIZONS_DAYS:
        trades, meta = run_horizon(h, end, prices, params, opt_hold=7)
        all_trades.extend(trades)
        per[h] = meta
    total = summarize_trades(all_trades)
    return {
        "end": end,
        "prices": prices,
        "errors": errors,
        "per_horizon": per,
        "all_trades": all_trades,
        "total": total,
        "params": params,
    }


@pytest.mark.live
@pytest.mark.slow
def test_each_horizon_meets_win_rate_target(live_backtest):
    for h, meta in live_backtest["per_horizon"].items():
        assert meta["n_trades"] > 0, f"{h}: expected trades from live data"
        assert meta["win_rate"] >= TARGET, (
            f"{h}: win_rate={meta['win_rate']:.2%} < {TARGET:.0%} "
            f"(n={meta['n_trades']})"
        )
    total = live_backtest["total"]
    assert total.n_trades >= MIN_TRADES_TOTAL
    assert total.win_rate >= TARGET


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.parametrize("horizon", list(HORIZONS_DAYS))
def test_horizon_breakdown_and_threshold(live_backtest, horizon):
    meta = live_backtest["per_horizon"][horizon]
    assert "by_asset_class" in meta
    for cls in ("stocks", "futures", "options"):
        assert cls in meta["by_asset_class"]
    assert meta["win_rate"] >= TARGET
    if horizon in ("3y", "5y"):
        assert meta["by_asset_class"]["options"]["n_trades"] >= 1


@pytest.mark.live
@pytest.mark.slow
def test_pooled_expectancy_not_absurdly_negative(live_backtest):
    total = live_backtest["total"]
    # High WR with catastrophic expectancy would be a red flag
    assert total.expectancy_pct > -0.05


@pytest.mark.live
@pytest.mark.slow
def test_live_fetch_coverage(live_backtest):
    assert len(live_backtest["prices"]) >= 15
    # Stocks + futures + options underlyings should mostly resolve
    for sym in STOCKS[:4] + OPTIONS_UNDERLYINGS[:3]:
        assert sym in live_backtest["prices"]


@pytest.mark.live
@pytest.mark.slow
def test_cached_report_agrees_with_definition():
    """Fixture JSON from last serious run documents the gate."""
    assert CACHE.exists(), "missing data/cache/multi_horizon_backtest.json"
    report = json.loads(CACHE.read_text())
    assert report.get("target_win_rate", 0) >= TARGET
    assert report.get("target_met_per_horizon") or report.get("target_met")
    below = report.get("horizons_below_target") or []
    assert below == []
    for h, meta in report.get("per_horizon", {}).items():
        if meta.get("n_trades", 0) > 0:
            assert meta["win_rate"] >= TARGET, f"cached {h} below target"


def test_options_params_are_modeled_not_tape():
    """Guardrail: default options path stays explicitly modeled BS credit spreads."""
    p = default_options_params(7)
    assert "short_otm_pct" in p and "rsi_entry" in p
    assert p["short_otm_pct"] >= 0.05
