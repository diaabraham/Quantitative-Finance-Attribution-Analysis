#!/usr/bin/env python3
"""
Multi-horizon rigorous backtest across stocks, futures, and options.

Win-rate definition (primary, documented):
  Trade win rate = (# closed trades with net PnL > 0 after spread + cost) /
                   (# closed trades).
  Rejected quotes never enter the denominator.
  Total win rate = same metric pooled across all horizons and asset classes.

Horizons use recent market windows ending at --end (default: today).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.providers import build_provider
from strategies.high_probability import (
    HighProbabilityMeanReversion,
    OptionsCreditPutPath,
    Trade,
    summarize_trades,
)

HORIZONS_DAYS = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "3y": 365 * 3,
    "5y": 365 * 5,
}

# Minimum calendar days of history to fetch (need warmup bars beyond horizon)
FETCH_PADDING_DAYS = {
    "1d": 90,
    "1w": 120,
    "1m": 180,
    "3m": 270,
    "6m": 400,
    "1y": 500,
    "3y": 365 * 3 + 60,
    "5y": 365 * 5 + 60,
}

STOCKS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "META", "GOOGL", "AMZN"]
FUTURES = ["ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "GC=F", "SI=F", "ZB=F"]
# Options modeled on liquid underlyings
OPTIONS_UNDERLYINGS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--end", default=None)
    p.add_argument("--provider", default=os.getenv("MARKET_DATA_PROVIDER", "yfinance"))
    p.add_argument(
        "--out-json",
        default=str(ROOT / "data" / "cache" / "multi_horizon_backtest.json"),
    )
    p.add_argument("--min-trades-total", type=int, default=40)
    p.add_argument("--target-win-rate", type=float, default=0.87)
    return p.parse_args()


def fetch_ohlc(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.rename(columns=str.title)
    for col in ("Open", "High", "Low", "Close"):
        if col not in hist.columns:
            return pd.DataFrame()
    out = hist[["Open", "High", "Low", "Close"]].copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    return out.dropna()


def window_slice(df: pd.DataFrame, end: pd.Timestamp, horizon_days: int, pad: int) -> pd.DataFrame:
    if df.empty:
        return df
    start = end - pd.Timedelta(days=horizon_days + pad)
    return df.loc[(df.index >= start) & (df.index <= end)]


def default_equity_params() -> dict:
    """
    Locked from offline grid on 5y liquid stocks/futures (see evidence notes).
    Preference: high WR with non-disastrous expectancy — not curve-fit to a single week.
    """
    return dict(
        z_entry=2.25,
        tp_atr=0.30,
        sl_atr=1.80,
        rsi_low=30.0,
        rsi_high=70.0,
        max_hold=8,
        max_spread_bps=35.0,
        cost_bps=1.5,
    )


def default_options_params(hold_days: int) -> dict:
    """Bull-put credit spread defaults from grid (positive expectancy configs preferred)."""
    return dict(
        short_otm_pct=0.06,
        width_pct=0.03,
        hold_days=max(3, hold_days),
        vol_lookback=20,
        rsi_entry=35.0,
        max_spread_bps=40.0,
    )


def eval_window_days(horizon: str) -> Tuple[int, int]:
    """
    Returns (eval_calendar_days, max_hold_bars).

    Short horizons use a minimum evaluation sample so trade-level win rates are
    estimable; the holding-period regime still matches the horizon label.
    """
    h = HORIZONS_DAYS[horizon]
    if horizon == "1d":
        return 90, 1
    if horizon == "1w":
        return 120, 5
    if horizon == "1m":
        return 150, 10
    if horizon == "3m":
        return h, 12
    if horizon == "6m":
        return h, 15
    if horizon == "1y":
        return h, 18
    if horizon == "3y":
        return h, 22
    return h, 25


def run_horizon(
    horizon: str,
    end: pd.Timestamp,
    prices: Dict[str, pd.DataFrame],
    stock_params: dict,
    opt_hold: int,
) -> Tuple[List[Trade], dict]:
    eval_days, max_hold = eval_window_days(horizon)
    h_days = HORIZONS_DAYS[horizon]
    params = dict(stock_params)
    params["max_hold"] = max_hold
    strat = HighProbabilityMeanReversion(**params)
    opt_hold_h = max(3, min(opt_hold, max_hold if max_hold >= 3 else 3))
    opt = OptionsCreditPutPath(**default_options_params(opt_hold_h))

    # Warmup pad for indicators; count only trades with entry inside eval window
    window_start = end - pd.Timedelta(days=eval_days)
    trades: List[Trade] = []

    for sym in STOCKS:
        df = prices.get(sym)
        if df is None or df.empty:
            continue
        hist = window_slice(df, end, eval_days, pad=80)
        tlist = strat.run(hist, sym, "stocks", horizon)
        trades.extend([t for t in tlist if t.entry_time >= window_start])

    for sym in FUTURES:
        df = prices.get(sym)
        if df is None or df.empty:
            continue
        hist = window_slice(df, end, eval_days, pad=80)
        tlist = strat.run(hist, sym, "futures", horizon)
        trades.extend([t for t in tlist if t.entry_time >= window_start])

    for sym in OPTIONS_UNDERLYINGS:
        df = prices.get(sym)
        if df is None or df.empty:
            continue
        hist = window_slice(df, end, eval_days, pad=80)
        tlist = opt.run(hist, sym, horizon)
        for t in tlist:
            t.horizon = horizon
        trades.extend([t for t in tlist if t.entry_time >= window_start])

    summary = summarize_trades(trades)
    by_class = {}
    for cls in ("stocks", "futures", "options"):
        sub = [t for t in trades if t.asset_class == cls]
        d = asdict(summarize_trades(sub))
        d.pop("trades", None)
        by_class[cls] = d

    meta = {
        "horizon": horizon,
        "horizon_days_label": h_days,
        "eval_calendar_days": eval_days,
        "max_hold_bars": max_hold,
        "window_start": str(window_start.date()),
        "window_end": str(end.date()),
        "win_rate": summary.win_rate,
        "n_trades": summary.n_trades,
        "n_wins": summary.n_wins,
        "expectancy_pct": summary.expectancy_pct,
        "profit_factor": summary.profit_factor,
        "total_pnl_pct": summary.total_pnl_pct,
        "by_asset_class": by_class,
    }
    return trades, meta


def main():
    args = _parse_args()
    end = pd.Timestamp(args.end or datetime.now().strftime("%Y-%m-%d")).normalize()
    # Fetch max lookback once
    start = (end - pd.Timedelta(days=365 * 5 + 90)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    universe = sorted(set(STOCKS + FUTURES + OPTIONS_UNDERLYINGS))
    print(f"Fetching {len(universe)} symbols {start} → {end_s} via history API…")
    prices: Dict[str, pd.DataFrame] = {}
    fetch_errors = {}
    for sym in universe:
        try:
            df = fetch_ohlc(sym, start, end_s)
            if df.empty:
                fetch_errors[sym] = "empty"
            else:
                prices[sym] = df
                print(f"  {sym}: {len(df)} bars")
        except Exception as exc:  # noqa: BLE001
            fetch_errors[sym] = str(exc)[:200]
            print(f"  {sym}: ERROR {exc}")

    params = default_equity_params()
    opt_hold = 7
    print("Equity params:", params)
    print("Options defaults:", default_options_params(opt_hold))

    all_trades: List[Trade] = []
    per_horizon = {}
    for h in HORIZONS_DAYS:
        print(f"Running horizon {h}…")
        trades, meta = run_horizon(h, end, prices, params, opt_hold)
        all_trades.extend(trades)
        per_horizon[h] = meta
        print(
            f"  {h}: WR={meta['win_rate']:.1%} trades={meta['n_trades']} "
            f"exp={meta['expectancy_pct']:.4f}"
        )

    total = summarize_trades(all_trades)
    by_class_total = {}
    for cls in ("stocks", "futures", "options"):
        sub = summarize_trades([t for t in all_trades if t.asset_class == cls])
        by_class_total[cls] = {
            "win_rate": sub.win_rate,
            "n_trades": sub.n_trades,
            "n_wins": sub.n_wins,
            "expectancy_pct": sub.expectancy_pct,
            "profit_factor": sub.profit_factor,
        }

    notes = [
        "Parameters locked from a documented grid favoring high trade win rate "
        "with preference for non-negative expectancy on options credit spreads.",
        "Short horizons (<1m) use a minimum ~90–150 calendar-day evaluation window "
        "so trade counts are estimable; holding-period regime matches the horizon label.",
        "Options path is a modeled bull-put credit spread (BS on underlying), "
        "not historical option NBBO tape.",
    ]

    hz_rates = [
        per_horizon[h]["win_rate"]
        for h in HORIZONS_DAYS
        if per_horizon[h]["n_trades"] > 0
    ]
    horizon_equal_wr = float(np.mean(hz_rates)) if hz_rates else 0.0

    target_met = (
        total.win_rate >= args.target_win_rate and total.n_trades >= args.min_trades_total
    )
    if not target_met:
        notes.append(
            "Target ≥87% pooled trade win rate with adequate trade count was not "
            "achieved without fabricating results. Numbers below are honest."
        )
    if total.expectancy_pct < 0:
        notes.append(
            "Expectancy (mean pnl_pct) is negative despite high win rate — "
            "win rate ≠ profitability."
        )

    report = {
        "generated_at": datetime.now().isoformat() + "Z",
        "win_rate_definition": (
            "Trade win rate = fraction of closed round-trip trades with net PnL > 0 "
            "after half-spread execution costs and per-leg cost_bps. Rejected quotes "
            "are not trades. Total = pooled across all horizons and asset classes."
        ),
        "target_win_rate": args.target_win_rate,
        "target_met": target_met,
        "provider": args.provider,
        "end": end_s,
        "universe": {
            "stocks": STOCKS,
            "futures": FUTURES,
            "options_underlyings": OPTIONS_UNDERLYINGS,
            "fetched": sorted(prices.keys()),
            "fetch_errors": fetch_errors,
        },
        "params": params,
        "notes": notes,
        "total": {
            "win_rate": total.win_rate,
            "win_rate_horizon_equal": horizon_equal_wr,
            "n_trades": total.n_trades,
            "n_wins": total.n_wins,
            "expectancy_pct": total.expectancy_pct,
            "profit_factor": total.profit_factor,
            "total_pnl_pct": total.total_pnl_pct,
        },
        "by_asset_class": by_class_total,
        "per_horizon": per_horizon,
        "sample_trades": [
            {
                "asset": t.asset,
                "asset_class": t.asset_class,
                "horizon": t.horizon,
                "side": t.side,
                "entry": str(t.entry_time.date()),
                "exit": str(t.exit_time.date()),
                "pnl_pct": t.pnl_pct,
                "reason_exit": t.reason_exit,
            }
            for t in all_trades[:40]
        ],
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\n=== TOTAL ===")
    print(
        f"Win rate: {total.win_rate:.2%}  ({total.n_wins}/{total.n_trades})  "
        f"target_met={target_met}"
    )
    print(f"Wrote {out}")
    return 0 if total.n_trades > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
