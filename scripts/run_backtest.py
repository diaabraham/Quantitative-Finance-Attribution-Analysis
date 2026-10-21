#!/usr/bin/env python3
"""
Rigorous update + backtest entrypoint for QuantFAA.

Exercises:
  - market-data providers (default yfinance + stooq fallback)
  - sector-rotation backtest with spread-aware execution
  - Brinson attribution on sector sleeves
  - recursive + ML strategy paths
  - Black–Scholes sanity
  - Norbert's Gambit, cross-chain + options swaps (synthetic legs when needed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attribution.brinson import BrinsonAttribution
from data.data_manager import DataManager
from data.providers import available_providers, build_provider
from main import SectorRotationAlpha
from pricing.black_scholes import BlackScholes, OptionContract
from strategies.execution import SpreadAwareExecutor
from strategies.ml_strategy import MLMomentumStrategy
from strategies.norberts_gambit import NorbertsGambit
from strategies.recursive import RecursiveStrategy
from strategies.swaps import CrossChainSwap, OptionsSwap
from strategies.swaps.options_swap import OptionsSwapSpec


SECTOR_TICKERS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE"]


def _parse_args():
    p = argparse.ArgumentParser(description="QuantFAA rigorous backtest")
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--provider", default=os.getenv("MARKET_DATA_PROVIDER", "yfinance"))
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument(
        "--out",
        default=str(ROOT / "data" / "cache" / "backtest_report.json"),
    )
    p.add_argument("--skip-ml", action="store_true")
    return p.parse_args()


def provider_smoke(provider_name: str, start: str, end: str) -> dict:
    results = {}
    for name in available_providers():
        entry = {"status": "skipped", "rows": 0, "error": None}
        try:
            if name in ("polygon", "massive") and not (
                os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
            ):
                entry["status"] = "skipped_no_key"
                results[name] = entry
                continue
            if name == "databento" and not os.getenv("DATABENTO_API_KEY"):
                entry["status"] = "skipped_no_key"
                results[name] = entry
                continue
            if name == "stooq" and name != provider_name:
                # Still try briefly; may need key / JS challenge
                pass
            prov = build_provider(name, fallbacks=[])
            df = prov.fetch_closes(["XLK", "XLF"], start, end)
            entry["status"] = "ok" if not df.empty else "empty"
            entry["rows"] = int(len(df))
            entry["columns"] = list(df.columns)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = str(exc)[:300]
        results[name] = entry
    return results


def run_sector_rotation(dm: DataManager, start: str, end: str) -> dict:
    strategy = SectorRotationAlpha(
        transaction_cost=0.001,
        max_position_size=0.25,
        max_sector_exposure=0.35,
        max_drawdown=0.25,
        target_volatility=0.15,
    )
    # Inject provider-aware data manager
    strategy.data_manager = dm
    data = strategy.fetch_data(start, end)
    strategy.calculate_momentum_indicators()
    signals = strategy.generate_signals(top_n=3)

    executor = SpreadAwareExecutor(max_spread_bps=80.0)
    # Build equal-weight targets from signals for spread filter
    raw_w = signals.div(signals.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    filtered_w, rejected = executor.apply_to_weights(raw_w, data)

    methods = {}
    for method in ["risk_parity", "equal_risk"]:
        pv, rets, weights = strategy.backtest(signals, weighting_method=method)
        # Overlay spread filter on weights path
        w2, _ = executor.apply_to_weights(weights.fillna(0.0), data)
        rets2 = (w2 * data.pct_change()).sum(axis=1)
        costs = (w2.diff().abs() * strategy.transaction_cost).sum(axis=1)
        rets2 = rets2 - costs
        pv2 = 100000 * (1 + rets2.fillna(0)).cumprod()
        methods[method] = {
            "final_value": float(pv2.dropna().iloc[-1]),
            "total_return": float(pv2.dropna().iloc[-1] / pv2.dropna().iloc[0] - 1),
            "ann_vol": float(rets2.std() * np.sqrt(252)),
            "sharpe": float(
                (rets2.mean() / rets2.std()) * np.sqrt(252)
            )
            if rets2.std() > 0
            else 0.0,
            "spread_rejected_mean": float(rejected.mean().mean()),
        }

    # Brinson on last-day sector sleeve vs equal-weight benchmark
    last = data.index[-1]
    port_w = filtered_w.loc[last].replace(0, np.nan).dropna()
    if port_w.empty:
        port_w = pd.Series(1.0 / len(data.columns), index=data.columns)
    else:
        port_w = port_w / port_w.sum()
    bench_w = pd.Series(1.0 / len(data.columns), index=data.columns)
    # Use trailing 21d returns as sector returns
    trail = data.pct_change(21).loc[last]
    brinson = BrinsonAttribution(
        portfolio_returns=trail.reindex(port_w.index).fillna(0),
        benchmark_returns=trail.reindex(bench_w.index).fillna(0),
        portfolio_weights=port_w.reindex(trail.index).fillna(0),
        benchmark_weights=bench_w.reindex(trail.index).fillna(0),
    )
    br = brinson.calculate_attribution()

    return {
        "tickers": list(data.columns),
        "rows": int(len(data)),
        "start": str(data.index[0].date()),
        "end": str(data.index[-1].date()),
        "methods": methods,
        "brinson": {k: float(v) for k, v in br.items()},
        "provider": getattr(dm.provider, "name", "unknown"),
    }


def run_recursive_and_ml(prices: pd.DataFrame, skip_ml: bool) -> dict:
    rec = RecursiveStrategy.multi_horizon(horizons=(20, 60, 120), top_n=3)
    w_rec = rec.generate_weights(prices)
    r_rec = (w_rec * prices.pct_change()).sum(axis=1).dropna()
    out = {
        "recursive": {
            "final_growth": float((1 + r_rec).cumprod().iloc[-1]),
            "sharpe": float((r_rec.mean() / r_rec.std()) * np.sqrt(252))
            if r_rec.std() > 0
            else 0.0,
            "avg_names": float((w_rec > 0).sum(axis=1).mean()),
        }
    }
    if not skip_ml:
        ml = MLMomentumStrategy(train_window=90, min_train=40, top_n=3)
        # Use a shorter window for runtime
        sample = prices.tail(min(len(prices), 320))
        w_ml = ml.generate_weights(sample)
        r_ml = (w_ml * sample.pct_change()).sum(axis=1).dropna()
        out["ml"] = {
            "final_growth": float((1 + r_ml).cumprod().iloc[-1]) if len(r_ml) else None,
            "sharpe": float((r_ml.mean() / r_ml.std()) * np.sqrt(252))
            if len(r_ml) and r_ml.std() > 0
            else 0.0,
            "avg_names": float((w_ml > 0).sum(axis=1).mean()),
        }
    return out


def run_pricing_and_swaps(prices: pd.DataFrame) -> dict:
    spot = float(prices.iloc[-1].mean())
    bs = BlackScholes(
        OptionContract(
            spot=spot,
            strike=spot,
            rate=0.04,
            dividend_yield=0.01,
            volatility=0.2,
            time_to_expiry=0.5,
            option_type="call",
        )
    )
    greeks = bs.greeks()

    # Synthetic CAD/USD dual listing from SPY + FX assumption for gambit demo
    spy = prices.iloc[:, 0].dropna()
    fx = pd.Series(1.35 + 0.02 * np.sin(np.linspace(0, 6, len(spy))), index=spy.index)
    cad = spy * fx
    usd = spy.copy()
    gambit = NorbertsGambit(journal_days=2, retail_fx_spread=0.015)
    gres = gambit.convert_cad_to_usd(cad, usd, 1.0 / fx, notional_cad=10_000)
    gbt = gambit.backtest_schedule(cad, usd, 1.0 / fx, notional_cad=10_000, every_n_days=42)

    # Cross-chain: use first ticker return path as mid proxy around 1.0
    mid = (prices.iloc[:, 0] / prices.iloc[:, 0].iloc[0]).dropna()
    xchain = CrossChainSwap().backtest_path(mid.tail(60), qty_src=1000.0)

    vol = prices.pct_change().rolling(20).std().mean(axis=1).dropna() * np.sqrt(252)
    spot_s = prices.mean(axis=1).reindex(vol.index).dropna()
    vol = vol.reindex(spot_s.index)
    oswap = OptionsSwap(
        OptionsSwapSpec(
            notional=100_000,
            fixed_rate=0.2,
            strike=float(spot_s.iloc[0]),
            tenor_years=0.25,
            side="pay_fixed",
        )
    )
    omtm = oswap.path_mtm(spot_s.tail(60), vol.tail(60))

    return {
        "black_scholes": {"price": float(bs.price()), **{k: float(v) for k, v in greeks.items()}},
        "norberts_gambit": {
            "aborted": gres.aborted,
            "reason": gres.reason,
            "usd_received": gres.usd_received,
            "savings_vs_retail": gres.savings_vs_retail,
            "schedule_rows": int(len(gbt)),
            "schedule_savings_mean": float(gbt["savings_vs_retail"].mean())
            if len(gbt)
            else None,
        },
        "cross_chain": {
            "rows": int(len(xchain)),
            "reject_rate": float(xchain["rejected"].mean()) if len(xchain) else None,
            "mean_fees": float(xchain["total_fees"].mean()) if len(xchain) else None,
        },
        "options_swap": {
            "rows": int(len(omtm)),
            "final_mtm": float(omtm["mtm"].iloc[-1]) if len(omtm) else None,
            "reject_rate": float(omtm["rejected"].mean()) if len(omtm) else None,
        },
    }


def main():
    args = _parse_args()
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    start = args.start or (
        datetime.now() - timedelta(days=int(365 * args.years))
    ).strftime("%Y-%m-%d")

    report = {
        "generated_at": datetime.now().isoformat() + "Z",
        "start": start,
        "end": end,
        "provider_requested": args.provider,
        "providers": {},
        "sector_rotation": {},
        "strategies": {},
        "derivatives": {},
        "ok": False,
    }

    print(f"Provider smoke ({start} → {end})…")
    report["providers"] = provider_smoke(args.provider, start, end)
    for name, info in report["providers"].items():
        print(f"  {name}: {info['status']} rows={info.get('rows')}")

    dm = DataManager(provider=args.provider)
    print(f"Sector rotation via {dm.provider.name}…")
    report["sector_rotation"] = run_sector_rotation(dm, start, end)
    print("  Brinson:", report["sector_rotation"]["brinson"])
    print("  Methods:", report["sector_rotation"]["methods"])

    prices = dm.fetch_historical_data(SECTOR_TICKERS, start, end)
    print("Recursive / ML…")
    report["strategies"] = run_recursive_and_ml(prices, skip_ml=args.skip_ml)
    print("  ", report["strategies"])

    print("Pricing / swaps / Norbert…")
    report["derivatives"] = run_pricing_and_swaps(prices)
    print("  BS price:", report["derivatives"]["black_scholes"]["price"])
    print("  Gambit:", report["derivatives"]["norberts_gambit"])

    report["ok"] = report["sector_rotation"].get("rows", 0) > 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
