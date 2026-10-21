#!/usr/bin/env python3
"""
Rigorous multi-provider fetch + sector-rotation backtest harness.

Runs every configured provider (yfinance, stooq, massive, polygon, databento),
compares overlapping daily returns, and backtests the sector-rotation strategy
on each successful dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.providers import PROVIDER_NAMES, get_provider, list_available_providers
from data.providers.base import ProviderError
from main import SectorRotationAlpha


DEFAULT_SYMBOLS = ["XLK", "XLF", "XLV", "XLE", "XLI"]


@dataclass
class ProviderFetchResult:
    provider: str
    available: bool
    ok: bool
    rows: int = 0
    cols: int = 0
    start: str | None = None
    end: str | None = None
    error: str | None = None


@dataclass
class BacktestResult:
    provider: str
    method: str
    ok: bool
    total_return: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    volatility: float | None = None
    error: str | None = None


def _corr_matrix(frames: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
    series_map = {}
    for name, frame in frames.items():
        if symbol in frame.columns:
            rets = frame[symbol].pct_change().dropna()
            if not rets.empty:
                series_map[name] = rets
    if len(series_map) < 2:
        return pd.DataFrame()
    aligned = pd.DataFrame(series_map).dropna(how="any")
    if aligned.empty:
        return pd.DataFrame()
    return aligned.corr()


def fetch_all(
    providers: list[str],
    symbols: list[str],
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], list[ProviderFetchResult]]:
    frames: dict[str, pd.DataFrame] = {}
    reports: list[ProviderFetchResult] = []

    for name in providers:
        provider = get_provider(name)
        if not provider.available():
            reports.append(
                ProviderFetchResult(
                    provider=name,
                    available=False,
                    ok=False,
                    error=provider.unavailable_reason() or "unavailable",
                )
            )
            continue
        try:
            frame = provider.fetch_closes(symbols, start, end)
            frames[name] = frame
            reports.append(
                ProviderFetchResult(
                    provider=name,
                    available=True,
                    ok=True,
                    rows=int(len(frame)),
                    cols=int(frame.shape[1]),
                    start=str(frame.index.min().date()),
                    end=str(frame.index.max().date()),
                )
            )
            print(
                f"[OK] {name}: {frame.shape[1]} symbols, {len(frame)} rows "
                f"({frame.index.min().date()} → {frame.index.max().date()})"
            )
        except Exception as exc:  # noqa: BLE001
            reports.append(
                ProviderFetchResult(
                    provider=name,
                    available=True,
                    ok=False,
                    error=str(exc),
                )
            )
            print(f"[FAIL] {name}: {exc}")
    return frames, reports


def run_backtests(
    frames: dict[str, pd.DataFrame],
    start: str,
    end: str,
    methods: list[str],
) -> list[BacktestResult]:
    results: list[BacktestResult] = []
    for provider_name, frame in frames.items():
        symbols = list(frame.columns)
        try:
            strategy = SectorRotationAlpha(
                provider=provider_name,
                max_position_size=0.35,
                max_sector_exposure=0.45,
            )
            # Bypass network re-fetch: inject prices directly
            strategy.sector_etfs = {
                sym: {"symbol": sym, "cik": ""} for sym in symbols
            }
            strategy.data = frame.copy()
            # Benchmark: equal-weight of available names if market fetch fails
            try:
                strategy.benchmark_data = strategy.data_manager.get_market_data(
                    start_date=start, end_date=end
                )
            except Exception:
                strategy.benchmark_data = None
            if strategy.benchmark_data is None or getattr(strategy.benchmark_data, "empty", False):
                strategy.benchmark_data = frame.mean(axis=1)

            strategy.calculate_momentum_indicators()
            signals = strategy.generate_signals(top_n=min(3, len(symbols)))

            for method in methods:
                try:
                    portfolio_value, strategy_returns, _weights = strategy.backtest(
                        signals, weighting_method=method
                    )
                    metrics = strategy.performance_metrics or {}
                    total_return = float(portfolio_value.dropna().iloc[-1] / portfolio_value.dropna().iloc[0] - 1)
                    results.append(
                        BacktestResult(
                            provider=provider_name,
                            method=method,
                            ok=True,
                            total_return=total_return,
                            sharpe=_metric(metrics, "sharpe_ratio"),
                            max_drawdown=_metric(metrics, "max_drawdown"),
                            volatility=_metric(metrics, "volatility"),
                        )
                    )
                    print(
                        f"[BT] {provider_name}/{method}: "
                        f"return={total_return:.2%} sharpe={results[-1].sharpe}"
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        BacktestResult(
                            provider=provider_name,
                            method=method,
                            ok=False,
                            error=str(exc),
                        )
                    )
                    print(f"[BT-FAIL] {provider_name}/{method}: {exc}")
                    traceback.print_exc()
        except Exception as exc:  # noqa: BLE001
            results.append(
                BacktestResult(
                    provider=provider_name,
                    method="*",
                    ok=False,
                    error=str(exc),
                )
            )
            print(f"[BT-FAIL] {provider_name}: {exc}")
            traceback.print_exc()
    return results


def _metric(metrics: dict, key: str) -> float | None:
    if not metrics:
        return None

    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    # tolerate alternate key styles
    for candidate in (
        key,
        key.replace("_", " "),
        key.replace("_", " ").title(),
        key.upper(),
    ):
        if candidate in metrics:
            try:
                return float(metrics[candidate])
            except Exception:
                return None
    needle = _norm(key)
    for k, v in metrics.items():
        if needle and needle in _norm(k):
            try:
                return float(v)
            except Exception:
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="*",
        default=list(PROVIDER_NAMES),
        help="Providers to exercise (default: all)",
    )
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["risk_parity", "equal_risk"],
    )
    parser.add_argument(
        "--out",
        default="data/cache/provider_backtest_report.json",
    )
    args = parser.parse_args()

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    start = args.start or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    print("=" * 72)
    print("Provider availability")
    print("=" * 72)
    for p in list_available_providers(order=args.providers):
        print(f"  available: {p.name}")
    for name in args.providers:
        p = get_provider(name)
        if not p.available():
            print(f"  skipped:   {name} ({p.unavailable_reason()})")

    print("\n" + "=" * 72)
    print(f"Fetching {args.symbols}  {start} → {end}")
    print("=" * 72)
    frames, fetch_reports = fetch_all(args.providers, args.symbols, start, end)

    print("\n" + "=" * 72)
    print("Cross-provider return correlations (XLK when present)")
    print("=" * 72)
    symbol = args.symbols[0]
    corr = _corr_matrix(frames, symbol)
    if corr.empty:
        print("  insufficient overlapping data for correlation matrix")
    else:
        print(corr.round(4).to_string())

    print("\n" + "=" * 72)
    print("Backtests")
    print("=" * 72)
    bt_results = run_backtests(frames, start, end, args.methods)

    report = {
        "generated_at": datetime.now().isoformat() + "Z",
        "start": start,
        "end": end,
        "symbols": args.symbols,
        "fetch": [asdict(r) for r in fetch_reports],
        "correlations": corr.round(6).to_dict() if not corr.empty else {},
        "backtests": [asdict(r) for r in bt_results],
        "summary": {
            "providers_ok": sum(1 for r in fetch_reports if r.ok),
            "providers_failed": sum(1 for r in fetch_reports if r.available and not r.ok),
            "providers_skipped": sum(1 for r in fetch_reports if not r.available),
            "backtests_ok": sum(1 for r in bt_results if r.ok),
            "backtests_failed": sum(1 for r in bt_results if not r.ok),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report → {out_path}")
    print(json.dumps(report["summary"], indent=2))

    # Exit non-zero only if every attempted available provider failed
    attempted = [r for r in fetch_reports if r.available]
    if attempted and not any(r.ok for r in attempted):
        return 2
    if not any(r.ok for r in bt_results):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
