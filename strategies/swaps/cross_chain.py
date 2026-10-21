"""
Cross-chain swap support (modeled).

Represents bridging / atomic-swap style FX between chain notions (e.g. USDC-ETH
vs USDC-SOL) with:
  - quoted price
  - bridge fee + gas
  - latency (blocks / minutes)
  - spread-aware abort when liquidity is thin

This is a financial model for attribution/backtests, not an on-chain executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from strategies.execution import SpreadAwareExecutor


@dataclass
class CrossChainQuote:
    src_chain: str
    dst_chain: str
    src_asset: str
    dst_asset: str
    mid: float
    spread: float
    bridge_fee_bps: float
    gas_src: float
    gas_dst: float
    latency_minutes: float


@dataclass
class CrossChainFill:
    qty_src: float
    qty_dst: float
    effective_rate: float
    total_fees: float
    rejected: bool
    reason: str = ""
    quote: Optional[CrossChainQuote] = None


class CrossChainSwap:
    def __init__(
        self,
        bridge_fee_bps: float = 8.0,
        max_spread_bps: float = 40.0,
        gas_src: float = 2.0,
        gas_dst: float = 0.5,
        latency_minutes: float = 15.0,
        executor: Optional[SpreadAwareExecutor] = None,
    ):
        self.bridge_fee_bps = bridge_fee_bps
        self.gas_src = gas_src
        self.gas_dst = gas_dst
        self.latency_minutes = latency_minutes
        self.executor = executor or SpreadAwareExecutor(max_spread_bps=max_spread_bps)

    def quote(
        self,
        src_chain: str,
        dst_chain: str,
        src_asset: str,
        dst_asset: str,
        mid: float,
        spread: float,
    ) -> CrossChainQuote:
        return CrossChainQuote(
            src_chain=src_chain,
            dst_chain=dst_chain,
            src_asset=src_asset,
            dst_asset=dst_asset,
            mid=mid,
            spread=spread,
            bridge_fee_bps=self.bridge_fee_bps,
            gas_src=self.gas_src,
            gas_dst=self.gas_dst,
            latency_minutes=self.latency_minutes,
        )

    def execute(self, qty_src: float, quote: CrossChainQuote) -> CrossChainFill:
        ok, reason, size_mult = self.executor.should_trade(
            "sell", quote.mid, quote.spread
        )
        if not ok:
            return CrossChainFill(0.0, 0.0, np.nan, 0.0, True, reason, quote)

        qty = qty_src * size_mult
        half_spread_cost = 0.5 * quote.spread * quote.mid * qty
        bridge_fee = qty * quote.mid * (quote.bridge_fee_bps / 1e4)
        gas = quote.gas_src + quote.gas_dst
        total_fees = half_spread_cost + bridge_fee + gas
        # Selling src for dst at mid - half spread
        rate = quote.mid * (1.0 - 0.5 * quote.spread) * (
            1.0 - quote.bridge_fee_bps / 1e4
        )
        qty_dst = qty * rate
        # Deduct gas in dst units approximately
        qty_dst = max(qty_dst - gas, 0.0)
        effective = qty_dst / qty if qty else np.nan
        return CrossChainFill(qty, qty_dst, effective, total_fees, False, "ok", quote)

    def backtest_path(
        self,
        mid_series: pd.Series,
        spread_series: Optional[pd.Series] = None,
        qty_src: float = 1.0,
        src_chain: str = "ethereum",
        dst_chain: str = "solana",
        src_asset: str = "USDC",
        dst_asset: str = "USDC",
    ) -> pd.DataFrame:
        rows = []
        for ts, mid in mid_series.dropna().items():
            spread = (
                float(spread_series.loc[ts])
                if spread_series is not None and ts in spread_series.index
                else 0.001
            )
            q = self.quote(src_chain, dst_chain, src_asset, dst_asset, float(mid), spread)
            fill = self.execute(qty_src, q)
            rows.append(
                {
                    "timestamp": ts,
                    "mid": mid,
                    "spread": spread,
                    "rejected": fill.rejected,
                    "reason": fill.reason,
                    "qty_dst": fill.qty_dst,
                    "effective_rate": fill.effective_rate,
                    "total_fees": fill.total_fees,
                    "latency_minutes": q.latency_minutes,
                }
            )
        return pd.DataFrame(rows).set_index("timestamp")
