"""
Norbert's Gambit (often misspelled "Norbit's Gambit").

CAD↔USD conversion via a dual-listed / dual-currency security:
  1. Buy CAD listing
  2. Journal to USD listing (bookkeeping transfer; modeled as lag)
  3. Sell USD listing

Compares all-in cost vs retail FX spread. Spread-aware execution is applied
on both legs so wide markets abort the gambit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .execution import SpreadAwareExecutor


@dataclass
class GambitResult:
    notional_cad: float
    usd_received: float
    effective_fx: float
    retail_fx: float
    retail_usd: float
    savings_vs_retail: float
    journal_days: int
    legs: pd.DataFrame
    aborted: bool
    reason: str = ""


class NorbertsGambit:
    """
    Simulate Norbert's Gambit on paired CAD/USD price series.

    Parameters
    ----------
    journal_days : int
        Business days between buy and sell (broker journal lag).
    retail_fx_spread : float
        Retail FX half-spread charged by broker (e.g. 0.015 = 1.5%).
    commission : float
        Per-leg commission as fraction of notional.
    pair : tuple[str, str]
        Default ETF pair labels (DLR / DLR.U style).
    """

    def __init__(
        self,
        journal_days: int = 2,
        retail_fx_spread: float = 0.015,
        commission: float = 0.0005,
        max_spread_bps: float = 15.0,
        executor: Optional[SpreadAwareExecutor] = None,
    ):
        self.journal_days = journal_days
        self.retail_fx_spread = retail_fx_spread
        self.commission = commission
        self.executor = executor or SpreadAwareExecutor(max_spread_bps=max_spread_bps)

    def convert_cad_to_usd(
        self,
        cad_prices: pd.Series,
        usd_prices: pd.Series,
        spot_fx: pd.Series,
        notional_cad: float,
        trade_date: Optional[pd.Timestamp] = None,
        cad_high: Optional[pd.Series] = None,
        cad_low: Optional[pd.Series] = None,
        usd_high: Optional[pd.Series] = None,
        usd_low: Optional[pd.Series] = None,
    ) -> GambitResult:
        """
        Run a single gambit conversion starting at ``trade_date`` (or first aligned date).
        """
        df = pd.concat(
            {
                "cad": cad_prices.astype(float),
                "usd": usd_prices.astype(float),
                "fx": spot_fx.astype(float),
            },
            axis=1,
        ).dropna()
        if df.empty:
            return GambitResult(
                notional_cad, 0.0, np.nan, np.nan, 0.0, 0.0, self.journal_days,
                pd.DataFrame(), True, "no_aligned_prices",
            )

        if trade_date is None:
            trade_date = df.index[0]
        if trade_date not in df.index:
            # nearest forward
            future = df.index[df.index >= trade_date]
            if len(future) == 0:
                return GambitResult(
                    notional_cad, 0.0, np.nan, np.nan, 0.0, 0.0, self.journal_days,
                    pd.DataFrame(), True, "trade_date_oob",
                )
            trade_date = future[0]

        loc = df.index.get_loc(trade_date)
        sell_loc = loc + self.journal_days
        if sell_loc >= len(df.index):
            return GambitResult(
                notional_cad, 0.0, np.nan, np.nan, 0.0, 0.0, self.journal_days,
                pd.DataFrame(), True, "insufficient_history_for_journal",
            )
        sell_date = df.index[sell_loc]

        cad_mid = float(df.loc[trade_date, "cad"])
        usd_mid = float(df.loc[sell_date, "usd"])
        fx_buy = float(df.loc[trade_date, "fx"])
        fx_sell = float(df.loc[sell_date, "fx"])

        cad_spread = self._spread(cad_high, cad_low, cad_prices, trade_date, cad_mid)
        usd_spread = self._spread(usd_high, usd_low, usd_prices, sell_date, usd_mid)

        buy_fill = self.executor.simulate_fill("buy", 1.0, cad_mid, cad_spread)
        if buy_fill.rejected:
            return GambitResult(
                notional_cad, 0.0, np.nan, fx_buy * (1 - self.retail_fx_spread),
                notional_cad * fx_buy * (1 - self.retail_fx_spread), 0.0,
                self.journal_days, pd.DataFrame(), True, f"buy_leg:{buy_fill.reason}",
            )

        shares = notional_cad / buy_fill.fill_price
        buy_cost = shares * buy_fill.fill_price * (1 + self.commission)

        sell_fill = self.executor.simulate_fill("sell", shares, usd_mid, usd_spread)
        if sell_fill.rejected:
            return GambitResult(
                notional_cad, 0.0, np.nan, fx_buy * (1 - self.retail_fx_spread),
                notional_cad * fx_buy * (1 - self.retail_fx_spread), 0.0,
                self.journal_days, pd.DataFrame(), True, f"sell_leg:{sell_fill.reason}",
            )

        usd_received = shares * sell_fill.fill_price * (1 - self.commission)
        effective_fx = usd_received / buy_cost  # USD per CAD
        retail_fx = fx_buy * (1.0 - self.retail_fx_spread)
        retail_usd = notional_cad * retail_fx
        savings = usd_received - retail_usd

        legs = pd.DataFrame(
            [
                {
                    "leg": "buy_cad",
                    "date": trade_date,
                    "mid": cad_mid,
                    "fill": buy_fill.fill_price,
                    "spread": cad_spread,
                    "shares": shares,
                    "fx": fx_buy,
                },
                {
                    "leg": "sell_usd",
                    "date": sell_date,
                    "mid": usd_mid,
                    "fill": sell_fill.fill_price,
                    "spread": usd_spread,
                    "shares": shares,
                    "fx": fx_sell,
                },
            ]
        )
        return GambitResult(
            notional_cad=notional_cad,
            usd_received=usd_received,
            effective_fx=effective_fx,
            retail_fx=retail_fx,
            retail_usd=retail_usd,
            savings_vs_retail=savings,
            journal_days=self.journal_days,
            legs=legs,
            aborted=False,
        )

    def backtest_schedule(
        self,
        cad_prices: pd.Series,
        usd_prices: pd.Series,
        spot_fx: pd.Series,
        notional_cad: float,
        every_n_days: int = 21,
    ) -> pd.DataFrame:
        """Run gambit on a schedule; return summary rows."""
        df = pd.concat(
            {"cad": cad_prices, "usd": usd_prices, "fx": spot_fx}, axis=1
        ).dropna()
        rows = []
        for i in range(0, len(df.index) - self.journal_days - 1, every_n_days):
            res = self.convert_cad_to_usd(
                cad_prices, usd_prices, spot_fx, notional_cad, trade_date=df.index[i]
            )
            rows.append(
                {
                    "trade_date": df.index[i],
                    "aborted": res.aborted,
                    "reason": res.reason,
                    "usd_received": res.usd_received,
                    "effective_fx": res.effective_fx,
                    "retail_fx": res.retail_fx,
                    "savings_vs_retail": res.savings_vs_retail,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _spread(high, low, closes, date, mid) -> float:
        if high is not None and low is not None and date in high.index and date in low.index:
            return SpreadAwareExecutor.estimate_spread(
                float(high.loc[date]), float(low.loc[date]), mid
            )
        # 10 bps default tight ETF spread
        return 0.001
