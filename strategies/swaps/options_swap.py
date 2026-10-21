"""
Options swap support.

Models a variance/volatility-style or option-package swap where two parties
exchange option cashflows (or a forward on implied vol) with mark-to-market
via Black–Scholes. Spread-aware entry avoids paying up in wide option markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from pricing.black_scholes import BlackScholes, OptionContract
from strategies.execution import SpreadAwareExecutor

SwapSide = Literal["pay_fixed", "receive_fixed"]


@dataclass
class OptionsSwapSpec:
    notional: float
    fixed_rate: float  # fixed implied-vol or premium rate
    option_type: Literal["call", "put"] = "call"
    strike: float = 100.0
    tenor_years: float = 0.25
    rate: float = 0.04
    dividend_yield: float = 0.0
    side: SwapSide = "pay_fixed"


@dataclass
class OptionsSwapMTM:
    fair_vol: float
    floating_premium: float
    fixed_premium: float
    mtm: float
    rejected: bool
    reason: str = ""


class OptionsSwap:
    def __init__(
        self,
        spec: OptionsSwapSpec,
        max_spread_bps: float = 50.0,
        executor: Optional[SpreadAwareExecutor] = None,
    ):
        self.spec = spec
        self.executor = executor or SpreadAwareExecutor(max_spread_bps=max_spread_bps)

    def mark_to_market(
        self,
        spot: float,
        fair_vol: float,
        quote_spread: float = 0.0,
    ) -> OptionsSwapMTM:
        ok, reason, _ = self.executor.should_trade("buy", spot, quote_spread)
        if not ok:
            return OptionsSwapMTM(fair_vol, np.nan, np.nan, 0.0, True, reason)

        floating = BlackScholes(
            OptionContract(
                spot=spot,
                strike=self.spec.strike,
                rate=self.spec.rate,
                dividend_yield=self.spec.dividend_yield,
                volatility=fair_vol,
                time_to_expiry=self.spec.tenor_years,
                option_type=self.spec.option_type,
            )
        ).price()
        fixed = BlackScholes(
            OptionContract(
                spot=spot,
                strike=self.spec.strike,
                rate=self.spec.rate,
                dividend_yield=self.spec.dividend_yield,
                volatility=self.spec.fixed_rate,
                time_to_expiry=self.spec.tenor_years,
                option_type=self.spec.option_type,
            )
        ).price()

        # pay_fixed: receive floating premium, pay fixed
        diff = floating - fixed
        if self.spec.side == "receive_fixed":
            diff = -diff
        mtm = diff * self.spec.notional / max(spot, 1e-9)
        return OptionsSwapMTM(fair_vol, floating, fixed, mtm, False, "ok")

    def path_mtm(
        self,
        spots: pd.Series,
        vols: pd.Series,
        spreads: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        rows = []
        aligned = pd.concat({"spot": spots, "vol": vols}, axis=1).dropna()
        for ts, row in aligned.iterrows():
            spread = (
                float(spreads.loc[ts])
                if spreads is not None and ts in spreads.index
                else 0.001
            )
            m = self.mark_to_market(float(row["spot"]), float(row["vol"]), spread)
            rows.append(
                {
                    "timestamp": ts,
                    "spot": row["spot"],
                    "fair_vol": m.fair_vol,
                    "floating_premium": m.floating_premium,
                    "fixed_premium": m.fixed_premium,
                    "mtm": m.mtm,
                    "rejected": m.rejected,
                    "reason": m.reason,
                }
            )
        return pd.DataFrame(rows).set_index("timestamp")
