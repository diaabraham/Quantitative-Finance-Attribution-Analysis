"""Spread-aware / adverse-selection-aware execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FillResult:
    filled_qty: float
    fill_price: float
    mid: float
    spread: float
    slipped: float
    rejected: bool
    reason: str = ""


class SpreadAwareExecutor:
    """
    Avoid getting stuck in large spreads / losing to market makers.

    Rules:
    - Reject (or clip) orders when quoted spread exceeds ``max_spread_bps``.
    - Size down when short-term adverse selection proxy is elevated
      (return since mid move against the trade).
    - Apply half-spread + impact as execution cost for backtests.
    """

    def __init__(
        self,
        max_spread_bps: float = 25.0,
        adverse_window: int = 5,
        adverse_threshold: float = 0.002,
        impact_bps: float = 2.0,
        min_fill_ratio: float = 0.0,
    ):
        self.max_spread_bps = max_spread_bps
        self.adverse_window = adverse_window
        self.adverse_threshold = adverse_threshold
        self.impact_bps = impact_bps
        self.min_fill_ratio = min_fill_ratio

    @staticmethod
    def estimate_spread(high: float, low: float, close: float) -> float:
        """Corwin–Schultz-ish proxy from daily H/L when true quotes unavailable."""
        if close <= 0 or high <= 0 or low <= 0 or high < low:
            return np.nan
        # Simple relative range proxy (half of H-L over mid)
        mid = 0.5 * (high + low)
        return max(high - low, 0.0) / mid if mid else np.nan

    def should_trade(
        self,
        side: str,
        mid: float,
        spread: float,
        recent_returns: Optional[pd.Series] = None,
    ) -> tuple[bool, str, float]:
        """
        Returns (ok, reason, size_multiplier).
        """
        if mid <= 0 or not np.isfinite(spread):
            return False, "invalid_quote", 0.0
        spread_bps = 1e4 * spread
        if spread_bps > self.max_spread_bps:
            return False, f"spread_too_wide:{spread_bps:.1f}bps", 0.0

        size_mult = 1.0
        if recent_returns is not None and len(recent_returns) > 0:
            window = recent_returns.tail(self.adverse_window)
            # Adverse if buying into recent up-move / selling into down-move
            move = float(window.sum())
            if side.lower() in ("buy", "b", "long") and move > self.adverse_threshold:
                size_mult = 0.35
                return True, "adverse_selection_clip", size_mult
            if side.lower() in ("sell", "s", "short") and move < -self.adverse_threshold:
                size_mult = 0.35
                return True, "adverse_selection_clip", size_mult
        return True, "ok", size_mult

    def simulate_fill(
        self,
        side: str,
        qty: float,
        mid: float,
        spread: float,
        recent_returns: Optional[pd.Series] = None,
    ) -> FillResult:
        ok, reason, size_mult = self.should_trade(side, mid, spread, recent_returns)
        if not ok:
            return FillResult(0.0, mid, mid, spread, 0.0, True, reason)

        filled = qty * size_mult
        if abs(filled) < abs(qty) * self.min_fill_ratio and self.min_fill_ratio > 0:
            return FillResult(0.0, mid, mid, spread, 0.0, True, "below_min_fill")

        half = 0.5 * spread * mid
        impact = mid * (self.impact_bps / 1e4)
        if side.lower() in ("buy", "b", "long"):
            fill_price = mid + half + impact
        else:
            fill_price = mid - half - impact
        slipped = abs(fill_price - mid) / mid
        return FillResult(filled, fill_price, mid, spread, slipped, False, reason)

    def apply_to_weights(
        self,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
        highs: Optional[pd.DataFrame] = None,
        lows: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter target weight changes that would cross too-wide spreads.

        Returns (executable_weights, diagnostics DataFrame of rejected notional fractions).
        """
        executable = target_weights.copy().astype(float)
        diagnostics = pd.DataFrame(0.0, index=target_weights.index, columns=target_weights.columns)
        returns = prices.pct_change()

        for t in range(1, len(target_weights)):
            idx = target_weights.index[t]
            prev = target_weights.index[t - 1]
            for col in target_weights.columns:
                delta = float(target_weights.loc[idx, col] - target_weights.loc[prev, col])
                if abs(delta) < 1e-12:
                    continue
                mid = float(prices.loc[idx, col]) if col in prices.columns else np.nan
                if highs is not None and lows is not None and col in highs.columns:
                    spread = self.estimate_spread(
                        float(highs.loc[idx, col]),
                        float(lows.loc[idx, col]),
                        mid,
                    )
                else:
                    # Fallback: use recent volatility as spread proxy
                    vol = float(returns[col].iloc[max(0, t - 20) : t].std())
                    spread = 2.0 * vol if np.isfinite(vol) else np.nan
                side = "buy" if delta > 0 else "sell"
                ok, reason, size_mult = self.should_trade(
                    side, mid, spread if np.isfinite(spread) else np.nan, returns[col].iloc[:t]
                )
                if not ok:
                    executable.loc[idx, col] = executable.loc[prev, col]
                    diagnostics.loc[idx, col] = abs(delta)
                elif size_mult < 1.0:
                    executable.loc[idx, col] = (
                        executable.loc[prev, col] + delta * size_mult
                    )
                    diagnostics.loc[idx, col] = abs(delta) * (1.0 - size_mult)
        return executable, diagnostics
