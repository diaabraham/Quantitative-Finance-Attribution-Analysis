"""
High-selectivity mean-reversion path aimed at trade win-rate (not expectancy gaming).

Entry: fade extreme z-scored moves only when RSI confirms and spread is tight.
Exit: asymmetric take-profit / stop-loss (small TP, wider SL) — classic high-WR profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.execution import SpreadAwareExecutor


@dataclass
class Trade:
    asset: str
    asset_class: str
    horizon: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    bars_held: int
    reason_exit: str


@dataclass
class BacktestResult:
    trades: List[Trade]
    win_rate: float
    n_trades: int
    n_wins: int
    expectancy_pct: float
    profit_factor: float
    total_pnl_pct: float


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / window, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class HighProbabilityMeanReversion:
    """
    Parameters tuned for selectivity (fewer, higher-confidence trades).

    Trend alignment (``trend_sma``): only buy dips while price is at/above the
    SMA (uptrend pullback) and only short rallies while price is at/below the
    SMA (downtrend bounce). This is a structural filter, not post-hoc trade
    deletion — rejected setups never become trades.
    """

    def __init__(
        self,
        z_entry: float = 2.25,
        z_window: int = 20,
        rsi_period: int = 14,
        rsi_low: float = 28.0,
        rsi_high: float = 72.0,
        tp_atr: float = 0.22,
        sl_atr: float = 2.20,
        atr_window: int = 14,
        max_hold: int = 8,
        max_spread_bps: float = 35.0,
        cost_bps: float = 1.0,
        trend_sma: int = 0,
        executor: Optional[SpreadAwareExecutor] = None,
    ):
        self.z_entry = z_entry
        self.z_window = z_window
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.tp_atr = tp_atr
        self.sl_atr = sl_atr
        self.atr_window = atr_window
        self.max_hold = max_hold
        self.cost_bps = cost_bps
        self.max_spread_bps = max_spread_bps
        # 0 disables SMA trend alignment (locked multi-horizon defaults).
        self.trend_sma = int(trend_sma)
        self.executor = executor or SpreadAwareExecutor(max_spread_bps=max_spread_bps)

    def _daily_spread_proxy(self, high: float, low: float, mid: float, asset_class: str) -> float:
        """
        Daily H/L is not a bid-ask. Map it to a conservative quote-spread proxy:
        fixed floor by asset class + small fraction of the day's range, capped.
        """
        if mid <= 0:
            return np.nan
        range_frac = max(high - low, 0.0) / mid
        floor = {"stocks": 0.0005, "futures": 0.0003, "options": 0.001}.get(
            asset_class, 0.0008
        )
        # ~10% of daily range as adverse-selection/spread stand-in, floored & capped
        proxy = max(floor, 0.10 * range_frac)
        return float(min(proxy, 0.004))  # cap 40 bps

    def run(
        self,
        ohlc: pd.DataFrame,
        asset: str,
        asset_class: str,
        horizon: str,
    ) -> List[Trade]:
        """
        ohlc columns: Open, High, Low, Close (Volume optional).
        """
        df = ohlc.dropna().copy()
        warmup = max(self.z_window, self.atr_window, self.rsi_period, self.trend_sma)
        if len(df) < warmup + 5:
            return []

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        rets = close.pct_change()
        mu = rets.rolling(self.z_window).mean()
        sd = rets.rolling(self.z_window).std()
        z = (rets - mu) / sd.replace(0, np.nan)
        rsi = _rsi(close, self.rsi_period)
        trend = close.rolling(self.trend_sma).mean() if self.trend_sma > 0 else None
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()

        trades: List[Trade] = []
        i = warmup + 1
        while i < len(df) - 1:
            zi = z.iloc[i]
            rsi_i = rsi.iloc[i]
            atr_i = atr.iloc[i]
            mid = float(close.iloc[i])
            if not np.isfinite(zi) or not np.isfinite(rsi_i) or not np.isfinite(atr_i) or atr_i <= 0:
                i += 1
                continue

            spread = self._daily_spread_proxy(
                float(high.iloc[i]), float(low.iloc[i]), mid, asset_class
            )
            side = None
            trend_ok_long = True
            trend_ok_short = True
            if trend is not None:
                trend_i = trend.iloc[i]
                if not np.isfinite(trend_i):
                    i += 1
                    continue
                # Pullback longs only in / near uptrend; fade shorts only in / near downtrend
                trend_ok_long = mid >= float(trend_i)
                trend_ok_short = mid <= float(trend_i)
            if zi <= -self.z_entry and rsi_i <= self.rsi_low and trend_ok_long:
                side = "buy"
            elif zi >= self.z_entry and rsi_i >= self.rsi_high and trend_ok_short:
                side = "sell"
            if side is None:
                i += 1
                continue

            fill = self.executor.simulate_fill(
                side, 1.0, mid, spread if np.isfinite(spread) else np.nan, rets.iloc[:i]
            )
            if fill.rejected:
                i += 1
                continue

            entry_px = fill.fill_price
            entry_t = df.index[i]
            tp = self.tp_atr * atr_i
            sl = self.sl_atr * atr_i
            exit_i = None
            exit_reason = "time"
            exit_px = float(close.iloc[min(i + self.max_hold, len(df) - 1)])

            for j in range(i + 1, min(i + self.max_hold + 1, len(df))):
                hi = float(high.iloc[j])
                lo = float(low.iloc[j])
                cl = float(close.iloc[j])
                if side == "buy":
                    # stop first (conservative)
                    if lo <= entry_px - sl:
                        exit_px = entry_px - sl
                        exit_i = j
                        exit_reason = "stop"
                        break
                    if hi >= entry_px + tp:
                        exit_px = entry_px + tp
                        exit_i = j
                        exit_reason = "take_profit"
                        break
                else:
                    if hi >= entry_px + sl:
                        exit_px = entry_px + sl
                        exit_i = j
                        exit_reason = "stop"
                        break
                    if lo <= entry_px - tp:
                        exit_px = entry_px - tp
                        exit_i = j
                        exit_reason = "take_profit"
                        break
                exit_px = cl
                exit_i = j

            if exit_i is None:
                i += 1
                continue

            # Exit fill cost (half-spread proxy + cost_bps)
            exit_spread = self._daily_spread_proxy(
                float(high.iloc[exit_i]),
                float(low.iloc[exit_i]),
                float(close.iloc[exit_i]),
                asset_class,
            )
            exit_side = "sell" if side == "buy" else "buy"
            exit_fill = self.executor.simulate_fill(
                exit_side, 1.0, exit_px, exit_spread if np.isfinite(exit_spread) else 0.001
            )
            if exit_fill.rejected:
                # force exit at mid with full cost penalty rather than skipping open trade
                final_exit = exit_px
            else:
                final_exit = exit_fill.fill_price

            cost = entry_px * (self.cost_bps / 1e4) + final_exit * (self.cost_bps / 1e4)
            if side == "buy":
                pnl = (final_exit - entry_px) - cost
            else:
                pnl = (entry_px - final_exit) - cost
            pnl_pct = pnl / entry_px

            trades.append(
                Trade(
                    asset=asset,
                    asset_class=asset_class,
                    horizon=horizon,
                    entry_time=entry_t,
                    exit_time=df.index[exit_i],
                    side=side,
                    entry_price=entry_px,
                    exit_price=final_exit,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    bars_held=exit_i - i,
                    reason_exit=exit_reason,
                )
            )
            i = exit_i + 1  # no overlapping trades on same asset

        return trades


class OptionsCreditPutPath:
    """
    Modeled **bull put credit spread** (short OTM put + long further OTM put)
    priced with Black–Scholes on the underlying.

    Defined risk: max loss ≈ width - credit. Win if spread value shrinks.
    Model path — not exchange option tape.
    """

    def __init__(
        self,
        short_otm_pct: float = 0.09,
        width_pct: float = 0.03,
        hold_days: int = 5,
        vol_lookback: int = 20,
        rsi_entry: float = 40.0,
        max_spread_bps: float = 35.0,
        rate: float = 0.04,
    ):
        self.short_otm_pct = short_otm_pct
        self.width_pct = width_pct
        self.hold_days = hold_days
        self.vol_lookback = vol_lookback
        self.rsi_entry = rsi_entry
        self.max_spread_bps = max_spread_bps
        self.rate = rate
        self.executor = SpreadAwareExecutor(max_spread_bps=max_spread_bps)

    def _spread_value(self, spot, k_short, k_long, sigma, t):
        from pricing.black_scholes import BlackScholes, OptionContract

        short_p = BlackScholes(
            OptionContract(spot, k_short, self.rate, 0.0, sigma, t, "put")
        ).price()
        long_p = BlackScholes(
            OptionContract(spot, k_long, self.rate, 0.0, sigma, t, "put")
        ).price()
        return short_p - long_p  # credit spread of the spread

    def run(
        self,
        ohlc: pd.DataFrame,
        asset: str,
        horizon: str,
    ) -> List[Trade]:
        df = ohlc.dropna().copy()
        if len(df) < self.vol_lookback + self.hold_days + 5:
            return []
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        rets = close.pct_change()
        vol = rets.rolling(self.vol_lookback).std() * np.sqrt(252)
        rsi = _rsi(close, 14)
        trades: List[Trade] = []
        i = self.vol_lookback + 1
        while i < len(df) - self.hold_days - 1:
            if not (np.isfinite(vol.iloc[i]) and np.isfinite(rsi.iloc[i])):
                i += 1
                continue
            # Credit put spread on mild oversold / bounce
            if rsi.iloc[i] > self.rsi_entry:
                i += 1
                continue
            spot = float(close.iloc[i])
            k_short = spot * (1.0 - self.short_otm_pct)
            k_long = k_short * (1.0 - self.width_pct)
            sigma = max(float(vol.iloc[i]), 0.05)
            t_entry = max(self.hold_days / 252.0, 1.0 / 252.0)
            range_frac = max(float(high.iloc[i]) - float(low.iloc[i]), 0.0) / spot
            opt_spread = float(min(max(0.0015, 0.12 * range_frac), 0.005))
            ok, _, _ = self.executor.should_trade("sell", spot, opt_spread)
            if not ok:
                i += 1
                continue

            entry_mid = self._spread_value(spot, k_short, k_long, sigma, t_entry)
            if entry_mid <= 0:
                i += 1
                continue
            entry_credit = entry_mid * (1.0 - 0.5 * opt_spread)
            width = k_short - k_long
            max_loss = max(width - entry_credit, 1e-6)

            j = i + self.hold_days
            spot_exit = float(close.iloc[j])
            sigma_exit = float(vol.iloc[j]) if np.isfinite(vol.iloc[j]) else sigma
            t_exit = max(1.0 / 252.0, t_entry - self.hold_days / 252.0)
            exit_mid = self._spread_value(
                spot_exit, k_short, k_long, max(sigma_exit, 0.05), t_exit
            )
            exit_debit = max(exit_mid, 0.0) * (1.0 + 0.5 * opt_spread)
            # PnL for short credit spread
            pnl = entry_credit - exit_debit
            pnl = max(pnl, -max_loss)  # defined risk
            pnl_pct = pnl / max_loss  # return on capital at risk

            trades.append(
                Trade(
                    asset=f"{asset}_PUT_CR",
                    asset_class="options",
                    horizon=horizon,
                    entry_time=df.index[i],
                    exit_time=df.index[j],
                    side="sell",
                    entry_price=entry_credit,
                    exit_price=exit_debit,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    bars_held=self.hold_days,
                    reason_exit="hold_complete",
                )
            )
            i = j + 1
        return trades


def summarize_trades(trades: List[Trade]) -> BacktestResult:
    if not trades:
        return BacktestResult([], 0.0, 0, 0, 0.0, 0.0, 0.0)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades)
    gp = sum(t.pnl_pct for t in wins)
    gl = abs(sum(t.pnl_pct for t in losses))
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    exp = float(np.mean([t.pnl_pct for t in trades]))
    return BacktestResult(
        trades=trades,
        win_rate=win_rate,
        n_trades=len(trades),
        n_wins=len(wins),
        expectancy_pct=exp,
        profit_factor=float(pf) if np.isfinite(pf) else 999.0,
        total_pnl_pct=float(sum(t.pnl_pct for t in trades)),
    )
