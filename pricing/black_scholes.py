"""Black–Scholes–Merton pricing and Greeks."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Literal, Optional

from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionContract:
    spot: float
    strike: float
    rate: float
    dividend_yield: float
    volatility: float
    time_to_expiry: float  # in years
    option_type: OptionType = "call"


class BlackScholes:
    """European option pricing under Black–Scholes–Merton."""

    def __init__(self, contract: OptionContract):
        if contract.spot <= 0 or contract.strike <= 0:
            raise ValueError("spot and strike must be positive")
        if contract.time_to_expiry < 0:
            raise ValueError("time_to_expiry must be non-negative")
        if contract.volatility < 0:
            raise ValueError("volatility must be non-negative")
        self.c = contract

    def _d1_d2(self) -> tuple[float, float]:
        c = self.c
        if c.time_to_expiry == 0 or c.volatility == 0:
            # Degenerate: intrinsic only
            forward = c.spot * exp((c.rate - c.dividend_yield) * c.time_to_expiry)
            intrinsic_sign = 1.0 if forward >= c.strike else -1.0
            d1 = 50.0 * intrinsic_sign
            return d1, d1
        vol_sqrt_t = c.volatility * sqrt(c.time_to_expiry)
        d1 = (
            log(c.spot / c.strike)
            + (c.rate - c.dividend_yield + 0.5 * c.volatility**2) * c.time_to_expiry
        ) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return d1, d2

    def price(self) -> float:
        c = self.c
        if c.time_to_expiry == 0:
            if c.option_type == "call":
                return max(c.spot - c.strike, 0.0)
            return max(c.strike - c.spot, 0.0)

        d1, d2 = self._d1_d2()
        df_r = exp(-c.rate * c.time_to_expiry)
        df_q = exp(-c.dividend_yield * c.time_to_expiry)
        if c.option_type == "call":
            return c.spot * df_q * norm.cdf(d1) - c.strike * df_r * norm.cdf(d2)
        return c.strike * df_r * norm.cdf(-d2) - c.spot * df_q * norm.cdf(-d1)

    def greeks(self) -> dict:
        c = self.c
        if c.time_to_expiry == 0:
            intrinsic_call = 1.0 if c.spot > c.strike else 0.0
            delta = intrinsic_call if c.option_type == "call" else intrinsic_call - 1.0
            return {
                "delta": delta,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }

        d1, d2 = self._d1_d2()
        df_r = exp(-c.rate * c.time_to_expiry)
        df_q = exp(-c.dividend_yield * c.time_to_expiry)
        pdf_d1 = norm.pdf(d1)
        sqrt_t = sqrt(c.time_to_expiry)

        gamma = df_q * pdf_d1 / (c.spot * c.volatility * sqrt_t)
        vega = c.spot * df_q * pdf_d1 * sqrt_t / 100.0  # per 1 vol point

        if c.option_type == "call":
            delta = df_q * norm.cdf(d1)
            theta = (
                -c.spot * df_q * pdf_d1 * c.volatility / (2 * sqrt_t)
                - c.rate * c.strike * df_r * norm.cdf(d2)
                + c.dividend_yield * c.spot * df_q * norm.cdf(d1)
            ) / 365.0
            rho = c.strike * c.time_to_expiry * df_r * norm.cdf(d2) / 100.0
        else:
            delta = -df_q * norm.cdf(-d1)
            theta = (
                -c.spot * df_q * pdf_d1 * c.volatility / (2 * sqrt_t)
                + c.rate * c.strike * df_r * norm.cdf(-d2)
                - c.dividend_yield * c.spot * df_q * norm.cdf(-d1)
            ) / 365.0
            rho = -c.strike * c.time_to_expiry * df_r * norm.cdf(-d2) / 100.0

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    time_to_expiry: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """Newton–Raphson implied vol; returns None if it fails to converge."""
    if market_price <= 0 or time_to_expiry <= 0:
        return None
    vol = 0.2
    for _ in range(max_iter):
        model = BlackScholes(
            OptionContract(
                spot, strike, rate, dividend_yield, vol, time_to_expiry, option_type
            )
        )
        price = model.price()
        vega = model.greeks()["vega"] * 100.0  # back to per unit vol
        diff = price - market_price
        if abs(diff) < tol:
            return float(vol)
        if vega <= 1e-12:
            break
        vol = max(1e-6, vol - diff / vega)
    return None
