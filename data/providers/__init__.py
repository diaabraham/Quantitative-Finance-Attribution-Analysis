"""Market-data provider registry."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Union

from .base import MarketDataProvider
from .databento_provider import DatabentoProvider
from .errors import MarketDataError, ProviderAuthError, ProviderConfigError
from .polygon_rest import MassiveProvider, PolygonProvider
from .stooq_provider import StooqProvider
from .yfinance_provider import YFinanceProvider

PROVIDERS = {
    "yfinance": YFinanceProvider,
    "stooq": StooqProvider,
    "polygon": PolygonProvider,
    "massive": MassiveProvider,
    "databento": DatabentoProvider,
}

DEFAULT_PROVIDER = "yfinance"
DEFAULT_FALLBACKS = ("stooq",)


def available_providers() -> List[str]:
    return list(PROVIDERS.keys())


def create_provider(name: Optional[str] = None, **kwargs) -> MarketDataProvider:
    """Instantiate a provider by name (env: MARKET_DATA_PROVIDER)."""
    key = (name or os.getenv("MARKET_DATA_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ProviderConfigError(
            f"Unknown market-data provider '{key}'. Choose from: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[key](**kwargs)


class FallbackProvider(MarketDataProvider):
    """Try providers in order until one returns data."""

    name = "fallback"

    def __init__(self, providers: Sequence[MarketDataProvider]):
        if not providers:
            raise ProviderConfigError("FallbackProvider requires at least one provider")
        self.providers = list(providers)
        self.name = "+".join(p.name for p in self.providers)

    def fetch_closes(self, symbols, start_date, end_date):
        errors = []
        for provider in self.providers:
            try:
                return provider.fetch_closes(symbols, start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider.name}: {exc}")
                continue
        raise MarketDataError(
            "All market-data providers failed:\n- " + "\n- ".join(errors)
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "providers": [p.describe() for p in self.providers],
        }


def build_provider(
    name: Optional[str] = None,
    fallbacks: Optional[Union[str, Iterable[str]]] = None,
    **kwargs,
) -> MarketDataProvider:
    """
    Build primary provider with optional fallbacks.

    Env:
      MARKET_DATA_PROVIDER   primary (default yfinance)
      MARKET_DATA_FALLBACKS  comma-separated list (default: stooq)
    """
    primary = create_provider(name, **kwargs)
    if fallbacks is None:
        raw = os.getenv("MARKET_DATA_FALLBACKS", ",".join(DEFAULT_FALLBACKS))
        fallback_names = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(fallbacks, str):
        fallback_names = [x.strip() for x in fallbacks.split(",") if x.strip()]
    else:
        fallback_names = list(fallbacks)

    chain = [primary]
    for fb in fallback_names:
        if fb.lower() == primary.name:
            continue
        try:
            chain.append(create_provider(fb))
        except ProviderAuthError:
            # Skip providers that need keys we don't have
            continue
        except Exception:
            continue

    if len(chain) == 1:
        return primary
    return FallbackProvider(chain)


__all__ = [
    "PROVIDERS",
    "available_providers",
    "build_provider",
    "create_provider",
    "FallbackProvider",
    "YFinanceProvider",
    "StooqProvider",
    "PolygonProvider",
    "MassiveProvider",
    "DatabentoProvider",
    "MarketDataProvider",
    "MarketDataError",
    "ProviderAuthError",
    "ProviderConfigError",
]
