"""Provider factory and auto-resolution helpers."""

from __future__ import annotations

from typing import Iterable

from data.providers.base import PriceProvider, ProviderError
from data.providers.databento_provider import DatabentoProvider
from data.providers.polygon_compatible import MassiveProvider, PolygonProvider
from data.providers.stooq_provider import StooqProvider
from data.providers.yfinance_provider import YFinanceProvider

# Preferred order for auto mode: free first, then paid vendors.
_DEFAULT_ORDER = (
    "yfinance",
    "stooq",
    "massive",
    "polygon",
    "databento",
)

_FACTORIES = {
    "yfinance": YFinanceProvider,
    "stooq": StooqProvider,
    "massive": MassiveProvider,
    "polygon": PolygonProvider,
    "databento": DatabentoProvider,
}

PROVIDER_NAMES = tuple(_FACTORIES.keys())


def get_provider(name: str, **kwargs) -> PriceProvider:
    key = name.strip().lower()
    if key not in _FACTORIES:
        raise ValueError(
            f"Unknown provider {name!r}. Choose one of: {', '.join(PROVIDER_NAMES)}"
        )
    return _FACTORIES[key](**kwargs)


def list_available_providers(order: Iterable[str] | None = None) -> list[PriceProvider]:
    names = tuple(order) if order is not None else _DEFAULT_ORDER
    providers: list[PriceProvider] = []
    for name in names:
        provider = get_provider(name)
        if provider.available():
            providers.append(provider)
    return providers


def resolve_provider(
    name: str | PriceProvider | None = "auto",
    *,
    fallback: bool = True,
) -> PriceProvider:
    """
    Resolve a provider by name.

    ``auto`` selects the first available provider in the default order.
    When ``fallback`` is True and a named paid provider is unavailable,
    falls through to the next available free provider.
    """
    if isinstance(name, PriceProvider):
        return name

    requested = (name or "auto").strip().lower()
    if requested == "auto":
        available = list_available_providers()
        if not available:
            raise ProviderError("No market data providers are available")
        return available[0]

    provider = get_provider(requested)
    if provider.available():
        return provider
    if not fallback:
        raise ProviderError(provider.unavailable_reason() or f"{requested}: unavailable")

    for candidate in list_available_providers():
        if candidate.name != provider.name:
            return candidate
    raise ProviderError(
        provider.unavailable_reason()
        or f"{requested}: unavailable and no fallback providers found"
    )
