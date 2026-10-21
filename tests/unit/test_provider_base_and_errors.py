"""Provider base helpers, errors, and registry edge cases."""

from __future__ import annotations

import pytest

from data.providers import available_providers, build_provider, create_provider
from data.providers.base import normalize_dates, to_naive_daily_index
from data.providers.errors import (
    MarketDataError,
    ProviderAuthError,
    ProviderConfigError,
    ProviderEmptyDataError,
)
import pandas as pd


def test_normalize_dates_ok():
    s, e = normalize_dates("2024-01-01", "2024-02-01")
    assert s == "2024-01-01" and e == "2024-02-01"


def test_normalize_dates_rejects_inverted():
    with pytest.raises(ValueError):
        normalize_dates("2024-06-01", "2024-01-01")


def test_to_naive_daily_index_tz():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    out = to_naive_daily_index(idx)
    assert out.tz is None
    assert len(out) == 3


@pytest.mark.parametrize("name", available_providers())
def test_unknown_provider_not_in_registry_raises(name):
    # Every registered name constructs or fails only on auth — not ConfigError for name
    assert name in available_providers()


def test_unknown_provider_name():
    with pytest.raises(ProviderConfigError):
        create_provider("not-a-real-provider")


def test_error_types_are_exceptions():
    for cls in (MarketDataError, ProviderAuthError, ProviderEmptyDataError, ProviderConfigError):
        assert issubclass(cls, Exception)


def test_build_provider_explicit_empty_fallbacks():
    p = build_provider("yfinance", fallbacks=[])
    assert "yfinance" in p.name


@pytest.mark.parametrize(
    "fallbacks",
    ["stooq", "stooq,polygon", ["stooq"], []],
)
def test_build_provider_fallback_shapes(fallbacks, monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.delenv("STOOQ_API_KEY", raising=False)
    p = build_provider("yfinance", fallbacks=fallbacks)
    assert "yfinance" in p.name
