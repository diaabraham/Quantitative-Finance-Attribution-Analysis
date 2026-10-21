"""Shared pytest fixtures, markers, and live-data helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    config.addinivalue_line("markers", "live: hits live network market-data APIs")
    config.addinivalue_line(
        "markers", "requires_key(name): skip unless env API key is set"
    )
    config.addinivalue_line("markers", "slow: longer multi-horizon / bulk fetches")
    config.addinivalue_line(
        "markers", "options_tape: live option chain/quote tape (yfinance)"
    )


def _env_key(name: str) -> str | None:
    v = os.getenv(name, "").strip()
    return v or None


def has_api_key(*names: str) -> bool:
    return any(_env_key(n) for n in names)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def api_keys_present() -> dict:
    return {
        "stooq": bool(_env_key("STOOQ_API_KEY")),
        "polygon": bool(_env_key("POLYGON_API_KEY")),
        "massive": bool(_env_key("MASSIVE_API_KEY") or _env_key("POLYGON_API_KEY")),
        "databento": bool(_env_key("DATABENTO_API_KEY")),
        "sec": bool(_env_key("SEC_API_KEY")),
    }


@pytest.fixture(scope="session")
def live_end() -> str:
    """Stable calendar end for live fetches (naive UTC date of run)."""
    import pandas as pd

    return (
        pd.Timestamp.now(tz="UTC")
        .tz_convert(None)
        .normalize()
        .strftime("%Y-%m-%d")
    )


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("requires_key")
    if marker is None:
        return
    names = marker.args or ()
    kwargs_names = marker.kwargs.get("names") or ()
    keys = tuple(names) + tuple(kwargs_names)
    if not keys:
        return
    if not has_api_key(*keys):
        pytest.skip(f"requires API key(s): {', '.join(keys)}")
