"""Unit tests for market-data providers (mocked network / SDKs)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.providers.base import ProviderError, normalize_frame
from data.providers.databento_provider import DatabentoProvider
from data.providers.polygon_compatible import MassiveProvider, PolygonProvider
from data.providers.registry import get_provider, list_available_providers, resolve_provider
from data.providers.stooq_provider import StooqProvider, to_stooq_symbol
from data.providers.yfinance_provider import YFinanceProvider, _extract_close


def _sample_ohlc(days=5, start="2024-01-02"):
    idx = pd.bdate_range(start=start, periods=days)
    return pd.DataFrame(
        {
            "Open": range(100, 100 + days),
            "High": range(101, 101 + days),
            "Low": range(99, 99 + days),
            "Close": [10.0 + i for i in range(days)],
            "Volume": [1000] * days,
        },
        index=idx,
    )


class TestHelpers:
    def test_to_stooq_symbol(self):
        assert to_stooq_symbol("XLK") == "xlk.us"
        assert to_stooq_symbol("^GSPC") == "^spx"
        assert to_stooq_symbol("AAPL.US") == "aapl.us"

    def test_extract_close_multiindex(self):
        idx = pd.bdate_range("2024-01-02", periods=3)
        cols = pd.MultiIndex.from_product([["Close", "Volume"], ["XLK"]])
        df = pd.DataFrame(
            [[10.0, 1], [11.0, 1], [12.0, 1]],
            index=idx,
            columns=cols,
        )
        series = _extract_close(df, "XLK")
        assert list(series.values) == [10.0, 11.0, 12.0]

    def test_normalize_frame_rejects_empty(self):
        with pytest.raises(ProviderError):
            normalize_frame(pd.DataFrame())


class TestRegistry:
    def test_known_providers(self):
        for name in ("yfinance", "stooq", "massive", "polygon", "databento"):
            assert get_provider(name).name == name

    def test_resolve_auto(self):
        provider = resolve_provider("auto")
        assert provider.available()
        assert provider.name in {"yfinance", "stooq"}

    def test_list_available_includes_free(self):
        names = [p.name for p in list_available_providers()]
        assert "yfinance" in names
        assert "stooq" in names


class TestYFinanceProvider:
    def test_fetch_closes_mocked(self):
        provider = YFinanceProvider()
        ohlc = _sample_ohlc()
        with patch("data.providers.yfinance_provider.yf.download", return_value=ohlc):
            frame = provider.fetch_closes(["XLK"], "2024-01-02", "2024-01-10")
        assert list(frame.columns) == ["XLK"]
        assert len(frame) == 5
        assert frame["XLK"].iloc[0] == 10.0

    def test_fetch_closes_all_fail(self):
        provider = YFinanceProvider()
        with patch(
            "data.providers.yfinance_provider.yf.download",
            return_value=pd.DataFrame(),
        ):
            with pytest.raises(ProviderError):
                provider.fetch_closes(["BAD"], "2024-01-02", "2024-01-10")


class TestStooqProvider:
    def test_fetch_closes_mocked(self):
        csv = "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,10.5,100\n2024-01-03,1,1,1,11.0,100\n"
        mock_resp = MagicMock(status_code=200, text=csv)
        session = MagicMock()
        session.get.return_value = mock_resp
        provider = StooqProvider(session=session)
        frame = provider.fetch_closes(["XLK"], "2024-01-02", "2024-01-05")
        assert frame.loc["2024-01-02", "XLK"] == 10.5
        assert session.get.called

    def test_api_key_required_payload(self):
        mock_resp = MagicMock(
            status_code=200,
            text="Get your apikey:\n1. Open https://stooq.com/q/d/?s=AAPL.US&get_apikey",
        )
        session = MagicMock()
        session.get.return_value = mock_resp
        provider = StooqProvider(session=session)
        with pytest.raises(ProviderError, match="API key required"):
            provider.fetch_closes(["AAPL"], "2024-01-02", "2024-01-05")


class TestPolygonCompatible:
    def _bar(self, ts_ms, close):
        bar = MagicMock()
        bar.timestamp = ts_ms
        bar.close = close
        return bar

    def test_massive_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        assert MassiveProvider().available() is False

    def test_polygon_fetch_mocked(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        provider = PolygonProvider(api_key="test-key")
        client = MagicMock()
        client.list_aggs.return_value = [
            self._bar(1704153600000, 100.0),  # 2024-01-02 approx
            self._bar(1704240000000, 101.5),
        ]
        provider._client = client
        frame = provider.fetch_closes(["AAPL"], "2024-01-02", "2024-01-05")
        assert list(frame.columns) == ["AAPL"]
        assert len(frame) == 2
        assert frame["AAPL"].iloc[-1] == 101.5

    def test_massive_uses_massive_host(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        provider = MassiveProvider(api_key="test-key")
        assert provider.default_base == "https://api.massive.com"
        assert provider.sdk_import == "massive"


class TestDatabentoProvider:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        assert DatabentoProvider().available() is False

    def test_fetch_closes_mocked(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "db-test-key")
        provider = DatabentoProvider(api_key="db-test-key")
        idx = pd.to_datetime(
            ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z", "2024-01-02T00:00:00Z"]
        )
        raw = pd.DataFrame(
            {
                "symbol": ["XLK", "XLK", "XLF"],
                "close": [10.0, 11.0, 20.0],
                "open": [1, 1, 1],
            },
            index=idx,
        )
        store = MagicMock()
        store.to_df.return_value = raw
        client = MagicMock()
        client.timeseries.get_range.return_value = store
        provider._client = client

        frame = provider.fetch_closes(["XLK", "XLF"], "2024-01-02", "2024-01-04")
        assert set(frame.columns) == {"XLK", "XLF"}
        assert frame.loc[pd.Timestamp("2024-01-03"), "XLK"] == 11.0
