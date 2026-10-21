import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from data.data_manager import DataManager
from data.providers.base import ProviderError


class TestDataManager:
    @pytest.fixture
    def data_manager(self, tmp_path):
        return DataManager(
            data_dir=str(tmp_path / "raw"),
            cache_dir=str(tmp_path / "cache"),
            provider="yfinance",
            require_sec=False,
        )

    @pytest.fixture
    def sample_etfs(self):
        return {
            "Technology": {"symbol": "XLK", "cik": "0001067983"},
            "Financials": {"symbol": "XLF", "cik": "0001067983"},
            "Healthcare": {"symbol": "XLV", "cik": "0001067983"},
        }

    def test_get_sector_etfs(self, data_manager):
        etfs = data_manager.get_sector_etfs()
        assert isinstance(etfs, dict)
        assert len(etfs) > 0
        for sector, info in etfs.items():
            assert "symbol" in info
            assert "cik" in info

    def test_known_providers(self, data_manager):
        names = data_manager.known_providers()
        assert names == (
            "yfinance",
            "stooq",
            "massive",
            "polygon",
            "databento",
        )

    def test_fetch_historical_data_mocked(self, data_manager, sample_etfs):
        symbols = [info["symbol"] for info in sample_etfs.values()]
        idx = pd.bdate_range("2024-01-02", periods=5)
        fake = pd.DataFrame(
            {s: np.linspace(10, 12, len(idx)) for s in symbols},
            index=idx,
        )

        with patch.object(data_manager.provider, "fetch_closes", return_value=fake):
            data = data_manager.fetch_historical_data(
                symbols, "2024-01-02", "2024-01-10", use_cache=False
            )
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(symbol in data.columns for symbol in symbols)

    def test_fetch_sec_filings_without_key(self, data_manager, sample_etfs):
        cik = sample_etfs["Technology"]["cik"]
        filings = data_manager.fetch_sec_filings(cik, filing_type="10-K", limit=5)
        assert isinstance(filings, list)

    def test_extract_financial_metric(self, data_manager):
        test_content = """
        <table>
            <tr><td>Total Assets</td><td>$1,000,000</td></tr>
            <tr><td>Revenue</td><td>$500,000</td></tr>
        </table>
        """
        assert data_manager._parse_balance_sheet_metric(test_content) == 1000000.0
        assert data_manager._parse_income_statement_metric(test_content) == 500000.0

    def test_get_market_data_mocked(self, data_manager):
        idx = pd.bdate_range("2024-01-02", periods=5)
        series = pd.Series(np.linspace(100, 110, len(idx)), index=idx, name="^GSPC")
        with patch.object(data_manager.provider, "fetch_series", return_value=series):
            market_data = data_manager.get_market_data(
                start_date="2024-01-02", end_date="2024-01-10"
            )
        assert isinstance(market_data, pd.Series)
        assert not market_data.empty

    def test_get_risk_free_rate_fallback(self, data_manager):
        with patch.object(
            data_manager.provider,
            "fetch_series",
            side_effect=ProviderError("no tnx"),
        ):
            with patch(
                "data.data_manager.get_provider"
            ) as gp:
                yf = gp.return_value
                idx = pd.bdate_range("2024-01-02", periods=3)
                yf.fetch_series.return_value = pd.Series([4.2, 4.1, 4.0], index=idx)
                rate = data_manager.get_risk_free_rate()
        assert isinstance(rate, float)
        assert 0 <= rate <= 1

    def test_data_caching(self, data_manager, sample_etfs):
        symbols = [info["symbol"] for info in sample_etfs.values()]
        idx = pd.bdate_range("2024-01-02", periods=5)
        fake = pd.DataFrame(
            {s: np.linspace(10, 12, len(idx)) for s in symbols},
            index=idx,
        )
        with patch.object(
            data_manager.provider, "fetch_closes", return_value=fake
        ) as fetch:
            data1 = data_manager.fetch_historical_data(
                symbols, "2024-01-02", "2024-01-10", use_cache=True
            )
            data2 = data_manager.fetch_historical_data(
                symbols, "2024-01-02", "2024-01-10", use_cache=True
            )
        assert fetch.call_count == 1
        pd.testing.assert_frame_equal(data1, data2)

    def test_error_handling(self, data_manager):
        with patch.object(
            data_manager.provider,
            "fetch_closes",
            side_effect=ProviderError("No price data retrieved"),
        ):
            with pytest.raises(ValueError):
                data_manager.fetch_historical_data(
                    ["INVALID_SYMBOL"], "2020-01-01", "2020-01-02", use_cache=False
                )

        with pytest.raises(ValueError):
            data_manager.fetch_historical_data(
                ["XLK"], "2020-01-02", "2020-01-01", use_cache=False
            )

        filings = data_manager.fetch_sec_filings("INVALID_CIK")
        assert isinstance(filings, list)

    def test_metric_parsing_edge_cases(self, data_manager):
        assert data_manager._parse_balance_sheet_metric("") is None
        assert data_manager._parse_income_statement_metric("No numbers here") is None
        assert (
            data_manager._parse_cash_flow_metric(
                "<table><tr><td>Invalid</td></tr></table>"
            )
            is None
        )
        content = """
        <table>
            <tr><td>Total Assets</td><td>$1,000</td><td>$2,000</td></tr>
        </table>
        """
        assert data_manager._parse_balance_sheet_metric(content) == 1000.0

    def test_require_sec_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SEC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SEC_API_KEY"):
            DataManager(
                data_dir=str(tmp_path / "raw"),
                cache_dir=str(tmp_path / "cache"),
                require_sec=True,
            )
