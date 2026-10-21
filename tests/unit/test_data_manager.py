import pytest
import pandas as pd
from datetime import datetime, timedelta
from data.data_manager import DataManager
from data.providers import YFinanceProvider


class TestDataManager:
    @pytest.fixture
    def data_manager(self):
        return DataManager(provider=YFinanceProvider())

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

    def test_fetch_historical_data(self, data_manager, sample_etfs):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        symbols = [info["symbol"] for info in sample_etfs.values()]

        data = data_manager.fetch_historical_data(
            symbols,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            use_cache=False,
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
        result = data_manager._parse_balance_sheet_metric(test_content)
        assert result == 1000000.0
        result = data_manager._parse_income_statement_metric(test_content)
        assert result == 500000.0

    def test_get_market_data(self, data_manager):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        market_data = data_manager.get_market_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        assert isinstance(market_data, pd.Series)
        assert not market_data.empty

    def test_get_risk_free_rate(self, data_manager):
        rate = data_manager.get_risk_free_rate()
        assert isinstance(rate, float)
        assert 0 <= rate <= 1

    def test_data_caching(self, data_manager, sample_etfs, tmp_path):
        data_manager.cache_dir = str(tmp_path)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        symbols = [info["symbol"] for info in sample_etfs.values()]
        start_s = start_date.strftime("%Y-%m-%d")
        end_s = end_date.strftime("%Y-%m-%d")

        data1 = data_manager.fetch_historical_data(symbols, start_s, end_s)
        data2 = data_manager.fetch_historical_data(symbols, start_s, end_s)
        # Cache round-trip may drop freq metadata; compare values/columns/index.
        pd.testing.assert_frame_equal(
            data1, data2, check_freq=False, check_exact=False, rtol=1e-10
        )

    def test_error_handling(self, data_manager):
        with pytest.raises((ValueError, Exception)):
            data_manager.fetch_historical_data(
                ["INVALID_SYMBOL_ZZZ"], "2020-01-01", "2020-01-02", use_cache=False
            )

        with pytest.raises(ValueError):
            data_manager.fetch_historical_data(["XLK"], "2020-01-02", "2020-01-01")

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
