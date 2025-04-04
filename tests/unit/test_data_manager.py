import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.data_manager import DataManager

class TestDataManager:
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    @pytest.fixture
    def sample_etfs(self):
        return {
            'Technology': {'symbol': 'XLK', 'cik': '0001067983'},
            'Financials': {'symbol': 'XLF', 'cik': '0001067983'},
            'Healthcare': {'symbol': 'XLV', 'cik': '0001067983'}
        }
    
    def test_get_sector_etfs(self, data_manager):
        etfs = data_manager.get_sector_etfs()
        assert isinstance(etfs, dict)
        assert len(etfs) > 0
        for sector, info in etfs.items():
            assert 'symbol' in info
            assert 'cik' in info
    
    def test_fetch_historical_data(self, data_manager, sample_etfs):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        symbols = [info['symbol'] for info in sample_etfs.values()]
        
        data = data_manager.fetch_historical_data(symbols, start_date, end_date)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(symbol in data.columns for symbol in symbols)
    
    def test_fetch_sec_filings(self, data_manager, sample_etfs):
        cik = sample_etfs['Technology']['cik']
        filings = data_manager.fetch_sec_filings(cik, filing_type='10-K', limit=5)
        assert isinstance(filings, list)
        if filings:  # If API key is available
            assert len(filings) <= 5
            assert all('filingDate' in filing for filing in filings)
    
    def test_extract_financial_metric(self, data_manager):
        # Mock filing URL and content
        test_url = "https://example.com/filing"
        test_content = """
        <table>
            <tr><td>Total Assets</td><td>$1,000,000</td></tr>
            <tr><td>Revenue</td><td>$500,000</td></tr>
        </table>
        """
        
        # Test balance sheet metric
        result = data_manager._parse_balance_sheet_metric(test_content)
        assert result == 1000000.0
        
        # Test income statement metric
        result = data_manager._parse_income_statement_metric(test_content)
        assert result == 500000.0
    
    def test_get_market_data(self, data_manager):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        market_data = data_manager.get_market_data(start_date=start_date, end_date=end_date)
        assert isinstance(market_data, pd.Series)
        assert not market_data.empty
    
    def test_get_risk_free_rate(self, data_manager):
        rate = data_manager.get_risk_free_rate()
        assert isinstance(rate, float)
        assert 0 <= rate <= 1
    
    def test_data_caching(self, data_manager, sample_etfs):
        # Test that data is properly cached
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        symbols = [info['symbol'] for info in sample_etfs.values()]
        
        # First fetch (should download)
        data1 = data_manager.fetch_historical_data(symbols, start_date, end_date)
        
        # Second fetch (should use cache)
        data2 = data_manager.fetch_historical_data(symbols, start_date, end_date)
        
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_error_handling(self, data_manager):
        # Test invalid symbol
        with pytest.raises(Exception):
            data_manager.fetch_historical_data(['INVALID_SYMBOL'], '2020-01-01', '2020-01-02')
        
        # Test invalid date range
        with pytest.raises(Exception):
            data_manager.fetch_historical_data(['XLK'], '2020-01-02', '2020-01-01')
        
        # Test invalid CIK
        with pytest.raises(Exception):
            data_manager.fetch_sec_filings('INVALID_CIK')
    
    def test_metric_parsing_edge_cases(self, data_manager):
        # Test empty content
        assert data_manager._parse_balance_sheet_metric("") is None
        
        # Test content without numbers
        assert data_manager._parse_income_statement_metric("No numbers here") is None
        
        # Test content with invalid format
        assert data_manager._parse_cash_flow_metric("<table><tr><td>Invalid</td></tr></table>") is None
        
        # Test content with multiple numbers
        content = """
        <table>
            <tr><td>Total Assets</td><td>$1,000</td><td>$2,000</td></tr>
        </table>
        """
        assert data_manager._parse_balance_sheet_metric(content) == 1000.0 