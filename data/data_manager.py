import pandas as pd
import numpy as np
import os
from sec_api import ExtractorApi, QueryApi
import yfinance as yf
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import requests
import re

class DataManager:
    def __init__(self, data_dir='data/raw', cache_dir='data/cache'):
        """
        Initialize the Data Manager
        
        Parameters:
        -----------
        data_dir : str
            Directory to store raw data
        cache_dir : str
            Directory to store processed/cached data
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.ensure_directories()
        
        # Load SEC API key from environment
        load_dotenv()
        self.sec_api_key = os.getenv('SEC_API_KEY')
        if not self.sec_api_key:
            raise ValueError("SEC_API_KEY not found in environment variables")
            
        self.extractor_api = ExtractorApi(self.sec_api_key)
        self.query_api = QueryApi(self.sec_api_key)
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'sec'), exist_ok=True)
        
    def get_sector_etfs(self):
        """
        Get sector ETF information from SEC filings
        This implementation uses both hardcoded data and SEC API for additional information
        """
        # Base ETF information
        sector_etfs = {
            'Technology': {'symbol': 'XLK', 'cik': '0001067983'},
            'Financials': {'symbol': 'XLF', 'cik': '0001067983'},
            'Healthcare': {'symbol': 'XLV', 'cik': '0001067983'},
            'Consumer Discretionary': {'symbol': 'XLY', 'cik': '0001067983'},
            'Consumer Staples': {'symbol': 'XLP', 'cik': '0001067983'},
            'Energy': {'symbol': 'XLE', 'cik': '0001067983'},
            'Industrials': {'symbol': 'XLI', 'cik': '0001067983'},
            'Materials': {'symbol': 'XLB', 'cik': '0001067983'},
            'Utilities': {'symbol': 'XLU', 'cik': '0001067983'},
            'Real Estate': {'symbol': 'XLRE', 'cik': '0001067983'}
        }
        
        # Fetch additional information from SEC
        for sector, info in sector_etfs.items():
            try:
                # Get latest 10-K filing
                filing = self.fetch_sec_filings(info['cik'], '10-K', limit=1)
                if filing:
                    # Extract relevant information
                    info['latest_filing'] = filing[0]['filingDate']
                    info['total_assets'] = self.extract_financial_metric(filing[0]['linkToHtml'], 'totalAssets')
                    info['expense_ratio'] = self.extract_financial_metric(filing[0]['linkToHtml'], 'expenseRatio')
            except Exception as e:
                print(f"Error fetching SEC data for {sector}: {e}")
                
        return sector_etfs
    
    def fetch_sec_filings(self, cik, filing_type='10-K', start_date=None, end_date=None, limit=10):
        """
        Fetch SEC filings for a company using the SEC API
        
        Parameters:
        -----------
        cik : str
            Central Index Key of the company
        filing_type : str
            Type of filing to fetch
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        limit : int
            Maximum number of filings to fetch
        """
        query = {
            "query": {
                "query_string": {
                    "query": f"cik:{cik} AND formType:\"{filing_type}\""
                }
            },
            "from": 0,
            "size": limit,
            "sort": [{"filingDate": {"order": "desc"}}]
        }
        
        if start_date:
            date_range = f"[{start_date} TO {end_date or '*'}]"
            query["query"]["query_string"]["query"] += f" AND filingDate:{date_range}"
            
        try:
            response = self.query_api.get_filings(query)
            return response['filings']
        except Exception as e:
            print(f"Error fetching SEC filings: {e}")
            return []
    
    def extract_financial_metric(self, filing_url, metric_name):
        """
        Extract specific financial metrics from SEC filings with enhanced parsing
        
        Parameters:
        -----------
        filing_url : str
            URL to the filing
        metric_name : str
            Name of the metric to extract
        """
        try:
            # Use the SEC API extractor to get specific sections
            section = self.extractor_api.get_section(filing_url, metric_name, "text")
            
            # Enhanced parsing for different metric types
            if metric_name in ['totalAssets', 'totalLiabilities', 'totalEquity']:
                return self._parse_balance_sheet_metric(section)
            elif metric_name in ['revenue', 'netIncome', 'operatingIncome']:
                return self._parse_income_statement_metric(section)
            elif metric_name in ['operatingCashFlow', 'investingCashFlow', 'financingCashFlow']:
                return self._parse_cash_flow_metric(section)
            elif metric_name == 'expenseRatio':
                return self._parse_expense_ratio(section)
            else:
                return self._parse_general_metric(section)
        except Exception as e:
            print(f"Error extracting metric {metric_name}: {e}")
            return None
    
    def _parse_balance_sheet_metric(self, text):
        """Parse balance sheet metrics with enhanced accuracy"""
        try:
            # Look for the metric in a table format
            table_pattern = r'<table.*?>(.*?)</table>'
            tables = re.findall(table_pattern, text, re.DOTALL)
            
            for table in tables:
                # Look for rows containing the metric
                rows = re.findall(r'<tr.*?>(.*?)</tr>', table, re.DOTALL)
                for row in rows:
                    cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
                    if len(cells) >= 2:
                        metric_name = re.sub(r'<.*?>', '', cells[0]).strip()
                        if any(keyword in metric_name.lower() for keyword in ['total assets', 'total liabilities', 'total equity']):
                            value = re.sub(r'[^\d.-]', '', cells[1])
                            return float(value) if value else None
            return None
        except Exception as e:
            print(f"Error parsing balance sheet metric: {e}")
            return None
    
    def _parse_income_statement_metric(self, text):
        """Parse income statement metrics with enhanced accuracy"""
        try:
            # Look for the metric in a table format
            table_pattern = r'<table.*?>(.*?)</table>'
            tables = re.findall(table_pattern, text, re.DOTALL)
            
            for table in tables:
                # Look for rows containing the metric
                rows = re.findall(r'<tr.*?>(.*?)</tr>', table, re.DOTALL)
                for row in rows:
                    cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
                    if len(cells) >= 2:
                        metric_name = re.sub(r'<.*?>', '', cells[0]).strip()
                        if any(keyword in metric_name.lower() for keyword in ['revenue', 'net income', 'operating income']):
                            value = re.sub(r'[^\d.-]', '', cells[1])
                            return float(value) if value else None
            return None
        except Exception as e:
            print(f"Error parsing income statement metric: {e}")
            return None
    
    def _parse_cash_flow_metric(self, text):
        """Parse cash flow metrics with enhanced accuracy"""
        try:
            # Look for the metric in a table format
            table_pattern = r'<table.*?>(.*?)</table>'
            tables = re.findall(table_pattern, text, re.DOTALL)
            
            for table in tables:
                # Look for rows containing the metric
                rows = re.findall(r'<tr.*?>(.*?)</tr>', table, re.DOTALL)
                for row in rows:
                    cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
                    if len(cells) >= 2:
                        metric_name = re.sub(r'<.*?>', '', cells[0]).strip()
                        if any(keyword in metric_name.lower() for keyword in ['operating cash', 'investing cash', 'financing cash']):
                            value = re.sub(r'[^\d.-]', '', cells[1])
                            return float(value) if value else None
            return None
        except Exception as e:
            print(f"Error parsing cash flow metric: {e}")
            return None
    
    def _parse_expense_ratio(self, text):
        """Parse expense ratio with enhanced accuracy"""
        try:
            # Look for expense ratio in text
            pattern = r'expense\s*ratio\s*[:=]?\s*([\d.]+)%?'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100
            return None
        except Exception as e:
            print(f"Error parsing expense ratio: {e}")
            return None
    
    def _parse_general_metric(self, text):
        """Parse general metrics with enhanced accuracy"""
        try:
            # Look for numbers in the text
            numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', text)
            if numbers:
                # Clean and convert the first number found
                number = numbers[0].replace('$', '').replace(',', '').replace('%', '')
                return float(number)
            return None
        except Exception as e:
            print(f"Error parsing general metric: {e}")
            return None
    
    def fetch_historical_data(self, symbols, start_date, end_date):
        """
        Fetch historical price data and cache it
        
        Parameters:
        -----------
        symbols : list
            List of symbols to fetch
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        """
        cache_file = os.path.join(self.cache_dir, f'price_data_{start_date}_{end_date}.csv')
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Fetch new data
        data = {}
        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date)
                data[symbol] = df['Adj Close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        df = pd.DataFrame(data)
        df.to_csv(cache_file)
        return df
    
    def get_risk_free_rate(self):
        """Fetch risk-free rate data (e.g., 10-year Treasury yield)"""
        try:
            df = yf.download('^TNX', period='1y')['Adj Close']
            return df.iloc[-1] / 100  # Convert to decimal
        except Exception as e:
            print(f"Error fetching risk-free rate: {e}")
            return 0.02  # Default to 2% if fetch fails
    
    def get_market_data(self, symbol='^GSPC', start_date=None, end_date=None):
        """Fetch market benchmark data"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            return df['Adj Close']
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None 