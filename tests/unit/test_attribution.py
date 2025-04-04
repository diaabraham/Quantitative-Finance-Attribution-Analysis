import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from attribution.brinson import BrinsonAttribution
from attribution.factor_attribution import FactorAttribution
from attribution.risk_attribution import RiskAttribution
from attribution.transaction_cost import TransactionCostAttribution
from attribution import AttributionAnalysis

@pytest.fixture
def sample_returns():
    """Create sample returns data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    portfolio_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)),
                                index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0003, 0.008, len(dates)),
                                index=dates)
    return portfolio_returns, benchmark_returns

@pytest.fixture
def sample_factor_returns():
    """Create sample factor returns data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    factors = ['market', 'size', 'value', 'momentum']
    factor_returns = pd.DataFrame(
        np.random.normal(0.0002, 0.005, (len(dates), len(factors))),
        index=dates,
        columns=factors
    )
    return factor_returns

@pytest.fixture
def sample_trades():
    """Create sample trade data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    trades = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['AAPL'] * len(dates),
        'quantity': np.random.randint(-100, 100, len(dates)),
        'price': np.random.normal(150, 10, len(dates)),
        'side': np.where(np.random.randint(0, 2, len(dates)) == 1, 'BUY', 'SELL')
    })
    return trades

@pytest.fixture
def sample_market_data():
    """Create sample market data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    market_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['AAPL'] * len(dates),
        'open': np.random.normal(150, 10, len(dates)),
        'high': np.random.normal(155, 10, len(dates)),
        'low': np.random.normal(145, 10, len(dates)),
        'close': np.random.normal(152, 10, len(dates)),
        'volume': np.random.randint(1000000, 2000000, len(dates))
    })
    return market_data

class TestBrinsonAttribution:
    """Test Brinson attribution"""
    
    def test_initialization(self, sample_returns):
        """Test initialization"""
        portfolio_returns, benchmark_returns = sample_returns
        brinson = BrinsonAttribution(
            portfolio_returns,
            benchmark_returns,
            pd.Series(1.0, index=portfolio_returns.index),
            pd.Series(1.0, index=benchmark_returns.index)
        )
        assert isinstance(brinson, BrinsonAttribution)
    
    def test_calculate_attribution(self, sample_returns):
        """Test attribution calculation"""
        portfolio_returns, benchmark_returns = sample_returns
        brinson = BrinsonAttribution(
            portfolio_returns,
            benchmark_returns,
            pd.Series(1.0, index=portfolio_returns.index),
            pd.Series(1.0, index=benchmark_returns.index)
        )
        attribution = brinson.calculate_attribution()
        assert isinstance(attribution, dict)
        assert 'allocation' in attribution
        assert 'selection' in attribution
        assert 'interaction' in attribution
        assert 'total' in attribution
    
    def test_calculate_rolling_attribution(self, sample_returns):
        """Test rolling attribution calculation"""
        portfolio_returns, benchmark_returns = sample_returns
        brinson = BrinsonAttribution(
            portfolio_returns,
            benchmark_returns,
            pd.Series(1.0, index=portfolio_returns.index),
            pd.Series(1.0, index=benchmark_returns.index)
        )
        rolling_attribution = brinson.calculate_rolling_attribution(window=20)
        assert isinstance(rolling_attribution, pd.DataFrame)
        assert len(rolling_attribution) > 0

class TestFactorAttribution:
    """Test factor attribution"""
    
    def test_initialization(self, sample_returns, sample_factor_returns):
        """Test initialization"""
        portfolio_returns, _ = sample_returns
        factor = FactorAttribution(portfolio_returns, sample_factor_returns)
        assert isinstance(factor, FactorAttribution)
    
    def test_estimate_factor_exposures(self, sample_returns, sample_factor_returns):
        """Test factor exposure estimation"""
        portfolio_returns, _ = sample_returns
        factor = FactorAttribution(portfolio_returns, sample_factor_returns)
        exposures = factor.estimate_factor_exposures()
        assert isinstance(exposures, pd.DataFrame)
        assert len(exposures) > 0
    
    def test_calculate_factor_contributions(self, sample_returns, sample_factor_returns):
        """Test factor contribution calculation"""
        portfolio_returns, _ = sample_returns
        factor = FactorAttribution(portfolio_returns, sample_factor_returns)
        exposures = factor.estimate_factor_exposures()
        contributions = factor.calculate_factor_contributions(exposures)
        assert isinstance(contributions, pd.DataFrame)
        assert len(contributions) > 0

class TestRiskAttribution:
    """Test risk attribution"""
    
    def test_initialization(self, sample_returns, sample_factor_returns):
        """Test initialization"""
        portfolio_returns, _ = sample_returns
        risk = RiskAttribution(
            portfolio_returns,
            sample_factor_returns,
            pd.Series(1.0, index=portfolio_returns.index),
            pd.DataFrame(1.0, index=portfolio_returns.index,
                        columns=sample_factor_returns.columns)
        )
        assert isinstance(risk, RiskAttribution)
    
    def test_calculate_marginal_contributions(self, sample_returns, sample_factor_returns):
        """Test marginal contribution calculation"""
        portfolio_returns, _ = sample_returns
        risk = RiskAttribution(
            portfolio_returns,
            sample_factor_returns,
            pd.Series(1.0, index=portfolio_returns.index),
            pd.DataFrame(1.0, index=portfolio_returns.index,
                        columns=sample_factor_returns.columns)
        )
        marginal_contrib = risk.calculate_marginal_contributions()
        assert isinstance(marginal_contrib, pd.Series)
        assert len(marginal_contrib) > 0
    
    def test_calculate_factor_risk_contributions(self, sample_returns, sample_factor_returns):
        """Test factor risk contribution calculation"""
        portfolio_returns, _ = sample_returns
        risk = RiskAttribution(
            portfolio_returns,
            sample_factor_returns,
            pd.Series(1.0, index=portfolio_returns.index),
            pd.DataFrame(1.0, index=portfolio_returns.index,
                        columns=sample_factor_returns.columns)
        )
        factor_contrib = risk.calculate_factor_risk_contributions()
        assert isinstance(factor_contrib, pd.Series)
        assert len(factor_contrib) > 0

class TestTransactionCostAttribution:
    """Test transaction cost attribution"""
    
    def test_initialization(self, sample_trades, sample_market_data):
        """Test initialization"""
        cost = TransactionCostAttribution(sample_trades, sample_market_data)
        assert isinstance(cost, TransactionCostAttribution)
    
    def test_calculate_total_costs(self, sample_trades, sample_market_data):
        """Test total cost calculation"""
        cost = TransactionCostAttribution(sample_trades, sample_market_data)
        total_costs = cost.calculate_total_costs()
        assert isinstance(total_costs, pd.DataFrame)
        assert len(total_costs) > 0
    
    def test_calculate_implementation_shortfall(self, sample_trades, sample_market_data):
        """Test implementation shortfall calculation"""
        cost = TransactionCostAttribution(sample_trades, sample_market_data)
        shortfall = cost.calculate_implementation_shortfall()
        assert isinstance(shortfall, pd.DataFrame)
        assert len(shortfall) > 0

class TestAttributionAnalysis:
    """Test main attribution analysis"""
    
    def test_initialization(self, sample_returns, sample_factor_returns,
                          sample_trades, sample_market_data):
        """Test initialization"""
        portfolio_returns, benchmark_returns = sample_returns
        analysis = AttributionAnalysis(
            portfolio_returns,
            benchmark_returns,
            sample_factor_returns,
            sample_trades,
            sample_market_data
        )
        assert isinstance(analysis, AttributionAnalysis)
    
    def test_run_attribution_analysis(self, sample_returns, sample_factor_returns,
                                    sample_trades, sample_market_data):
        """Test complete attribution analysis"""
        portfolio_returns, benchmark_returns = sample_returns
        analysis = AttributionAnalysis(
            portfolio_returns,
            benchmark_returns,
            sample_factor_returns,
            sample_trades,
            sample_market_data
        )
        results = analysis.run_attribution_analysis()
        assert isinstance(results, dict)
        assert 'brinson' in results
        assert 'factor' in results
        assert 'risk' in results
        assert 'transaction_cost' in results
    
    def test_calculate_performance_metrics(self, sample_returns, sample_factor_returns,
                                         sample_trades, sample_market_data):
        """Test performance metrics calculation"""
        portfolio_returns, benchmark_returns = sample_returns
        analysis = AttributionAnalysis(
            portfolio_returns,
            benchmark_returns,
            sample_factor_returns,
            sample_trades,
            sample_market_data
        )
        metrics = analysis.calculate_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'tracking_error' in metrics
        assert 'information_ratio' in metrics
        assert 'total_cost' in metrics
        assert 'cost_as_pct' in metrics 