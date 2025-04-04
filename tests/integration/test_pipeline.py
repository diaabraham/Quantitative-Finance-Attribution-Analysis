import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from attribution import AttributionAnalysis
from attribution.brinson import BrinsonAttribution
from attribution.factor_attribution import FactorAttribution
from attribution.risk_attribution import RiskAttribution
from attribution.transaction_cost import TransactionCostAttribution
from factor_models.models import FamaFrenchModel, BarraModel, AxiomaModel
from risk.advanced_metrics import AdvancedRiskMetrics
from transaction_costs.models import TransactionCostModel

@pytest.fixture
def sample_data():
    """Create sample data for integration testing"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    # Generate portfolio and benchmark returns
    portfolio_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates)
    
    # Generate factor returns
    factor_returns = pd.DataFrame({
        'market': np.random.normal(0.0003, 0.01, len(dates)),
        'size': np.random.normal(0.0002, 0.008, len(dates)),
        'value': np.random.normal(0.0001, 0.006, len(dates)),
        'momentum': np.random.normal(0.0004, 0.009, len(dates))
    }, index=dates)
    
    # Generate trade data
    trades = []
    for date in dates:
        n_trades = np.random.randint(1, 6)
        for _ in range(n_trades):
            trades.append({
                'timestamp': date,
                'symbol': f'STOCK_{np.random.randint(1, 11)}',
                'quantity': np.random.randint(100, 1001),
                'price': np.random.uniform(10, 100),
                'side': np.random.choice(['BUY', 'SELL'])
            })
    
    # Generate market data
    market_data = []
    for date in dates:
        for symbol in [f'STOCK_{i}' for i in range(1, 11)]:
            base_price = np.random.uniform(10, 100)
            market_data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price * 1.01,
                'volume': np.random.randint(10000, 100000)
            })
    
    return {
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'factor_returns': factor_returns,
        'trades': pd.DataFrame(trades),
        'market_data': pd.DataFrame(market_data)
    }

def test_complete_pipeline(sample_data):
    """Test the complete attribution analysis pipeline"""
    # Initialize attribution analysis
    analysis = AttributionAnalysis(
        portfolio_returns=sample_data['portfolio_returns'],
        benchmark_returns=sample_data['benchmark_returns'],
        factor_returns=sample_data['factor_returns'],
        trades=sample_data['trades'],
        market_data=sample_data['market_data']
    )
    
    # Run complete attribution analysis
    results = analysis.run_attribution_analysis()
    
    # Test Brinson attribution results
    assert 'brinson' in results
    brinson_results = results['brinson']
    assert isinstance(brinson_results, dict)
    assert 'allocation_effect' in brinson_results
    assert 'selection_effect' in brinson_results
    assert 'interaction_effect' in brinson_results
    
    # Test factor attribution results
    assert 'factor' in results
    factor_results = results['factor']
    assert isinstance(factor_results, dict)
    assert 'exposures' in factor_results
    assert 'contributions' in factor_results
    assert 'r_squared' in factor_results
    
    # Test risk attribution results
    assert 'risk' in results
    risk_results = results['risk']
    assert isinstance(risk_results, dict)
    assert 'marginal_contributions' in risk_results
    assert 'component_contributions' in risk_results
    assert 'factor_risk' in risk_results
    
    # Test transaction cost results
    assert 'transaction_costs' in results
    cost_results = results['transaction_costs']
    assert isinstance(cost_results, dict)
    assert 'fixed_costs' in cost_results
    assert 'proportional_costs' in cost_results
    assert 'market_impact' in cost_results
    assert 'opportunity_costs' in cost_results

def test_rolling_attribution(sample_data):
    """Test rolling attribution analysis"""
    analysis = AttributionAnalysis(
        portfolio_returns=sample_data['portfolio_returns'],
        benchmark_returns=sample_data['benchmark_returns'],
        factor_returns=sample_data['factor_returns'],
        trades=sample_data['trades'],
        market_data=sample_data['market_data']
    )
    
    # Calculate rolling attribution
    window = 20
    rolling_results = analysis.calculate_rolling_attribution(window)
    
    # Test rolling results
    assert isinstance(rolling_results, dict)
    assert 'brinson' in rolling_results
    assert 'factor' in rolling_results
    assert 'risk' in rolling_results
    assert 'transaction_costs' in rolling_results
    
    # Test rolling Brinson attribution
    brinson_rolling = rolling_results['brinson']
    assert isinstance(brinson_rolling, pd.DataFrame)
    assert len(brinson_rolling) > window
    assert 'allocation_effect' in brinson_rolling.columns
    assert 'selection_effect' in brinson_rolling.columns
    assert 'interaction_effect' in brinson_rolling.columns
    
    # Test rolling factor attribution
    factor_rolling = rolling_results['factor']
    assert isinstance(factor_rolling, pd.DataFrame)
    assert len(factor_rolling) > window
    assert 'market' in factor_rolling.columns
    assert 'size' in factor_rolling.columns
    assert 'value' in factor_rolling.columns
    assert 'momentum' in factor_rolling.columns

def test_performance_metrics(sample_data):
    """Test performance metrics calculation"""
    analysis = AttributionAnalysis(
        portfolio_returns=sample_data['portfolio_returns'],
        benchmark_returns=sample_data['benchmark_returns'],
        factor_returns=sample_data['factor_returns'],
        trades=sample_data['trades'],
        market_data=sample_data['market_data']
    )
    
    # Calculate performance metrics
    metrics = analysis.calculate_performance_metrics()
    
    # Test metrics
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'annualized_return' in metrics
    assert 'volatility' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'tracking_error' in metrics
    assert 'total_costs' in metrics
    
    # Test metric values
    assert isinstance(metrics['total_return'], float)
    assert isinstance(metrics['annualized_return'], float)
    assert isinstance(metrics['volatility'], float)
    assert isinstance(metrics['sharpe_ratio'], float)
    assert isinstance(metrics['max_drawdown'], float)
    assert isinstance(metrics['tracking_error'], float)
    assert isinstance(metrics['total_costs'], float)

def test_trading_optimization(sample_data):
    """Test trading optimization"""
    analysis = AttributionAnalysis(
        portfolio_returns=sample_data['portfolio_returns'],
        benchmark_returns=sample_data['benchmark_returns'],
        factor_returns=sample_data['factor_returns'],
        trades=sample_data['trades'],
        market_data=sample_data['market_data']
    )
    
    # Optimize trading schedule
    target_quantity = 1000
    time_horizon = 5
    schedule = analysis.optimize_trading(target_quantity, time_horizon)
    
    # Test schedule
    assert isinstance(schedule, pd.DataFrame)
    assert 'timestamp' in schedule.columns
    assert 'quantity' in schedule.columns
    assert 'price' in schedule.columns
    
    # Test schedule properties
    assert len(schedule) == time_horizon
    assert schedule['quantity'].sum() == target_quantity
    assert all(schedule['quantity'] >= 0)

def test_visualization(sample_data):
    """Test visualization methods"""
    analysis = AttributionAnalysis(
        portfolio_returns=sample_data['portfolio_returns'],
        benchmark_returns=sample_data['benchmark_returns'],
        factor_returns=sample_data['factor_returns'],
        trades=sample_data['trades'],
        market_data=sample_data['market_data']
    )
    
    # Test attribution results plot
    fig = analysis.plot_attribution_results()
    assert fig is not None
    
    # Test performance metrics plot
    fig = analysis.plot_performance_metrics()
    assert fig is not None
    
    # Test rolling attribution plot
    fig = analysis.plot_rolling_attribution(window=20)
    assert fig is not None

def test_edge_cases():
    """Test edge cases"""
    # Test with empty data
    empty_data = {
        'portfolio_returns': pd.Series(),
        'benchmark_returns': pd.Series(),
        'factor_returns': pd.DataFrame(),
        'trades': pd.DataFrame(),
        'market_data': pd.DataFrame()
    }
    analysis = AttributionAnalysis(**empty_data)
    
    # Test with single data point
    single_data = {
        'portfolio_returns': pd.Series([0.0005], index=[datetime(2020, 1, 1)]),
        'benchmark_returns': pd.Series([0.0003], index=[datetime(2020, 1, 1)]),
        'factor_returns': pd.DataFrame([[0.0003, 0.0002, 0.0001, 0.0004]], 
                                     columns=['market', 'size', 'value', 'momentum'],
                                     index=[datetime(2020, 1, 1)]),
        'trades': pd.DataFrame([{
            'timestamp': datetime(2020, 1, 1),
            'symbol': 'STOCK_1',
            'quantity': 100,
            'price': 50.0,
            'side': 'BUY'
        }]),
        'market_data': pd.DataFrame([{
            'timestamp': datetime(2020, 1, 1),
            'symbol': 'STOCK_1',
            'open': 49.0,
            'high': 51.0,
            'low': 48.0,
            'close': 50.0,
            'volume': 10000
        }])
    }
    analysis = AttributionAnalysis(**single_data)
    
    # Test with NaN values
    nan_data = {
        'portfolio_returns': pd.Series([np.nan]),
        'benchmark_returns': pd.Series([np.nan]),
        'factor_returns': pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]]),
        'trades': pd.DataFrame([{
            'timestamp': datetime(2020, 1, 1),
            'symbol': 'STOCK_1',
            'quantity': np.nan,
            'price': np.nan,
            'side': 'BUY'
        }]),
        'market_data': pd.DataFrame([{
            'timestamp': datetime(2020, 1, 1),
            'symbol': 'STOCK_1',
            'open': np.nan,
            'high': np.nan,
            'low': np.nan,
            'close': np.nan,
            'volume': np.nan
        }])
    }
    analysis = AttributionAnalysis(**nan_data)

def test_performance(sample_data):
    """Test performance with large datasets"""
    # Create large dataset
    dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    large_data = {
        'portfolio_returns': pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates),
        'benchmark_returns': pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates),
        'factor_returns': pd.DataFrame(np.random.normal(0.0003, 0.01, (len(dates), 4)),
                                     columns=['market', 'size', 'value', 'momentum'],
                                     index=dates),
        'trades': pd.DataFrame([{
            'timestamp': date,
            'symbol': f'STOCK_{np.random.randint(1, 101)}',
            'quantity': np.random.randint(100, 1001),
            'price': np.random.uniform(10, 100),
            'side': np.random.choice(['BUY', 'SELL'])
        } for date in dates]),
        'market_data': pd.DataFrame([{
            'timestamp': date,
            'symbol': f'STOCK_{i}',
            'open': np.random.uniform(10, 100),
            'high': np.random.uniform(10, 100) * 1.02,
            'low': np.random.uniform(10, 100) * 0.98,
            'close': np.random.uniform(10, 100) * 1.01,
            'volume': np.random.randint(10000, 100000)
        } for date in dates for i in range(1, 101)])
    }
    
    # Initialize analysis
    analysis = AttributionAnalysis(**large_data)
    
    # Time the calculations
    import time
    start_time = time.time()
    
    # Run complete analysis
    results = analysis.run_attribution_analysis()
    rolling_results = analysis.calculate_rolling_attribution(window=20)
    metrics = analysis.calculate_performance_metrics()
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Verify that calculations complete within reasonable time
    assert calculation_time < 60  # Should complete within 60 seconds 