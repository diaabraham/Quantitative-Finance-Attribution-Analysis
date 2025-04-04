import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transaction_costs.models import TransactionCostModel

@pytest.fixture
def sample_trades():
    """Create sample trade data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    trades = []
    for date in dates:
        # Generate random number of trades per day (1-5)
        n_trades = np.random.randint(1, 6)
        for _ in range(n_trades):
            trades.append({
                'timestamp': date,
                'symbol': f'STOCK_{np.random.randint(1, 11)}',
                'quantity': np.random.randint(100, 1001),
                'price': np.random.uniform(10, 100),
                'side': np.random.choice(['BUY', 'SELL'])
            })
    
    return pd.DataFrame(trades)

@pytest.fixture
def sample_market_data():
    """Create sample market data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    symbols = [f'STOCK_{i}' for i in range(1, 11)]
    
    market_data = []
    for date in dates:
        for symbol in symbols:
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
    
    return pd.DataFrame(market_data)

def test_transaction_cost_model(sample_trades, sample_market_data):
    """Test transaction cost model calculations"""
    # Initialize model
    model = TransactionCostModel(sample_trades, sample_market_data)
    
    # Test fixed cost calculation
    fixed_costs = model.calculate_fixed_costs()
    assert isinstance(fixed_costs, pd.Series)
    assert len(fixed_costs) == len(sample_trades)
    assert fixed_costs.min() >= 0
    
    # Test proportional cost calculation
    prop_costs = model.calculate_proportional_costs()
    assert isinstance(prop_costs, pd.Series)
    assert len(prop_costs) == len(sample_trades)
    assert prop_costs.min() >= 0
    
    # Test market impact calculation
    impact_costs = model.calculate_market_impact()
    assert isinstance(impact_costs, pd.Series)
    assert len(impact_costs) == len(sample_trades)
    assert impact_costs.min() >= 0
    
    # Test opportunity cost calculation
    opp_costs = model.calculate_opportunity_costs()
    assert isinstance(opp_costs, pd.Series)
    assert len(opp_costs) == len(sample_trades)
    assert opp_costs.min() >= 0
    
    # Test total cost calculation
    total_costs = model.calculate_total_costs()
    assert isinstance(total_costs, pd.Series)
    assert len(total_costs) == len(sample_trades)
    assert total_costs.min() >= 0

def test_trading_optimization(sample_trades, sample_market_data):
    """Test trading optimization methods"""
    model = TransactionCostModel(sample_trades, sample_market_data)
    
    # Test VWAP optimization
    vwap_schedule = model.optimize_vwap()
    assert isinstance(vwap_schedule, pd.DataFrame)
    assert 'timestamp' in vwap_schedule.columns
    assert 'quantity' in vwap_schedule.columns
    assert 'price' in vwap_schedule.columns
    
    # Test TWAP optimization
    twap_schedule = model.optimize_twap()
    assert isinstance(twap_schedule, pd.DataFrame)
    assert 'timestamp' in twap_schedule.columns
    assert 'quantity' in twap_schedule.columns
    assert 'price' in twap_schedule.columns
    
    # Test POV optimization
    pov_schedule = model.optimize_pov()
    assert isinstance(pov_schedule, pd.DataFrame)
    assert 'timestamp' in pov_schedule.columns
    assert 'quantity' in pov_schedule.columns
    assert 'price' in pov_schedule.columns

def test_cost_analysis(sample_trades, sample_market_data):
    """Test cost analysis methods"""
    model = TransactionCostModel(sample_trades, sample_market_data)
    
    # Test cost breakdown
    cost_breakdown = model.analyze_cost_breakdown()
    assert isinstance(cost_breakdown, dict)
    assert 'fixed_costs' in cost_breakdown
    assert 'proportional_costs' in cost_breakdown
    assert 'market_impact' in cost_breakdown
    assert 'opportunity_costs' in cost_breakdown
    
    # Test cost attribution
    cost_attribution = model.analyze_cost_attribution()
    assert isinstance(cost_attribution, pd.DataFrame)
    assert 'symbol' in cost_attribution.columns
    assert 'total_cost' in cost_attribution.columns
    
    # Test cost trends
    cost_trends = model.analyze_cost_trends()
    assert isinstance(cost_trends, pd.DataFrame)
    assert 'date' in cost_trends.columns
    assert 'avg_cost' in cost_trends.columns

def test_visualization(sample_trades, sample_market_data):
    """Test visualization methods"""
    model = TransactionCostModel(sample_trades, sample_market_data)
    
    # Test cost breakdown plot
    fig = model.plot_cost_breakdown()
    assert fig is not None
    
    # Test cost attribution plot
    fig = model.plot_cost_attribution()
    assert fig is not None
    
    # Test cost trends plot
    fig = model.plot_cost_trends()
    assert fig is not None
    
    # Test optimization comparison plot
    fig = model.plot_optimization_comparison()
    assert fig is not None

def test_edge_cases():
    """Test edge cases"""
    # Test with empty trades
    empty_trades = pd.DataFrame(columns=['timestamp', 'symbol', 'quantity', 'price', 'side'])
    empty_market = pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    model = TransactionCostModel(empty_trades, empty_market)
    
    # Test with single trade
    single_trade = pd.DataFrame([{
        'timestamp': datetime(2020, 1, 1),
        'symbol': 'STOCK_1',
        'quantity': 100,
        'price': 50.0,
        'side': 'BUY'
    }])
    single_market = pd.DataFrame([{
        'timestamp': datetime(2020, 1, 1),
        'symbol': 'STOCK_1',
        'open': 49.0,
        'high': 51.0,
        'low': 48.0,
        'close': 50.0,
        'volume': 10000
    }])
    model = TransactionCostModel(single_trade, single_market)
    
    # Test with large trades
    large_trades = pd.DataFrame([{
        'timestamp': datetime(2020, 1, 1),
        'symbol': 'STOCK_1',
        'quantity': 1000000,
        'price': 50.0,
        'side': 'BUY'
    }])
    model = TransactionCostModel(large_trades, single_market)
    
    # Test with zero quantity
    zero_trades = pd.DataFrame([{
        'timestamp': datetime(2020, 1, 1),
        'symbol': 'STOCK_1',
        'quantity': 0,
        'price': 50.0,
        'side': 'BUY'
    }])
    model = TransactionCostModel(zero_trades, single_market)

def test_performance(sample_trades, sample_market_data):
    """Test performance with large datasets"""
    # Create large dataset
    dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='D')
    symbols = [f'STOCK_{i}' for i in range(1, 101)]  # 100 symbols
    
    large_trades = []
    for date in dates:
        n_trades = np.random.randint(1, 11)
        for _ in range(n_trades):
            large_trades.append({
                'timestamp': date,
                'symbol': np.random.choice(symbols),
                'quantity': np.random.randint(100, 1001),
                'price': np.random.uniform(10, 100),
                'side': np.random.choice(['BUY', 'SELL'])
            })
    
    large_market = []
    for date in dates:
        for symbol in symbols:
            base_price = np.random.uniform(10, 100)
            large_market.append({
                'timestamp': date,
                'symbol': symbol,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price * 1.01,
                'volume': np.random.randint(10000, 100000)
            })
    
    # Initialize model
    model = TransactionCostModel(pd.DataFrame(large_trades), pd.DataFrame(large_market))
    
    # Time the calculations
    import time
    start_time = time.time()
    
    # Calculate all metrics
    fixed_costs = model.calculate_fixed_costs()
    prop_costs = model.calculate_proportional_costs()
    impact_costs = model.calculate_market_impact()
    opp_costs = model.calculate_opportunity_costs()
    total_costs = model.calculate_total_costs()
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Verify that calculations complete within reasonable time
    assert calculation_time < 30  # Should complete within 30 seconds 