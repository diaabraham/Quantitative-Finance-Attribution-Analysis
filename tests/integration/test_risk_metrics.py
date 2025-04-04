import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from risk.advanced_metrics import AdvancedRiskMetrics

@pytest.fixture
def sample_returns():
    """Create sample returns data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    return returns

@pytest.fixture
def sample_benchmark():
    """Create sample benchmark returns"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates)
    return benchmark

def test_advanced_risk_metrics(sample_returns, sample_benchmark):
    """Test advanced risk metrics calculation"""
    # Initialize risk metrics
    metrics = AdvancedRiskMetrics(sample_returns, sample_benchmark)
    
    # Test Value at Risk calculations
    var_metrics = metrics.calculate_var_metrics()
    assert isinstance(var_metrics, dict)
    assert 'historical_var' in var_metrics
    assert 'parametric_var' in var_metrics
    assert 'modified_var' in var_metrics
    assert 'conditional_var' in var_metrics
    
    # Test risk-adjusted return metrics
    risk_metrics = metrics.calculate_risk_adjusted_metrics()
    assert isinstance(risk_metrics, dict)
    assert 'sharpe_ratio' in risk_metrics
    assert 'sortino_ratio' in risk_metrics
    assert 'omega_ratio' in risk_metrics
    assert 'calmar_ratio' in risk_metrics
    assert 'information_ratio' in risk_metrics
    assert 'm2_measure' in risk_metrics
    assert 'm3_measure' in risk_metrics
    assert 'm4_measure' in risk_metrics
    
    # Test drawdown analysis
    drawdown_metrics = metrics.calculate_drawdown_metrics()
    assert isinstance(drawdown_metrics, dict)
    assert 'max_drawdown' in drawdown_metrics
    assert 'avg_drawdown' in drawdown_metrics
    assert 'drawdown_duration' in drawdown_metrics
    assert 'recovery_time' in drawdown_metrics
    
    # Test tail risk metrics
    tail_metrics = metrics.calculate_tail_risk_metrics()
    assert isinstance(tail_metrics, dict)
    assert 'skewness' in tail_metrics
    assert 'kurtosis' in tail_metrics
    assert 'var_95' in tail_metrics
    assert 'expected_shortfall_95' in tail_metrics
    assert 'tail_ratio' in tail_metrics
    
    # Test rolling metrics
    window = 20
    rolling_metrics = metrics.calculate_rolling_metrics(window)
    assert isinstance(rolling_metrics, pd.DataFrame)
    assert len(rolling_metrics) > window
    assert 'rolling_sharpe' in rolling_metrics.columns
    assert 'rolling_sortino' in rolling_metrics.columns
    assert 'rolling_var' in rolling_metrics.columns
    
    # Test correlation analysis
    corr_metrics = metrics.calculate_correlation_metrics()
    assert isinstance(corr_metrics, dict)
    assert 'correlation' in corr_metrics
    assert 'beta' in corr_metrics
    assert 'tracking_error' in corr_metrics
    assert 'information_ratio' in corr_metrics

def test_risk_metric_calculations(sample_returns, sample_benchmark):
    """Test specific risk metric calculations"""
    metrics = AdvancedRiskMetrics(sample_returns, sample_benchmark)
    
    # Test Modified VaR calculation
    modified_var = metrics.calculate_modified_var(confidence_level=0.95)
    assert isinstance(modified_var, float)
    assert modified_var < 0  # VaR should be negative
    
    # Test Conditional VaR calculation
    conditional_var = metrics.calculate_conditional_var(confidence_level=0.95)
    assert isinstance(conditional_var, float)
    assert conditional_var < modified_var  # ES should be more negative than VaR
    
    # Test M3 Measure calculation
    m3_measure = metrics.calculate_m3_measure()
    assert isinstance(m3_measure, float)
    
    # Test M4 Measure calculation
    m4_measure = metrics.calculate_m4_measure()
    assert isinstance(m4_measure, float)
    
    # Test Omega Ratio calculation
    omega_ratio = metrics.calculate_omega_ratio(threshold=0.0)
    assert isinstance(omega_ratio, float)
    assert omega_ratio > 0
    
    # Test Tail Ratio calculation
    tail_ratio = metrics.calculate_tail_ratio()
    assert isinstance(tail_ratio, float)
    assert tail_ratio > 0

def test_risk_metric_visualization(sample_returns, sample_benchmark):
    """Test risk metric visualization"""
    metrics = AdvancedRiskMetrics(sample_returns, sample_benchmark)
    
    # Test returns distribution plot
    fig = metrics.plot_returns_distribution()
    assert fig is not None
    
    # Test drawdown plot
    fig = metrics.plot_drawdown()
    assert fig is not None
    
    # Test rolling metrics plot
    fig = metrics.plot_rolling_metrics(window=20)
    assert fig is not None
    
    # Test correlation plot
    fig = metrics.plot_correlation()
    assert fig is not None
    
    # Test risk metrics summary plot
    fig = metrics.plot_risk_metrics_summary()
    assert fig is not None

def test_risk_metric_edge_cases():
    """Test risk metrics with edge cases"""
    # Test with zero returns
    zero_returns = pd.Series(np.zeros(100))
    zero_benchmark = pd.Series(np.zeros(100))
    metrics = AdvancedRiskMetrics(zero_returns, zero_benchmark)
    
    # Test with constant returns
    constant_returns = pd.Series(np.ones(100) * 0.0001)
    constant_benchmark = pd.Series(np.ones(100) * 0.0001)
    metrics = AdvancedRiskMetrics(constant_returns, constant_benchmark)
    
    # Test with NaN values
    nan_returns = pd.Series([np.nan] * 100)
    nan_benchmark = pd.Series([np.nan] * 100)
    metrics = AdvancedRiskMetrics(nan_returns, nan_benchmark)
    
    # Test with infinite values
    inf_returns = pd.Series([np.inf] * 100)
    inf_benchmark = pd.Series([np.inf] * 100)
    metrics = AdvancedRiskMetrics(inf_returns, inf_benchmark)
    
    # Test with very small values
    small_returns = pd.Series(np.random.normal(1e-10, 1e-10, 100))
    small_benchmark = pd.Series(np.random.normal(1e-10, 1e-10, 100))
    metrics = AdvancedRiskMetrics(small_returns, small_benchmark)
    
    # Test with very large values
    large_returns = pd.Series(np.random.normal(1e10, 1e10, 100))
    large_benchmark = pd.Series(np.random.normal(1e10, 1e10, 100))
    metrics = AdvancedRiskMetrics(large_returns, large_benchmark)

def test_risk_metric_performance(sample_returns, sample_benchmark):
    """Test risk metric performance with large datasets"""
    # Create large dataset
    dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    large_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    large_benchmark = pd.Series(np.random.normal(0.0003, 0.008, len(dates)), index=dates)
    
    # Initialize metrics
    metrics = AdvancedRiskMetrics(large_returns, large_benchmark)
    
    # Time the calculations
    import time
    start_time = time.time()
    
    # Calculate all metrics
    var_metrics = metrics.calculate_var_metrics()
    risk_metrics = metrics.calculate_risk_adjusted_metrics()
    drawdown_metrics = metrics.calculate_drawdown_metrics()
    tail_metrics = metrics.calculate_tail_risk_metrics()
    rolling_metrics = metrics.calculate_rolling_metrics(window=20)
    corr_metrics = metrics.calculate_correlation_metrics()
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Verify that calculations complete within reasonable time
    assert calculation_time < 10  # Should complete within 10 seconds 