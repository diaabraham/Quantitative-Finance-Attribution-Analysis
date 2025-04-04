import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from factor_models.models import FamaFrenchModel, BarraModel, AxiomaModel

@pytest.fixture
def sample_returns():
    """Create sample returns data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    return returns

@pytest.fixture
def sample_factors():
    """Create sample factor data"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    # Fama-French factors
    ff_factors = pd.DataFrame({
        'market': np.random.normal(0.0003, 0.01, len(dates)),
        'size': np.random.normal(0.0002, 0.008, len(dates)),
        'value': np.random.normal(0.0001, 0.006, len(dates)),
        'momentum': np.random.normal(0.0004, 0.009, len(dates))
    }, index=dates)
    
    # BARRA factors
    barra_factors = pd.DataFrame({
        'market': np.random.normal(0.0003, 0.01, len(dates)),
        'size': np.random.normal(0.0002, 0.008, len(dates)),
        'value': np.random.normal(0.0001, 0.006, len(dates)),
        'momentum': np.random.normal(0.0004, 0.009, len(dates)),
        'volatility': np.random.normal(0.0001, 0.005, len(dates)),
        'growth': np.random.normal(0.0002, 0.007, len(dates)),
        'liquidity': np.random.normal(0.0001, 0.004, len(dates)),
        'leverage': np.random.normal(0.0001, 0.005, len(dates))
    }, index=dates)
    
    # Axioma factors
    axioma_factors = pd.DataFrame({
        'market': np.random.normal(0.0003, 0.01, len(dates)),
        'size': np.random.normal(0.0002, 0.008, len(dates)),
        'value': np.random.normal(0.0001, 0.006, len(dates)),
        'momentum': np.random.normal(0.0004, 0.009, len(dates)),
        'volatility': np.random.normal(0.0001, 0.005, len(dates)),
        'growth': np.random.normal(0.0002, 0.007, len(dates)),
        'liquidity': np.random.normal(0.0001, 0.004, len(dates)),
        'leverage': np.random.normal(0.0001, 0.005, len(dates)),
        'quality': np.random.normal(0.0002, 0.006, len(dates)),
        'yield': np.random.normal(0.0003, 0.007, len(dates))
    }, index=dates)
    
    return {
        'fama_french': ff_factors,
        'barra': barra_factors,
        'axioma': axioma_factors
    }

def test_fama_french_model(sample_returns, sample_factors):
    """Test Fama-French model"""
    model = FamaFrenchModel(sample_returns, sample_factors['fama_french'])
    
    # Test factor exposure estimation
    exposures = model.estimate_exposures()
    assert isinstance(exposures, pd.Series)
    assert len(exposures) == len(sample_factors['fama_french'].columns)
    assert 'market' in exposures.index
    assert 'size' in exposures.index
    assert 'value' in exposures.index
    assert 'momentum' in exposures.index
    
    # Test factor contribution calculation
    contributions = model.calculate_contributions()
    assert isinstance(contributions, pd.Series)
    assert len(contributions) == len(sample_factors['fama_french'].columns)
    
    # Test model fit
    r_squared = model.calculate_r_squared()
    assert isinstance(r_squared, float)
    assert 0 <= r_squared <= 1
    
    # Test rolling exposure estimation
    window = 20
    rolling_exposures = model.estimate_rolling_exposures(window)
    assert isinstance(rolling_exposures, pd.DataFrame)
    assert len(rolling_exposures) > window
    assert all(col in rolling_exposures.columns for col in sample_factors['fama_french'].columns)

def test_barra_model(sample_returns, sample_factors):
    """Test BARRA model"""
    model = BarraModel(sample_returns, sample_factors['barra'])
    
    # Test factor exposure estimation
    exposures = model.estimate_exposures()
    assert isinstance(exposures, pd.Series)
    assert len(exposures) == len(sample_factors['barra'].columns)
    
    # Test factor contribution calculation
    contributions = model.calculate_contributions()
    assert isinstance(contributions, pd.Series)
    assert len(contributions) == len(sample_factors['barra'].columns)
    
    # Test model fit
    r_squared = model.calculate_r_squared()
    assert isinstance(r_squared, float)
    assert 0 <= r_squared <= 1
    
    # Test risk decomposition
    risk_decomp = model.decompose_risk()
    assert isinstance(risk_decomp, dict)
    assert 'factor_risk' in risk_decomp
    assert 'specific_risk' in risk_decomp
    assert 'total_risk' in risk_decomp
    
    # Test style factor analysis
    style_analysis = model.analyze_style_factors()
    assert isinstance(style_analysis, pd.DataFrame)
    assert 'exposure' in style_analysis.columns
    assert 'contribution' in style_analysis.columns

def test_axioma_model(sample_returns, sample_factors):
    """Test Axioma model"""
    model = AxiomaModel(sample_returns, sample_factors['axioma'])
    
    # Test factor exposure estimation
    exposures = model.estimate_exposures()
    assert isinstance(exposures, pd.Series)
    assert len(exposures) == len(sample_factors['axioma'].columns)
    
    # Test factor contribution calculation
    contributions = model.calculate_contributions()
    assert isinstance(contributions, pd.Series)
    assert len(contributions) == len(sample_factors['axioma'].columns)
    
    # Test model fit
    r_squared = model.calculate_r_squared()
    assert isinstance(r_squared, float)
    assert 0 <= r_squared <= 1
    
    # Test factor interaction analysis
    interactions = model.analyze_factor_interactions()
    assert isinstance(interactions, pd.DataFrame)
    assert 'factor1' in interactions.columns
    assert 'factor2' in interactions.columns
    assert 'correlation' in interactions.columns
    
    # Test custom factor creation
    custom_factor = model.create_custom_factor(['size', 'value'], weights=[0.5, 0.5])
    assert isinstance(custom_factor, pd.Series)
    assert len(custom_factor) == len(sample_returns)
    
    # Test factor stability analysis
    stability = model.analyze_factor_stability()
    assert isinstance(stability, pd.DataFrame)
    assert 'exposure' in stability.columns
    assert 't_stat' in stability.columns
    assert 'p_value' in stability.columns

def test_model_comparison(sample_returns, sample_factors):
    """Test comparison between different factor models"""
    # Initialize models
    ff_model = FamaFrenchModel(sample_returns, sample_factors['fama_french'])
    barra_model = BarraModel(sample_returns, sample_factors['barra'])
    axioma_model = AxiomaModel(sample_returns, sample_factors['axioma'])
    
    # Compare R-squared values
    ff_r2 = ff_model.calculate_r_squared()
    barra_r2 = barra_model.calculate_r_squared()
    axioma_r2 = axioma_model.calculate_r_squared()
    
    assert isinstance(ff_r2, float)
    assert isinstance(barra_r2, float)
    assert isinstance(axioma_r2, float)
    
    # Compare factor contributions
    ff_contrib = ff_model.calculate_contributions()
    barra_contrib = barra_model.calculate_contributions()
    axioma_contrib = axioma_model.calculate_contributions()
    
    assert isinstance(ff_contrib, pd.Series)
    assert isinstance(barra_contrib, pd.Series)
    assert isinstance(axioma_contrib, pd.Series)
    
    # Compare risk decomposition
    ff_risk = ff_model.decompose_risk()
    barra_risk = barra_model.decompose_risk()
    axioma_risk = axioma_model.decompose_risk()
    
    assert isinstance(ff_risk, dict)
    assert isinstance(barra_risk, dict)
    assert isinstance(axioma_risk, dict)

def test_edge_cases():
    """Test edge cases"""
    # Test with zero returns
    zero_returns = pd.Series(np.zeros(100))
    zero_factors = pd.DataFrame(np.zeros((100, 4)), columns=['market', 'size', 'value', 'momentum'])
    model = FamaFrenchModel(zero_returns, zero_factors)
    
    # Test with constant returns
    constant_returns = pd.Series(np.ones(100) * 0.0001)
    constant_factors = pd.DataFrame(np.ones((100, 4)) * 0.0001, columns=['market', 'size', 'value', 'momentum'])
    model = FamaFrenchModel(constant_returns, constant_factors)
    
    # Test with NaN values
    nan_returns = pd.Series([np.nan] * 100)
    nan_factors = pd.DataFrame([[np.nan] * 4] * 100, columns=['market', 'size', 'value', 'momentum'])
    model = FamaFrenchModel(nan_returns, nan_factors)
    
    # Test with infinite values
    inf_returns = pd.Series([np.inf] * 100)
    inf_factors = pd.DataFrame([[np.inf] * 4] * 100, columns=['market', 'size', 'value', 'momentum'])
    model = FamaFrenchModel(inf_returns, inf_factors)

def test_performance(sample_returns, sample_factors):
    """Test performance with large datasets"""
    # Create large dataset
    dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    large_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    large_factors = pd.DataFrame(np.random.normal(0.0003, 0.01, (len(dates), 10)), 
                               columns=[f'factor_{i}' for i in range(10)],
                               index=dates)
    
    # Initialize model
    model = FamaFrenchModel(large_returns, large_factors)
    
    # Time the calculations
    import time
    start_time = time.time()
    
    # Calculate all metrics
    exposures = model.estimate_exposures()
    contributions = model.calculate_contributions()
    r_squared = model.calculate_r_squared()
    rolling_exposures = model.estimate_rolling_exposures(window=20)
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Verify that calculations complete within reasonable time
    assert calculation_time < 30  # Should complete within 30 seconds 