import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy import stats

def calculate_factor_correlation(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix between factors.
    
    Args:
        factor_returns: DataFrame of factor returns
        
    Returns:
        Correlation matrix as DataFrame
    """
    return factor_returns.corr()

def calculate_factor_volatility(factor_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate the volatility of each factor.
    
    Args:
        factor_returns: DataFrame of factor returns
        
    Returns:
        Series of factor volatilities
    """
    return factor_returns.std()

def calculate_factor_skewness(factor_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate the skewness of each factor.
    
    Args:
        factor_returns: DataFrame of factor returns
        
    Returns:
        Series of factor skewness values
    """
    return factor_returns.skew()

def calculate_factor_kurtosis(factor_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate the kurtosis of each factor.
    
    Args:
        factor_returns: DataFrame of factor returns
        
    Returns:
        Series of factor kurtosis values
    """
    return factor_returns.kurtosis()

def calculate_factor_autocorrelation(factor_returns: pd.DataFrame, 
                                   lags: int = 5) -> pd.DataFrame:
    """
    Calculate the autocorrelation of each factor for specified lags.
    
    Args:
        factor_returns: DataFrame of factor returns
        lags: Number of lags to calculate
        
    Returns:
        DataFrame of autocorrelation values
    """
    autocorr = {}
    for factor in factor_returns.columns:
        autocorr[factor] = [factor_returns[factor].autocorr(lag) 
                           for lag in range(1, lags + 1)]
    return pd.DataFrame(autocorr, index=range(1, lags + 1))

def calculate_factor_ic(factor_returns: pd.DataFrame, 
                       forward_returns: pd.Series,
                       method: str = 'spearman') -> pd.Series:
    """
    Calculate the Information Coefficient (IC) between factors and forward returns.
    
    Args:
        factor_returns: DataFrame of factor returns
        forward_returns: Series of forward returns
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Series of IC values
    """
    if method == 'pearson':
        ic = factor_returns.corrwith(forward_returns)
    elif method == 'spearman':
        ic = factor_returns.corrwith(forward_returns, method='spearman')
    else:
        raise ValueError("Method must be either 'pearson' or 'spearman'")
    return ic

def calculate_factor_ir(factor_returns: pd.DataFrame,
                       forward_returns: pd.Series,
                       method: str = 'spearman') -> pd.Series:
    """
    Calculate the Information Ratio (IR) of each factor.
    
    Args:
        factor_returns: DataFrame of factor returns
        forward_returns: Series of forward returns
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Series of IR values
    """
    ic = calculate_factor_ic(factor_returns, forward_returns, method)
    ic_std = ic.std()
    return ic / ic_std if ic_std != 0 else pd.Series(0, index=ic.index)

def calculate_factor_turnover(factor_returns: pd.DataFrame,
                            window: int = 20) -> pd.Series:
    """
    Calculate the turnover of each factor.
    
    Args:
        factor_returns: DataFrame of factor returns
        window: Rolling window size
        
    Returns:
        Series of factor turnover values
    """
    return factor_returns.rolling(window).std().mean()

def calculate_factor_decay(factor_returns: pd.DataFrame,
                          forward_returns: pd.Series,
                          max_lag: int = 20) -> pd.DataFrame:
    """
    Calculate the decay of factor predictive power over time.
    
    Args:
        factor_returns: DataFrame of factor returns
        forward_returns: Series of forward returns
        max_lag: Maximum number of lags to calculate
        
    Returns:
        DataFrame of decay values
    """
    decay = {}
    for lag in range(1, max_lag + 1):
        decay[lag] = calculate_factor_ic(factor_returns.shift(lag), 
                                       forward_returns)
    return pd.DataFrame(decay) 