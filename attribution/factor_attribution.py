import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class FactorAttribution:
    """
    Implementation of factor attribution analysis using Fama-French and other factor models.
    
    This class provides methods to:
    1. Estimate factor exposures using regression
    2. Decompose returns into factor contributions
    3. Calculate factor-specific risk metrics
    4. Visualize factor attribution results
    
    Attributes:
    -----------
    returns : pd.Series
        Portfolio returns
    factor_returns : pd.DataFrame
        Factor returns (e.g., market, size, value, momentum)
    factor_names : List[str]
        Names of the factors
    """
    
    def __init__(self, returns: pd.Series, factor_returns: pd.DataFrame):
        """
        Initialize the factor attribution model.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        factor_returns : pd.DataFrame
            Factor returns with factors as columns
        """
        self.returns = returns
        self.factor_returns = factor_returns
        self.factor_names = factor_returns.columns.tolist()
        
        # Add constant for regression
        self.factor_returns = sm.add_constant(self.factor_returns)
        self.factor_names = ['alpha'] + self.factor_names
    
    def estimate_factor_exposures(self, window: Optional[int] = None) -> pd.DataFrame:
        """
        Estimate factor exposures using rolling regression.
        
        Parameters:
        -----------
        window : Optional[int]
            Rolling window size. If None, uses full sample.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index and factor exposures as columns
        """
        if window is None:
            # Full sample regression
            model = sm.OLS(self.returns, self.factor_returns)
            results = model.fit()
            exposures = pd.Series(results.params, index=self.factor_names)
            return pd.DataFrame([exposures], index=[self.returns.index[-1]])
        
        else:
            # Rolling regression
            exposures = pd.DataFrame(index=self.returns.index, columns=self.factor_names)
            
            for i in range(window, len(self.returns)):
                # Get data for the window
                returns_window = self.returns.iloc[i-window:i]
                factors_window = self.factor_returns.iloc[i-window:i]
                
                # Run regression
                model = sm.OLS(returns_window, factors_window)
                results = model.fit()
                
                # Store exposures
                exposures.iloc[i] = results.params
            
            return exposures
    
    def calculate_factor_contributions(self, exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate factor contributions to returns.
        
        Parameters:
        -----------
        exposures : pd.DataFrame
            Factor exposures from estimate_factor_exposures
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index and factor contributions as columns
        """
        contributions = pd.DataFrame(index=exposures.index, columns=self.factor_names)
        
        for factor in self.factor_names:
            if factor == 'alpha':
                contributions[factor] = exposures[factor]
            else:
                contributions[factor] = exposures[factor] * self.factor_returns[factor]
        
        return contributions
    
    def calculate_factor_risk_metrics(self, contributions: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk metrics for each factor.
        
        Parameters:
        -----------
        contributions : pd.DataFrame
            Factor contributions from calculate_factor_contributions
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of risk metrics for each factor
        """
        metrics = {}
        
        for factor in self.factor_names:
            factor_contrib = contributions[factor]
            
            metrics[f'{factor}_volatility'] = factor_contrib.std() * np.sqrt(252)
            metrics[f'{factor}_sharpe'] = factor_contrib.mean() / factor_contrib.std() * np.sqrt(252)
            metrics[f'{factor}_max_drawdown'] = self._calculate_max_drawdown(factor_contrib)
            metrics[f'{factor}_skew'] = stats.skew(factor_contrib)
            metrics[f'{factor}_kurtosis'] = stats.kurtosis(factor_contrib)
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min()
    
    def plot_factor_attribution(self, contributions: pd.DataFrame, 
                              figsize: Tuple[int, int] = (12, 8)):
        """
        Plot factor attribution results.
        
        Parameters:
        -----------
        contributions : pd.DataFrame
            Factor contributions from calculate_factor_contributions
        figsize : Tuple[int, int]
            Figure size (width, height)
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot cumulative contributions
        cum_contributions = (1 + contributions).cumprod()
        cum_contributions.plot(ax=ax1)
        ax1.set_title('Cumulative Factor Contributions')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(title='Factor')
        
        # Plot rolling contributions
        rolling_contrib = contributions.rolling(window=20).mean()
        rolling_contrib.plot(ax=ax2)
        ax2.set_title('Rolling Factor Contributions (20-day)')
        ax2.set_ylabel('Return Contribution')
        ax2.legend(title='Factor')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_rolling_factor_risk(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling factor risk metrics.
        
        Parameters:
        -----------
        window : int
            Rolling window size
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index and rolling risk metrics as columns
        """
        # Initialize results DataFrame
        metrics = ['volatility', 'sharpe', 'max_drawdown', 'skew', 'kurtosis']
        columns = [f'{factor}_{metric}' for factor in self.factor_names 
                  for metric in metrics]
        results = pd.DataFrame(index=self.returns.index, columns=columns)
        
        # Calculate rolling metrics
        for i in range(window, len(self.returns)):
            # Get data for the window
            returns_window = self.returns.iloc[i-window:i]
            factors_window = self.factor_returns.iloc[i-window:i]
            
            # Create factor attribution model for the window
            factor_model = FactorAttribution(returns_window, factors_window)
            
            # Estimate exposures and calculate contributions
            exposures = factor_model.estimate_factor_exposures()
            contributions = factor_model.calculate_factor_contributions(exposures)
            
            # Calculate risk metrics
            risk_metrics = factor_model.calculate_factor_risk_metrics(contributions)
            
            # Store results
            results.iloc[i] = risk_metrics
        
        return results 