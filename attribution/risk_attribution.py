import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

class RiskAttribution:
    """
    Implementation of risk attribution analysis.
    
    This class provides methods to:
    1. Decompose portfolio risk into factor and idiosyncratic components
    2. Calculate marginal contributions to risk
    3. Calculate component contributions to risk
    4. Calculate tracking error decomposition
    
    Attributes:
    -----------
    returns : pd.Series
        Portfolio returns
    factor_returns : pd.DataFrame
        Factor returns
    weights : pd.Series
        Portfolio weights
    factor_exposures : pd.DataFrame
        Factor exposures
    """
    
    def __init__(self, returns: pd.Series, factor_returns: pd.DataFrame,
                 weights: pd.Series, factor_exposures: pd.DataFrame):
        """
        Initialize the risk attribution model.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        factor_returns : pd.DataFrame
            Factor returns
        weights : pd.Series
            Portfolio weights
        factor_exposures : pd.DataFrame
            Factor exposures
        """
        self.returns = returns
        self.factor_returns = factor_returns
        self.weights = weights
        self.factor_exposures = factor_exposures
        
        # Calculate covariance matrices
        self.factor_cov = factor_returns.cov()
        self.residual_cov = self._calculate_residual_covariance()
        self.total_cov = self._calculate_total_covariance()
    
    def _calculate_residual_covariance(self) -> pd.DataFrame:
        """Calculate residual covariance matrix"""
        # Calculate factor returns
        factor_returns = self.factor_exposures @ self.factor_returns.T
        
        # Calculate residuals
        residuals = self.returns - factor_returns
        
        return np.diag(residuals.var())
    
    def _calculate_total_covariance(self) -> pd.DataFrame:
        """Calculate total covariance matrix"""
        return self.factor_exposures @ self.factor_cov @ self.factor_exposures.T + \
               self.residual_cov
    
    def calculate_marginal_contributions(self) -> pd.Series:
        """
        Calculate marginal contributions to risk.
        
        Returns:
        --------
        pd.Series
            Marginal contributions to risk for each asset
        """
        portfolio_vol = np.sqrt(self.weights @ self.total_cov @ self.weights)
        marginal_contrib = (self.total_cov @ self.weights) / portfolio_vol
        
        return pd.Series(marginal_contrib, index=self.weights.index)
    
    def calculate_component_contributions(self) -> pd.Series:
        """
        Calculate component contributions to risk.
        
        Returns:
        --------
        pd.Series
            Component contributions to risk for each asset
        """
        marginal_contrib = self.calculate_marginal_contributions()
        component_contrib = self.weights * marginal_contrib
        
        return component_contrib
    
    def calculate_factor_risk_contributions(self) -> pd.Series:
        """
        Calculate factor contributions to risk.
        
        Returns:
        --------
        pd.Series
            Risk contributions from each factor
        """
        portfolio_vol = np.sqrt(self.weights @ self.total_cov @ self.weights)
        
        # Calculate factor risk contributions
        factor_risk = []
        for factor in self.factor_returns.columns:
            # Get factor exposure
            exposure = self.factor_exposures[factor]
            
            # Calculate contribution
            contrib = (exposure @ self.factor_cov[factor] @ exposure.T) / portfolio_vol
            factor_risk.append(contrib)
        
        return pd.Series(factor_risk, index=self.factor_returns.columns)
    
    def calculate_tracking_error(self, benchmark_weights: pd.Series) -> float:
        """
        Calculate tracking error.
        
        Parameters:
        -----------
        benchmark_weights : pd.Series
            Benchmark portfolio weights
        
        Returns:
        --------
        float
            Tracking error
        """
        active_weights = self.weights - benchmark_weights
        return np.sqrt(active_weights @ self.total_cov @ active_weights)
    
    def decompose_tracking_error(self, benchmark_weights: pd.Series) -> Dict[str, float]:
        """
        Decompose tracking error into factor and idiosyncratic components.
        
        Parameters:
        -----------
        benchmark_weights : pd.Series
            Benchmark portfolio weights
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing:
            - 'total': Total tracking error
            - 'factor': Factor component
            - 'idiosyncratic': Idiosyncratic component
        """
        active_weights = self.weights - benchmark_weights
        
        # Calculate components
        factor_te = np.sqrt(active_weights @ self.factor_exposures @ self.factor_cov @ 
                           self.factor_exposures.T @ active_weights)
        idio_te = np.sqrt(active_weights @ self.residual_cov @ active_weights)
        total_te = self.calculate_tracking_error(benchmark_weights)
        
        return {
            'total': total_te,
            'factor': factor_te,
            'idiosyncratic': idio_te
        }
    
    def calculate_rolling_risk_metrics(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.
        
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
        metrics = ['total_risk', 'factor_risk', 'idio_risk', 'tracking_error']
        results = pd.DataFrame(index=self.returns.index, columns=metrics)
        
        # Calculate rolling metrics
        for i in range(window, len(self.returns)):
            # Get data for the window
            returns_window = self.returns.iloc[i-window:i]
            factor_returns_window = self.factor_returns.iloc[i-window:i]
            weights_window = self.weights.iloc[i-window:i]
            exposures_window = self.factor_exposures.iloc[i-window:i]
            
            # Create risk attribution model for the window
            risk_model = RiskAttribution(
                returns_window,
                factor_returns_window,
                weights_window,
                exposures_window
            )
            
            # Calculate metrics
            total_risk = np.sqrt(weights_window @ risk_model.total_cov @ weights_window)
            factor_risk = np.sqrt(weights_window @ risk_model.factor_exposures @ 
                                risk_model.factor_cov @ risk_model.factor_exposures.T @ 
                                weights_window)
            idio_risk = np.sqrt(weights_window @ risk_model.residual_cov @ weights_window)
            
            # Calculate tracking error (assuming equal weights as benchmark)
            benchmark_weights = pd.Series(1/len(weights_window), index=weights_window.index)
            tracking_error = risk_model.calculate_tracking_error(benchmark_weights)
            
            # Store results
            results.iloc[i] = {
                'total_risk': total_risk,
                'factor_risk': factor_risk,
                'idio_risk': idio_risk,
                'tracking_error': tracking_error
            }
        
        return results
    
    def plot_risk_attribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot risk attribution results.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot component contributions
        component_contrib = self.calculate_component_contributions()
        component_contrib.plot(kind='bar', ax=ax1)
        ax1.set_title('Component Contributions to Risk')
        ax1.set_ylabel('Risk Contribution')
        
        # Plot factor risk contributions
        factor_contrib = self.calculate_factor_risk_contributions()
        factor_contrib.plot(kind='bar', ax=ax2)
        ax2.set_title('Factor Contributions to Risk')
        ax2.set_ylabel('Risk Contribution')
        
        plt.tight_layout()
        plt.show() 