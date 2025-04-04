from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .brinson import BrinsonAttribution
from .factor_attribution import FactorAttribution
from .risk_attribution import RiskAttribution
from .transaction_cost import TransactionCostAttribution

class AttributionAnalysis:
    """
    Main attribution analysis class that combines all attribution methods.
    
    This class provides a unified interface for:
    1. Brinson attribution (sector allocation and selection)
    2. Factor attribution (Fama-French and other factor models)
    3. Risk attribution (factor and idiosyncratic risk decomposition)
    4. Transaction cost attribution (implementation shortfall and market impact)
    
    Attributes:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    factor_returns : pd.DataFrame
        Factor returns
    trades : pd.DataFrame
        Trade data
    market_data : pd.DataFrame
        Market data
    """
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                 factor_returns: pd.DataFrame, trades: pd.DataFrame,
                 market_data: pd.DataFrame):
        """
        Initialize the attribution analysis.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
        factor_returns : pd.DataFrame
            Factor returns
        trades : pd.DataFrame
            Trade data
        market_data : pd.DataFrame
            Market data
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.factor_returns = factor_returns
        self.trades = trades
        self.market_data = market_data
        
        # Initialize attribution models
        self.brinson = BrinsonAttribution(
            portfolio_returns,
            benchmark_returns,
            pd.Series(1.0, index=portfolio_returns.index),  # Equal weights
            pd.Series(1.0, index=benchmark_returns.index)   # Equal weights
        )
        
        self.factor = FactorAttribution(
            portfolio_returns,
            factor_returns
        )
        
        self.risk = RiskAttribution(
            portfolio_returns,
            factor_returns,
            pd.Series(1.0, index=portfolio_returns.index),  # Equal weights
            pd.DataFrame(1.0, index=portfolio_returns.index,
                        columns=factor_returns.columns)      # Equal exposures
        )
        
        self.transaction_cost = TransactionCostAttribution(
            trades,
            market_data
        )
    
    def run_attribution_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete attribution analysis.
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing attribution results from all models
        """
        results = {}
        
        # Run Brinson attribution
        results['brinson'] = self.brinson.calculate_attribution_by_sector()
        
        # Run factor attribution
        factor_exposures = self.factor.estimate_factor_exposures()
        results['factor'] = self.factor.calculate_factor_contributions(factor_exposures)
        
        # Run risk attribution
        results['risk'] = pd.DataFrame({
            'marginal_contrib': self.risk.calculate_marginal_contributions(),
            'component_contrib': self.risk.calculate_component_contributions(),
            'factor_contrib': self.risk.calculate_factor_risk_contributions()
        })
        
        # Run transaction cost attribution
        results['transaction_cost'] = self.transaction_cost.calculate_total_costs()
        results['implementation_shortfall'] = \
            self.transaction_cost.calculate_implementation_shortfall()
        
        return results
    
    def calculate_rolling_attribution(self, window: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling attribution metrics.
        
        Parameters:
        -----------
        window : int
            Rolling window size
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing rolling attribution results
        """
        results = {}
        
        # Calculate rolling Brinson attribution
        results['brinson'] = self.brinson.calculate_rolling_attribution(window)
        
        # Calculate rolling factor attribution
        results['factor'] = self.factor.calculate_rolling_factor_risk(window)
        
        # Calculate rolling risk attribution
        results['risk'] = self.risk.calculate_rolling_risk_metrics(window)
        
        # Calculate rolling transaction costs
        results['transaction_cost'] = self.transaction_cost.calculate_rolling_costs(window)
        
        return results
    
    def plot_attribution_results(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot attribution results.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot Brinson attribution
        brinson_results = self.brinson.calculate_attribution_by_sector()
        brinson_results[['allocation', 'selection', 'interaction']].plot(
            kind='bar', stacked=True, ax=ax1
        )
        ax1.set_title('Brinson Attribution')
        ax1.set_ylabel('Return Contribution')
        ax1.legend(title='Effect')
        
        # Plot factor attribution
        factor_exposures = self.factor.estimate_factor_exposures()
        factor_results = self.factor.calculate_factor_contributions(factor_exposures)
        factor_results.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title('Factor Attribution')
        ax2.set_ylabel('Return Contribution')
        ax2.legend(title='Factor')
        
        # Plot risk attribution
        risk_results = pd.DataFrame({
            'marginal': self.risk.calculate_marginal_contributions(),
            'component': self.risk.calculate_component_contributions()
        })
        risk_results.plot(kind='bar', ax=ax3)
        ax3.set_title('Risk Attribution')
        ax3.set_ylabel('Risk Contribution')
        ax3.legend(title='Component')
        
        # Plot transaction cost attribution
        cost_results = self.transaction_cost.calculate_total_costs()
        cost_results[['fixed_cost', 'variable_cost', 'market_impact',
                     'opportunity_cost']].plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('Transaction Cost Attribution')
        ax4.set_ylabel('Cost ($)')
        ax4.legend(title='Component')
        
        plt.tight_layout()
        plt.show()
    
    def optimize_trading(self, target_quantity: float,
                        time_horizon: int = 1) -> pd.DataFrame:
        """
        Optimize trading schedule to minimize costs.
        
        Parameters:
        -----------
        target_quantity : float
            Target quantity to trade
        time_horizon : int
            Trading time horizon in days
        
        Returns:
        --------
        pd.DataFrame
            Optimized trading schedule
        """
        return self.transaction_cost.optimize_trading_schedule(
            target_quantity,
            time_horizon
        )
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Calculate return metrics
        metrics['total_return'] = (1 + self.portfolio_returns).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252/len(self.portfolio_returns)) - 1
        metrics['volatility'] = self.portfolio_returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
        
        # Calculate risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(self.portfolio_returns)
        metrics['tracking_error'] = (self.portfolio_returns - self.benchmark_returns).std() * np.sqrt(252)
        metrics['information_ratio'] = (metrics['annualized_return'] - 
                                      ((1 + self.benchmark_returns).prod() - 1) ** (252/len(self.benchmark_returns))) / \
                                     metrics['tracking_error']
        
        # Calculate cost metrics
        costs = self.transaction_cost.calculate_total_costs()
        metrics['total_cost'] = costs['total_cost'].sum()
        metrics['cost_as_pct'] = metrics['total_cost'] / abs(self.trades['quantity'] * self.trades['price']).sum()
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min() 