import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class BrinsonAttribution:
    """
    Implementation of the Brinson model for performance attribution.
    
    The Brinson model decomposes portfolio returns into:
    1. Allocation effect: Impact of overweighting/underweighting sectors
    2. Selection effect: Impact of stock selection within sectors
    3. Interaction effect: Combined effect of allocation and selection
    
    Attributes:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns by sector
    benchmark_returns : pd.Series
        Benchmark returns by sector
    portfolio_weights : pd.Series
        Portfolio weights by sector
    benchmark_weights : pd.Series
        Benchmark weights by sector
    """
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                 portfolio_weights: pd.Series, benchmark_weights: pd.Series):
        """
        Initialize the Brinson attribution model.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns by sector
        benchmark_returns : pd.Series
            Benchmark returns by sector
        portfolio_weights : pd.Series
            Portfolio weights by sector
        benchmark_weights : pd.Series
            Benchmark weights by sector
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.portfolio_weights = portfolio_weights
        self.benchmark_weights = benchmark_weights
        
        # Calculate total returns
        self.portfolio_total_return = self._calculate_total_return(
            self.portfolio_returns, self.portfolio_weights
        )
        self.benchmark_total_return = self._calculate_total_return(
            self.benchmark_returns, self.benchmark_weights
        )
    
    def _calculate_total_return(self, returns: pd.Series, weights: pd.Series) -> float:
        """Calculate total return from sector returns and weights"""
        return (returns * weights).sum()
    
    def calculate_attribution(self) -> Dict[str, float]:
        """
        Calculate the Brinson attribution effects.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing:
            - 'allocation': Allocation effect
            - 'selection': Selection effect
            - 'interaction': Interaction effect
            - 'total': Total active return
        """
        # Allocation effect
        allocation = self._calculate_allocation_effect()
        
        # Selection effect
        selection = self._calculate_selection_effect()
        
        # Interaction effect
        interaction = self._calculate_interaction_effect()
        
        # Total active return
        total_active = self.portfolio_total_return - self.benchmark_total_return
        
        return {
            'allocation': allocation,
            'selection': selection,
            'interaction': interaction,
            'total': total_active
        }
    
    def _calculate_allocation_effect(self) -> float:
        """
        Calculate the allocation effect.
        
        Allocation effect = Σ(w_p - w_b) * (R_b - R_b_total)
        where:
        - w_p: Portfolio weight
        - w_b: Benchmark weight
        - R_b: Benchmark sector return
        - R_b_total: Total benchmark return
        """
        weight_diff = self.portfolio_weights - self.benchmark_weights
        return_diff = self.benchmark_returns - self.benchmark_total_return
        return (weight_diff * return_diff).sum()
    
    def _calculate_selection_effect(self) -> float:
        """
        Calculate the selection effect.
        
        Selection effect = Σw_b * (R_p - R_b)
        where:
        - w_b: Benchmark weight
        - R_p: Portfolio sector return
        - R_b: Benchmark sector return
        """
        return_diff = self.portfolio_returns - self.benchmark_returns
        return (self.benchmark_weights * return_diff).sum()
    
    def _calculate_interaction_effect(self) -> float:
        """
        Calculate the interaction effect.
        
        Interaction effect = Σ(w_p - w_b) * (R_p - R_b)
        where:
        - w_p: Portfolio weight
        - w_b: Benchmark weight
        - R_p: Portfolio sector return
        - R_b: Benchmark sector return
        """
        weight_diff = self.portfolio_weights - self.benchmark_weights
        return_diff = self.portfolio_returns - self.benchmark_returns
        return (weight_diff * return_diff).sum()
    
    def calculate_attribution_by_sector(self) -> pd.DataFrame:
        """
        Calculate attribution effects by sector.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with sectors as index and attribution effects as columns
        """
        sectors = self.portfolio_returns.index
        
        # Initialize results DataFrame
        results = pd.DataFrame(index=sectors)
        
        # Calculate effects by sector
        results['allocation'] = (self.portfolio_weights - self.benchmark_weights) * \
                              (self.benchmark_returns - self.benchmark_total_return)
        
        results['selection'] = self.benchmark_weights * \
                             (self.portfolio_returns - self.benchmark_returns)
        
        results['interaction'] = (self.portfolio_weights - self.benchmark_weights) * \
                               (self.portfolio_returns - self.benchmark_returns)
        
        results['total'] = results.sum(axis=1)
        
        return results
    
    def plot_attribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the attribution effects.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Calculate attribution by sector
        attribution_by_sector = self.calculate_attribution_by_sector()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot sector attribution
        attribution_by_sector[['allocation', 'selection', 'interaction']].plot(
            kind='bar', stacked=True, ax=ax1
        )
        ax1.set_title('Sector Attribution Effects')
        ax1.set_ylabel('Return Contribution')
        ax1.legend(title='Effect')
        
        # Plot total effects
        total_effects = self.calculate_attribution()
        effects = pd.Series(total_effects)
        effects.plot(kind='bar', ax=ax2)
        ax2.set_title('Total Attribution Effects')
        ax2.set_ylabel('Return Contribution')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_rolling_attribution(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling attribution effects.
        
        Parameters:
        -----------
        window : int
            Rolling window size
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index and rolling attribution effects as columns
        """
        # Initialize results DataFrame
        results = pd.DataFrame(index=self.portfolio_returns.index)
        
        # Calculate rolling effects
        for i in range(window, len(self.portfolio_returns)):
            # Get data for the window
            portfolio_returns = self.portfolio_returns.iloc[i-window:i]
            benchmark_returns = self.benchmark_returns.iloc[i-window:i]
            portfolio_weights = self.portfolio_weights.iloc[i-window:i]
            benchmark_weights = self.benchmark_weights.iloc[i-window:i]
            
            # Create Brinson model for the window
            brinson = BrinsonAttribution(
                portfolio_returns,
                benchmark_returns,
                portfolio_weights,
                benchmark_weights
            )
            
            # Calculate attribution
            attribution = brinson.calculate_attribution()
            
            # Store results
            results.loc[portfolio_returns.index[-1]] = attribution
        
        return results 