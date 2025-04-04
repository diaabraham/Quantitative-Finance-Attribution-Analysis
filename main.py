import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, MACD
from ta.trend import SMAIndicator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from data.data_manager import DataManager
from risk.risk_manager import RiskManager

class SectorRotationAlpha:
    def __init__(self, transaction_cost=0.001, max_position_size=0.2, 
                 max_sector_exposure=0.3, max_drawdown=0.2,
                 target_volatility=0.15, min_correlation_threshold=0.3):
        """
        Initialize the Sector Rotation Alpha
        
        Parameters:
        -----------
        transaction_cost : float
            Cost per transaction as a fraction of trade value
        max_position_size : float
            Maximum position size as a fraction of portfolio
        max_sector_exposure : float
            Maximum exposure to any single sector
        max_drawdown : float
            Maximum allowed drawdown before position reduction
        target_volatility : float
            Target annualized portfolio volatility
        min_correlation_threshold : float
            Minimum correlation threshold for diversification
        """
        self.data_manager = DataManager()
        self.risk_manager = RiskManager(
            max_position_size=max_position_size,
            max_sector_exposure=max_sector_exposure,
            max_drawdown=max_drawdown,
            target_volatility=target_volatility,
            min_correlation_threshold=min_correlation_threshold
        )
        self.transaction_cost = transaction_cost
        self.sector_etfs = None
        self.data = None
        self.momentum_scores = None
        self.performance_metrics = None
        
    def fetch_data(self, start_date, end_date):
        """
        Fetch historical price data for all sector ETFs
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        """
        # Get sector ETF information
        self.sector_etfs = self.data_manager.get_sector_etfs()
        
        # Fetch historical data
        symbols = [info['symbol'] for info in self.sector_etfs.values()]
        self.data = self.data_manager.fetch_historical_data(symbols, start_date, end_date)
        
        # Fetch benchmark data
        self.benchmark_data = self.data_manager.get_market_data(start_date=start_date, end_date=end_date)
        
        return self.data
    
    def calculate_momentum_indicators(self):
        """
        Calculate momentum indicators for each sector
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please fetch data first.")
            
        momentum_scores = pd.DataFrame(index=self.data.index)
        
        for sector, info in self.sector_etfs.items():
            symbol = info['symbol']
            # Calculate RSI
            rsi = RSIIndicator(close=self.data[symbol], window=14)
            rsi_score = rsi.rsi()
            
            # Calculate MACD
            macd = MACD(close=self.data[symbol])
            macd_score = macd.macd_diff()
            
            # Calculate Price Momentum
            price_momentum = self.data[symbol].pct_change(periods=20)
            
            # Calculate Volatility
            volatility = self.data[symbol].pct_change().rolling(window=20).std()
            
            # Combine indicators into a single score
            momentum_scores[symbol] = (
                0.3 * (rsi_score / 100) +  # Normalize RSI to 0-1
                0.3 * (macd_score / macd_score.std()) +  # Normalize MACD
                0.2 * price_momentum +
                0.2 * (1 / volatility)  # Inverse volatility weighting
            )
            
        self.momentum_scores = momentum_scores
        return momentum_scores
    
    def generate_signals(self, top_n=3):
        """
        Generate trading signals based on momentum scores
        
        Parameters:
        -----------
        top_n : int
            Number of top sectors to select
        """
        if self.momentum_scores is None:
            raise ValueError("Momentum scores not calculated. Please run calculate_momentum_indicators first.")
            
        # Rank sectors by momentum score
        ranks = self.momentum_scores.rank(axis=1, ascending=False)
        
        # Create signals (1 for top N sectors, 0 otherwise)
        signals = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)
        for symbol in ranks.columns:
            signals[symbol] = (ranks[symbol] <= top_n).astype(int)
            
        return signals
    
    def backtest(self, signals, initial_capital=100000, weighting_method='risk_parity'):
        """
        Backtest the strategy with transaction costs
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame containing trading signals
        initial_capital : float
            Initial capital for backtesting
        weighting_method : str
            Method to use for position sizing ('risk_parity', 'min_variance', 'max_sharpe', 'equal_risk')
        """
        # Calculate daily returns
        returns = self.data.pct_change()
        
        # Calculate position weights with risk management
        weights = self.risk_manager.calculate_position_weights(signals, returns, method=weighting_method)
        
        # Check sector exposure limits
        sector_mapping = {sector: [info['symbol']] for sector, info in self.sector_etfs.items()}
        weights = self.risk_manager.check_sector_exposure(weights, sector_mapping)
        
        # Calculate strategy returns
        strategy_returns = (weights * returns).sum(axis=1)
        
        # Calculate transaction costs
        position_changes = weights.diff().abs()
        transaction_costs = (position_changes * self.transaction_cost).sum(axis=1)
        strategy_returns -= transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate portfolio value
        portfolio_value = initial_capital * cumulative_returns
        
        # Check drawdown limits
        if self.risk_manager.check_drawdown_limit(portfolio_value):
            # Implement drawdown protection (e.g., reduce position sizes)
            weights = weights * 0.5  # Example: reduce positions by 50%
            
        # Calculate performance metrics
        benchmark_returns = self.benchmark_data.pct_change()
        self.performance_metrics = self.risk_manager.calculate_risk_metrics(
            strategy_returns, 
            benchmark_returns
        )
        
        return portfolio_value, strategy_returns, weights
    
    def plot_results(self, portfolio_value, weights, strategy_returns):
        """
        Plot backtest results and portfolio composition
        
        Parameters:
        -----------
        portfolio_value : pd.Series
            Series containing portfolio values over time
        weights : pd.DataFrame
            DataFrame containing position weights over time
        strategy_returns : pd.Series
            Series containing strategy returns
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Portfolio performance
        ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
        ax1.plot(portfolio_value.index, portfolio_value, label='Strategy')
        if hasattr(self, 'benchmark_data'):
            benchmark_value = self.benchmark_data / self.benchmark_data.iloc[0] * portfolio_value.iloc[0]
            ax1.plot(benchmark_value.index, benchmark_value, label='Benchmark')
        ax1.set_title('Portfolio Performance')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Portfolio composition
        ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
        weights.plot.area(ax=ax2)
        ax2.set_title('Portfolio Composition')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Weight')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Risk metrics visualization
        self.risk_manager.plot_risk_metrics(strategy_returns, self.benchmark_data.pct_change())
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        for metric, value in self.performance_metrics.items():
            print(f"{metric}: {value:.4f}")

# Example usage
if __name__ == "__main__":
    # Initialize the strategy with risk parameters
    strategy = SectorRotationAlpha(
        transaction_cost=0.001,  # 0.1% per transaction
        max_position_size=0.2,   # Maximum 20% in any single position
        max_sector_exposure=0.3, # Maximum 30% in any single sector
        max_drawdown=0.2,        # Maximum 20% drawdown
        target_volatility=0.15,  # Target 15% annualized volatility
        min_correlation_threshold=0.3  # Minimum correlation for diversification
    )
    
    # Set date range (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Fetch data
    data = strategy.fetch_data(start_date.strftime('%Y-%m-%d'), 
                             end_date.strftime('%Y-%m-%d'))
    
    # Calculate momentum indicators
    momentum_scores = strategy.calculate_momentum_indicators()
    
    # Generate signals
    signals = strategy.generate_signals(top_n=3)
    
    # Backtest with different weighting methods
    weighting_methods = ['risk_parity', 'min_variance', 'max_sharpe', 'equal_risk']
    results = {}
    
    for method in weighting_methods:
        print(f"\nBacktesting with {method} weighting:")
        portfolio_value, strategy_returns, weights = strategy.backtest(signals, weighting_method=method)
        results[method] = {
            'portfolio_value': portfolio_value,
            'strategy_returns': strategy_returns,
            'weights': weights
        }
        strategy.plot_results(portfolio_value, weights, strategy_returns)
