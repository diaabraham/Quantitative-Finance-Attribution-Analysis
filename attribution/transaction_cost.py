import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

class TransactionCostAttribution:
    """
    Implementation of transaction cost attribution analysis.
    
    This class provides methods to:
    1. Calculate and decompose transaction costs
    2. Analyze market impact
    3. Calculate implementation shortfall
    4. Optimize trading schedules
    
    Attributes:
    -----------
    trades : pd.DataFrame
        Trade data with columns: ['timestamp', 'symbol', 'quantity', 'price', 'side']
    market_data : pd.DataFrame
        Market data with columns: ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    cost_params : Dict[str, float]
        Transaction cost parameters
    """
    
    def __init__(self, trades: pd.DataFrame, market_data: pd.DataFrame,
                 cost_params: Optional[Dict[str, float]] = None):
        """
        Initialize the transaction cost attribution model.
        
        Parameters:
        -----------
        trades : pd.DataFrame
            Trade data
        market_data : pd.DataFrame
            Market data
        cost_params : Optional[Dict[str, float]]
            Transaction cost parameters with keys:
            - 'fixed_cost': Fixed cost per trade
            - 'variable_cost': Variable cost as percentage of trade value
            - 'market_impact': Market impact coefficient
            - 'opportunity_cost': Opportunity cost coefficient
        """
        self.trades = trades
        self.market_data = market_data
        
        # Set default cost parameters if not provided
        self.cost_params = cost_params or {
            'fixed_cost': 0.0001,  # $0.01 per share
            'variable_cost': 0.0005,  # 5 bps
            'market_impact': 0.0001,  # 1 bp per 1% of ADV
            'opportunity_cost': 0.0002  # 2 bps per day
        }
    
    def calculate_total_costs(self) -> pd.DataFrame:
        """
        Calculate total transaction costs.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with costs broken down by component
        """
        # Initialize results DataFrame
        results = pd.DataFrame(index=self.trades.index)
        
        # Calculate fixed costs
        results['fixed_cost'] = self.cost_params['fixed_cost'] * \
                              abs(self.trades['quantity'])
        
        # Calculate variable costs
        results['variable_cost'] = self.cost_params['variable_cost'] * \
                                 abs(self.trades['quantity'] * self.trades['price'])
        
        # Calculate market impact
        results['market_impact'] = self._calculate_market_impact()
        
        # Calculate opportunity cost
        results['opportunity_cost'] = self._calculate_opportunity_cost()
        
        # Calculate total cost
        results['total_cost'] = results.sum(axis=1)
        
        return results
    
    def _calculate_market_impact(self) -> pd.Series:
        """
        Calculate market impact costs.
        
        Returns:
        --------
        pd.Series
            Market impact costs
        """
        # Calculate average daily volume
        adv = self.market_data.groupby('symbol')['volume'].mean()
        
        # Calculate trade size as percentage of ADV
        trade_size_pct = abs(self.trades['quantity']) / \
                        self.trades['symbol'].map(adv)
        
        # Calculate market impact
        return self.cost_params['market_impact'] * trade_size_pct * \
               abs(self.trades['quantity'] * self.trades['price'])
    
    def _calculate_opportunity_cost(self) -> pd.Series:
        """
        Calculate opportunity costs.
        
        Returns:
        --------
        pd.Series
            Opportunity costs
        """
        # Calculate days to execute
        days_to_execute = self._calculate_days_to_execute()
        
        # Calculate opportunity cost
        return self.cost_params['opportunity_cost'] * days_to_execute * \
               abs(self.trades['quantity'] * self.trades['price'])
    
    def _calculate_days_to_execute(self) -> pd.Series:
        """
        Calculate days to execute for each trade.
        
        Returns:
        --------
        pd.Series
            Days to execute
        """
        # Calculate average daily volume
        adv = self.market_data.groupby('symbol')['volume'].mean()
        
        # Calculate days to execute
        return abs(self.trades['quantity']) / self.trades['symbol'].map(adv)
    
    def calculate_implementation_shortfall(self) -> pd.DataFrame:
        """
        Calculate implementation shortfall.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with implementation shortfall components
        """
        # Initialize results DataFrame
        results = pd.DataFrame(index=self.trades.index)
        
        # Calculate arrival price
        arrival_price = self._calculate_arrival_price()
        
        # Calculate execution price
        execution_price = self.trades['price']
        
        # Calculate price impact
        results['price_impact'] = (execution_price - arrival_price) * \
                                self.trades['quantity']
        
        # Calculate delay cost
        results['delay_cost'] = self._calculate_delay_cost(arrival_price)
        
        # Calculate total implementation shortfall
        results['total_shortfall'] = results.sum(axis=1)
        
        return results
    
    def _calculate_arrival_price(self) -> pd.Series:
        """
        Calculate arrival price for each trade.
        
        Returns:
        --------
        pd.Series
            Arrival prices
        """
        # Get market data at trade time
        trade_times = self.trades['timestamp']
        market_data_at_trade = self.market_data.loc[
            self.market_data['timestamp'].isin(trade_times)
        ]
        
        # Calculate arrival price (using VWAP)
        arrival_price = market_data_at_trade.groupby('timestamp').apply(
            lambda x: (x['close'] * x['volume']).sum() / x['volume'].sum()
        )
        
        return arrival_price
    
    def _calculate_delay_cost(self, arrival_price: pd.Series) -> pd.Series:
        """
        Calculate delay cost.
        
        Parameters:
        -----------
        arrival_price : pd.Series
            Arrival prices
        
        Returns:
        --------
        pd.Series
            Delay costs
        """
        # Calculate days to execute
        days_to_execute = self._calculate_days_to_execute()
        
        # Calculate price change during delay
        price_change = self.trades['price'] - arrival_price
        
        # Calculate delay cost
        return price_change * self.trades['quantity'] * days_to_execute
    
    def optimize_trading_schedule(self, target_quantity: float,
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
        # Calculate optimal trade size
        adv = self.market_data['volume'].mean()
        optimal_trade_size = adv * 0.1  # 10% of ADV
        
        # Calculate number of trades
        num_trades = int(np.ceil(abs(target_quantity) / optimal_trade_size))
        
        # Create trading schedule
        schedule = pd.DataFrame(index=range(num_trades))
        
        # Calculate trade quantities
        schedule['quantity'] = np.sign(target_quantity) * optimal_trade_size
        schedule.loc[num_trades-1, 'quantity'] = target_quantity - \
                                                schedule['quantity'].sum() + \
                                                schedule.loc[num_trades-1, 'quantity']
        
        # Calculate trade times
        schedule['timestamp'] = pd.date_range(
            start=self.market_data.index[0],
            periods=num_trades,
            freq=f'{time_horizon}D'
        )
        
        return schedule
    
    def calculate_rolling_costs(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling transaction costs.
        
        Parameters:
        -----------
        window : int
            Rolling window size
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling cost metrics
        """
        # Initialize results DataFrame
        metrics = ['fixed_cost', 'variable_cost', 'market_impact',
                  'opportunity_cost', 'total_cost']
        results = pd.DataFrame(index=self.trades.index, columns=metrics)
        
        # Calculate rolling costs
        for i in range(window, len(self.trades)):
            # Get data for the window
            trades_window = self.trades.iloc[i-window:i]
            market_data_window = self.market_data.loc[
                self.market_data.index.isin(trades_window['timestamp'])
            ]
            
            # Create transaction cost model for the window
            cost_model = TransactionCostAttribution(
                trades_window,
                market_data_window,
                self.cost_params
            )
            
            # Calculate costs
            costs = cost_model.calculate_total_costs()
            
            # Store results
            results.iloc[i] = costs.mean()
        
        return results
    
    def plot_cost_attribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot cost attribution results.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot cost components
        costs = self.calculate_total_costs()
        costs[['fixed_cost', 'variable_cost', 'market_impact',
              'opportunity_cost']].plot(kind='bar', stacked=True, ax=ax1)
        ax1.set_title('Transaction Cost Components')
        ax1.set_ylabel('Cost ($)')
        ax1.legend(title='Component')
        
        # Plot implementation shortfall
        shortfall = self.calculate_implementation_shortfall()
        shortfall[['price_impact', 'delay_cost']].plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title('Implementation Shortfall Components')
        ax2.set_ylabel('Cost ($)')
        ax2.legend(title='Component')
        
        plt.tight_layout()
        plt.show() 