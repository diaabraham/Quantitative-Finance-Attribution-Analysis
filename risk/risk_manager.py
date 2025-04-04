import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union

class RiskManager:
    def __init__(self, max_position_size=0.2, max_sector_exposure=0.3, 
                 max_drawdown=0.2, var_confidence_level=0.95,
                 target_volatility=0.15, min_correlation_threshold=0.3,
                 lookback_period=252):
        """
        Initialize the Risk Manager with enhanced parameters
        
        Parameters:
        -----------
        max_position_size : float
            Maximum position size as a fraction of portfolio
        max_sector_exposure : float
            Maximum exposure to any single sector
        max_drawdown : float
            Maximum allowed drawdown before position reduction
        var_confidence_level : float
            Confidence level for Value at Risk calculation
        target_volatility : float
            Target annualized portfolio volatility
        min_correlation_threshold : float
            Minimum correlation threshold for diversification
        lookback_period : int
            Number of days to look back for calculations
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown = max_drawdown
        self.var_confidence_level = var_confidence_level
        self.target_volatility = target_volatility
        self.min_correlation_threshold = min_correlation_threshold
        self.lookback_period = lookback_period
        
    def calculate_var(self, returns, confidence_level=None):
        """
        Calculate Value at Risk using both parametric and historical methods
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns
        confidence_level : float, optional
            Confidence level for VaR calculation
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level
            
        # Historical VaR
        historical_var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        mean = returns.mean()
        std = returns.std()
        parametric_var = norm.ppf(1 - confidence_level, mean, std)
        
        return {
            'historical': historical_var,
            'parametric': parametric_var,
            'combined': (historical_var + parametric_var) / 2
        }
    
    def calculate_expected_shortfall(self, returns, confidence_level=None):
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns
        confidence_level : float, optional
            Confidence level for ES calculation
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level
            
        var = self.calculate_var(returns, confidence_level)['historical']
        return returns[returns <= var].mean()
    
    def calculate_position_weights(self, signals: pd.DataFrame, returns: pd.DataFrame, 
                                 method: str = 'risk_parity') -> pd.DataFrame:
        """
        Calculate position weights using various sophisticated methods
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame of trading signals
        returns : pd.DataFrame
            DataFrame of asset returns
        method : str
            Weighting method to use:
            - 'risk_parity': Equal risk contribution
            - 'min_variance': Minimum variance portfolio
            - 'max_sharpe': Maximum Sharpe ratio portfolio
            - 'equal_risk': Equal risk contribution based on volatility
            - 'black_litterman': Black-Litterman model
            - 'hierarchical_risk_parity': Hierarchical Risk Parity
            - 'mean_variance': Mean-variance optimization
            - 'risk_budgeting': Risk budgeting with custom budgets
        """
        if method == 'risk_parity':
            return self._risk_parity_weights(returns)
        elif method == 'min_variance':
            return self._min_variance_weights(returns)
        elif method == 'max_sharpe':
            return self._max_sharpe_weights(returns)
        elif method == 'equal_risk':
            return self._equal_risk_weights(returns)
        elif method == 'black_litterman':
            return self._black_litterman_weights(returns)
        elif method == 'hierarchical_risk_parity':
            return self._hierarchical_risk_parity_weights(returns)
        elif method == 'mean_variance':
            return self._mean_variance_weights(returns)
        elif method == 'risk_budgeting':
            return self._risk_budgeting_weights(returns)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    
    def _risk_parity_weights(self, returns):
        """Calculate risk parity weights"""
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Define objective function
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_contributions = weights * (cov_matrix @ weights) / portfolio_vol
            return np.sum((risk_contributions - 1/len(weights))**2)
        
        # Optimize weights
        n_assets = len(returns.columns)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, self.max_position_size) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return pd.Series(result.x, index=returns.columns)
    
    def _min_variance_weights(self, returns):
        """Calculate minimum variance weights"""
        cov_matrix = returns.cov()
        
        def objective(weights):
            return weights.T @ cov_matrix @ weights
        
        n_assets = len(returns.columns)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, self.max_position_size) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return pd.Series(result.x, index=returns.columns)
    
    def _max_sharpe_weights(self, returns):
        """Calculate maximum Sharpe ratio weights"""
        cov_matrix = returns.cov()
        expected_returns = returns.mean()
        
        def objective(weights):
            portfolio_return = weights.T @ expected_returns
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return -portfolio_return / portfolio_vol
        
        n_assets = len(returns.columns)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, self.max_position_size) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return pd.Series(result.x, index=returns.columns)
    
    def _equal_risk_weights(self, returns):
        """Calculate equal risk contribution weights"""
        volatilities = returns.std()
        weights = 1 / volatilities
        weights = weights / weights.sum()
        return weights.clip(0, self.max_position_size)
    
    def _black_litterman_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate weights using Black-Litterman model"""
        # Calculate market implied returns
        market_cap = self._get_market_caps(returns.columns)
        market_weights = market_cap / market_cap.sum()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate market implied returns
        risk_aversion = 2.5  # Typical value
        market_implied_returns = risk_aversion * cov_matrix @ market_weights
        
        # Add views (example: relative views)
        views = self._get_views(returns)
        view_confidences = self._get_view_confidences(returns)
        
        # Calculate posterior returns
        posterior_returns = self._calculate_posterior_returns(
            market_implied_returns, views, view_confidences, cov_matrix
        )
        
        # Optimize weights
        weights = self._optimize_weights(posterior_returns, cov_matrix)
        return weights
    
    def _hierarchical_risk_parity_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate weights using Hierarchical Risk Parity"""
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        distance_matrix = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Calculate weights based on clustering
        weights = self._calculate_hrp_weights(linkage_matrix, returns)
        return weights
    
    def _mean_variance_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate weights using mean-variance optimization"""
        # Calculate expected returns and covariance
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Define optimization problem
        def objective(weights):
            portfolio_return = weights.T @ expected_returns
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return -portfolio_return / portfolio_vol
        
        # Optimize weights
        n_assets = len(returns.columns)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, self.max_position_size) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return pd.Series(result.x, index=returns.columns)
    
    def _risk_budgeting_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate weights using risk budgeting"""
        # Define risk budgets (example: based on market cap)
        market_caps = self._get_market_caps(returns.columns)
        risk_budgets = market_caps / market_caps.sum()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Define optimization problem
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_contributions = weights * (cov_matrix @ weights) / portfolio_vol
            return np.sum((risk_contributions - risk_budgets)**2)
        
        # Optimize weights
        n_assets = len(returns.columns)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0, self.max_position_size) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return pd.Series(result.x, index=returns.columns)
    
    def calculate_risk_metrics(self, returns: pd.Series, 
                             benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics with enhanced analysis
        
        Parameters:
        -----------
        returns : pd.Series
            Series of strategy returns
        benchmark_returns : pd.Series, optional
            Series of benchmark returns
        """
        metrics = {}
        
        # Basic metrics
        metrics['Annualized Return'] = returns.mean() * 252
        metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
        metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility']
        
        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        metrics['Maximum Drawdown'] = drawdown.min()
        
        # Value at Risk
        var_results = self.calculate_var(returns)
        metrics['VaR (95%) - Historical'] = var_results['historical']
        metrics['VaR (95%) - Parametric'] = var_results['parametric']
        metrics['VaR (95%) - Combined'] = var_results['combined']
        
        # Expected Shortfall
        metrics['Expected Shortfall (95%)'] = self.calculate_expected_shortfall(returns)
        
        # Higher moments
        metrics['Skewness'] = skew(returns)
        metrics['Kurtosis'] = kurtosis(returns)
        
        # Tail risk metrics
        metrics['Tail Ratio'] = abs(returns[returns > 0].mean() / returns[returns < 0].mean())
        metrics['Gain to Pain Ratio'] = returns.sum() / abs(returns[returns < 0].sum())
        
        # Additional risk metrics
        metrics['Sortino Ratio'] = self._calculate_sortino_ratio(returns)
        metrics['Calmar Ratio'] = self._calculate_calmar_ratio(returns)
        metrics['Omega Ratio'] = self._calculate_omega_ratio(returns)
        metrics['Information Ratio'] = self._calculate_information_ratio(returns, benchmark_returns)
        metrics['Treynor Ratio'] = self._calculate_treynor_ratio(returns, benchmark_returns)
        
        # Risk-adjusted return metrics
        metrics['M2 Ratio'] = self._calculate_m2_ratio(returns, benchmark_returns)
        metrics['M3 Ratio'] = self._calculate_m3_ratio(returns, benchmark_returns)
        metrics['M4 Ratio'] = self._calculate_m4_ratio(returns, benchmark_returns)
        
        # Drawdown analysis
        drawdown_metrics = self._analyze_drawdowns(returns)
        metrics.update(drawdown_metrics)
        
        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(returns, benchmark_returns)
        metrics.update(rolling_metrics)
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.nan
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.nan
        return returns.mean() * np.sqrt(252) / downside_std
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        max_drawdown = self.calculate_drawdown((1 + returns).cumprod()).min()
        if max_drawdown == 0:
            return np.nan
        return returns.mean() * 252 / abs(max_drawdown)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        if losses == 0:
            return np.nan
        return gains / losses
    
    def _calculate_information_ratio(self, returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series]) -> float:
        """Calculate Information ratio"""
        if benchmark_returns is None:
            return np.nan
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        if tracking_error == 0:
            return np.nan
        return excess_returns.mean() * 252 / tracking_error
    
    def _calculate_treynor_ratio(self, returns: pd.Series, 
                                benchmark_returns: Optional[pd.Series]) -> float:
        """Calculate Treynor ratio"""
        if benchmark_returns is None:
            return np.nan
        beta = np.cov(returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
        if beta == 0:
            return np.nan
        return (returns.mean() * 252) / beta
    
    def _calculate_m2_ratio(self, returns: pd.Series, 
                           benchmark_returns: Optional[pd.Series]) -> float:
        """Calculate M2 ratio"""
        if benchmark_returns is None:
            return np.nan
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        strategy_vol = returns.std() * np.sqrt(252)
        if strategy_vol == 0:
            return np.nan
        return (returns.mean() * 252) * (benchmark_vol / strategy_vol)
    
    def _calculate_m3_ratio(self, returns: pd.Series, 
                           benchmark_returns: Optional[pd.Series]) -> float:
        """Calculate M3 ratio"""
        if benchmark_returns is None:
            return np.nan
        benchmark_skew = skew(benchmark_returns)
        strategy_skew = skew(returns)
        if strategy_skew == 0:
            return np.nan
        return (returns.mean() * 252) * (benchmark_skew / strategy_skew)
    
    def _calculate_m4_ratio(self, returns: pd.Series, 
                           benchmark_returns: Optional[pd.Series]) -> float:
        """Calculate M4 ratio"""
        if benchmark_returns is None:
            return np.nan
        benchmark_kurt = kurtosis(benchmark_returns)
        strategy_kurt = kurtosis(returns)
        if strategy_kurt == 0:
            return np.nan
        return (returns.mean() * 252) * (benchmark_kurt / strategy_kurt)
    
    def _analyze_drawdowns(self, returns: pd.Series) -> Dict[str, float]:
        """Analyze drawdown characteristics"""
        cum_returns = (1 + returns).cumprod()
        drawdown = self.calculate_drawdown(cum_returns)
        
        metrics = {}
        metrics['Max Drawdown'] = drawdown.min()
        metrics['Avg Drawdown'] = drawdown[drawdown < 0].mean()
        metrics['Drawdown Duration'] = len(drawdown[drawdown < 0])
        metrics['Recovery Time'] = self._calculate_recovery_time(drawdown)
        
        return metrics
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate average recovery time from drawdowns"""
        recovery_times = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                recovery_times.append(i - start_idx)
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_rolling_metrics(self, returns: pd.Series, 
                                 benchmark_returns: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate rolling risk metrics"""
        window = min(self.lookback_period, len(returns))
        
        metrics = {}
        metrics['Rolling Sharpe'] = returns.rolling(window).mean() / returns.rolling(window).std()
        metrics['Rolling Volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        if benchmark_returns is not None:
            metrics['Rolling Beta'] = returns.rolling(window).cov(benchmark_returns) / benchmark_returns.rolling(window).var()
            metrics['Rolling Alpha'] = returns.rolling(window).mean() - metrics['Rolling Beta'] * benchmark_returns.rolling(window).mean()
        
        return metrics
    
    def plot_risk_metrics(self, returns: pd.Series, 
                         benchmark_returns: Optional[pd.Series] = None):
        """
        Create comprehensive risk visualization with enhanced analysis
        
        Parameters:
        -----------
        returns : pd.Series
            Series of strategy returns
        benchmark_returns : pd.Series, optional
            Series of benchmark returns
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Returns distribution
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2)
        sns.histplot(returns, kde=True, ax=ax1)
        ax1.set_title('Returns Distribution')
        ax1.axvline(0, color='r', linestyle='--')
        
        # QQ plot
        ax2 = plt.subplot2grid((4, 3), (0, 2))
        self._plot_qq(returns, ax2)
        ax2.set_title('QQ Plot')
        
        # Cumulative returns
        ax3 = plt.subplot2grid((4, 3), (1, 0), colspan=3)
        cum_returns = (1 + returns).cumprod()
        ax3.plot(cum_returns.index, cum_returns, label='Strategy')
        if benchmark_returns is not None:
            cum_benchmark = (1 + benchmark_returns).cumprod()
            ax3.plot(cum_benchmark.index, cum_benchmark, label='Benchmark')
        ax3.set_title('Cumulative Returns')
        ax3.legend()
        
        # Drawdown
        ax4 = plt.subplot2grid((4, 3), (2, 0), colspan=3)
        drawdown = self.calculate_drawdown(cum_returns)
        ax4.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3)
        ax4.set_title('Drawdown')
        
        # Rolling metrics
        ax5 = plt.subplot2grid((4, 3), (3, 0))
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        ax5.plot(rolling_vol.index, rolling_vol)
        ax5.axhline(self.target_volatility, color='r', linestyle='--')
        ax5.set_title('Rolling Volatility')
        
        ax6 = plt.subplot2grid((4, 3), (3, 1))
        rolling_sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std() * np.sqrt(252)
        ax6.plot(rolling_sharpe.index, rolling_sharpe)
        ax6.set_title('Rolling Sharpe Ratio')
        
        if benchmark_returns is not None:
            ax7 = plt.subplot2grid((4, 3), (3, 2))
            rolling_beta = returns.rolling(window=20).cov(benchmark_returns) / benchmark_returns.rolling(window=20).var()
            ax7.plot(rolling_beta.index, rolling_beta)
            ax7.set_title('Rolling Beta')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_qq(self, returns: pd.Series, ax: plt.Axes):
        """Plot QQ plot for returns"""
        from scipy.stats import probplot
        probplot(returns, dist='norm', plot=ax)
        ax.set_title('QQ Plot')
        ax.grid(True)
    
    def calculate_drawdown(self, portfolio_value):
        """
        Calculate portfolio drawdown
        
        Parameters:
        -----------
        portfolio_value : pd.Series
            Series of portfolio values
        """
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        return drawdown
    
    def check_sector_exposure(self, weights, sector_mapping):
        """
        Check and adjust sector exposures
        
        Parameters:
        -----------
        weights : pd.DataFrame
            DataFrame of position weights
        sector_mapping : dict
            Mapping of assets to sectors
        """
        sector_exposure = pd.DataFrame(index=weights.index)
        
        # Calculate sector exposures
        for sector, assets in sector_mapping.items():
            sector_exposure[sector] = weights[assets].sum(axis=1)
            
        # Check for violations
        violations = sector_exposure > self.max_sector_exposure
        
        if violations.any().any():
            print("Warning: Sector exposure limits exceeded")
            # Here you would implement logic to adjust weights
            # This is a simplified version
            weights = weights * (1 - violations.astype(int))
            weights = weights.div(weights.sum(axis=1), axis=0)
            
        return weights
    
    def check_drawdown_limit(self, portfolio_value):
        """
        Check if drawdown exceeds limit
        
        Parameters:
        -----------
        portfolio_value : pd.Series
            Series of portfolio values
        """
        drawdown = self.calculate_drawdown(portfolio_value)
        if drawdown.min() < -self.max_drawdown:
            print(f"Warning: Maximum drawdown limit ({self.max_drawdown}) exceeded")
            return True
        return False 