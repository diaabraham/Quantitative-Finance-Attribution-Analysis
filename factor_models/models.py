import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class FactorModel:
    """Base class for factor models."""
    
    def __init__(self, returns: pd.Series, factor_returns: pd.DataFrame):
        """
        Initialize the factor model.
        
        Args:
            returns: Portfolio returns series
            factor_returns: DataFrame of factor returns
        """
        self.returns = returns
        self.factor_returns = factor_returns
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.exposures = None
        self.residuals = None
        self.r_squared = None
        
    def estimate_exposures(self) -> pd.Series:
        """Estimate factor exposures using linear regression."""
        # Scale the factor returns
        scaled_factors = self.scaler.fit_transform(self.factor_returns)
        
        # Fit the model
        self.model.fit(scaled_factors, self.returns)
        
        # Store exposures
        self.exposures = pd.Series(
            self.model.coef_,
            index=self.factor_returns.columns
        )
        
        # Calculate residuals and R-squared
        self.residuals = self.returns - self.model.predict(scaled_factors)
        self.r_squared = self.model.score(scaled_factors, self.returns)
        
        return self.exposures
    
    def calculate_contributions(self) -> pd.DataFrame:
        """Calculate factor contributions to returns."""
        if self.exposures is None:
            self.estimate_exposures()
            
        contributions = pd.DataFrame(
            self.factor_returns * self.exposures.values,
            index=self.factor_returns.index,
            columns=self.factor_returns.columns
        )
        
        return contributions
    
    def calculate_r_squared(self) -> float:
        """Calculate the R-squared of the model."""
        if self.r_squared is None:
            self.estimate_exposures()
        return self.r_squared

class FamaFrenchModel(FactorModel):
    """Fama-French 3-Factor Model implementation."""
    
    def __init__(self, returns: pd.Series, factor_returns: pd.DataFrame):
        """
        Initialize Fama-French model.
        
        Args:
            returns: Portfolio returns series
            factor_returns: DataFrame with columns ['market', 'size', 'value']
        """
        required_factors = ['market', 'size', 'value']
        if not all(factor in factor_returns.columns for factor in required_factors):
            raise ValueError("Factor returns must include 'market', 'size', and 'value' factors")
            
        super().__init__(returns, factor_returns[required_factors])
        
    def calculate_alpha(self) -> float:
        """Calculate the portfolio's alpha."""
        if self.exposures is None:
            self.estimate_exposures()
        return self.model.intercept_

class BarraModel(FactorModel):
    """BARRA Factor Model implementation."""
    
    def __init__(self, returns: pd.Series, factor_returns: pd.DataFrame):
        """
        Initialize BARRA model.
        
        Args:
            returns: Portfolio returns series
            factor_returns: DataFrame of BARRA factor returns
        """
        super().__init__(returns, factor_returns)
        
    def calculate_risk_contributions(self) -> pd.Series:
        """Calculate risk contributions of each factor."""
        if self.exposures is None:
            self.estimate_exposures()
            
        # Calculate factor covariance matrix
        factor_cov = self.factor_returns.cov()
        
        # Calculate portfolio variance
        portfolio_variance = self.exposures @ factor_cov @ self.exposures
        
        # Calculate marginal contributions to risk
        marginal_contributions = (factor_cov @ self.exposures) / np.sqrt(portfolio_variance)
        
        # Calculate risk contributions
        risk_contributions = self.exposures * marginal_contributions
        
        return risk_contributions

class AxiomaModel(FactorModel):
    """Axioma Factor Model implementation."""
    
    def __init__(self, returns: pd.Series, factor_returns: pd.DataFrame):
        """
        Initialize Axioma model.
        
        Args:
            returns: Portfolio returns series
            factor_returns: DataFrame of Axioma factor returns
        """
        super().__init__(returns, factor_returns)
        
    def calculate_specific_risk(self) -> float:
        """Calculate the specific risk of the portfolio."""
        if self.residuals is None:
            self.estimate_exposures()
        return np.std(self.residuals)
    
    def calculate_active_risk(self) -> float:
        """Calculate the active risk of the portfolio."""
        if self.exposures is None:
            self.estimate_exposures()
            
        # Calculate factor covariance matrix
        factor_cov = self.factor_returns.cov()
        
        # Calculate active risk
        active_risk = np.sqrt(self.exposures @ factor_cov @ self.exposures)
        
        return active_risk 