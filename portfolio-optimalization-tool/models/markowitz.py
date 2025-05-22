import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import streamlit as st

def compute_optimal_portfolio(prices, strategy='sharpe'):
    """
    Compute optimal portfolio weights using Markowitz model
    
    Parameters:
    - prices: DataFrame of asset prices (columns = assets, index = dates)
    - strategy: Optimization strategy ('sharpe', 'min_volatility', or 'max_return')
    
    Returns:
    - weights: Dict of optimal weights
    - performance: Tuple of (expected return, volatility, Sharpe ratio)
    - ef: EfficientFrontier object
    """
    try:
        # Validate input
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Prices must be a pandas DataFrame")
        if prices.empty:
            raise ValueError("Price data is empty")
        if prices.isnull().any().any():
            st.warning("Price data contains missing values. They will be forward/backward filled.")
            prices = prices.fillna(method='ffill').fillna(method='bfill')
            
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        
        # Validate expected returns and covariance
        if mu.isnull().any() or S.isnull().any().any():
            raise ValueError("Could not compute expected returns or covariance matrix")
        
        # Create efficient frontier object
        ef = EfficientFrontier(mu, S)
        
        # Apply selected optimization strategy
        try:
            if strategy == 'sharpe':
                ef.max_sharpe()
            elif strategy == 'min_volatility':
                ef.min_volatility()
            elif strategy == 'max_return':
                # For maximum return, invest everything in the asset with highest expected return
                max_return_asset = mu.idxmax()
                weights = {asset: 1.0 if asset == max_return_asset else 0.0 for asset in mu.index}
                ef._w = np.array([weights[asset] for asset in mu.index])
            else:
                raise ValueError(f"Unknown optimization strategy: {strategy}")
            
            # Get weights and clean them
            if strategy != 'max_return':
                weights = ef.clean_weights()
            
            # Get portfolio performance
            performance = ef.portfolio_performance()
            
            # Validate results
            if not weights or sum(weights.values()) < 0.99 or sum(weights.values()) > 1.01:
                raise ValueError("Optimization failed to produce valid weights")
                
            return weights, performance, ef
            
        except Exception as e:
            st.error(f"Portfolio optimization failed: {str(e)}")
            # Try minimum volatility as fallback
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            weights = ef.clean_weights()
            performance = ef.portfolio_performance()
            st.warning("Falling back to minimum volatility portfolio")
            return weights, performance, ef
            
    except Exception as e:
        st.error(f"Error in portfolio computation: {str(e)}")
        # Return equal weights as last resort
        n_assets = len(prices.columns)
        equal_weights = {col: 1.0/n_assets for col in prices.columns}
        return equal_weights, (0, 0, 0), None