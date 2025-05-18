import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models
from pypfopt.hierarchical_portfolio import HRPOpt

def compute_hrp_portfolio(prices):
    """
    Compute the optimal portfolio weights using Hierarchical Risk Parity
    
    Parameters:
    - prices: DataFrame of asset prices
    
    Returns: weights, performance metrics, hrp object, expected returns, covariance matrix
    """
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    
    # Initialize HRP model
    hrp = HRPOpt(returns=prices.pct_change().dropna())
    
    # Optimize
    weights = hrp.optimize()
    
    # Calculate performance metrics
    portfolio_return = np.sum(mu * np.array(list(weights.values())))
    portfolio_volatility = np.sqrt(
        np.dot(np.array(list(weights.values())).T, 
               np.dot(S, np.array(list(weights.values()))))
    )
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    performance = (portfolio_return, portfolio_volatility, sharpe_ratio)
    
    return weights, performance, hrp, mu, S