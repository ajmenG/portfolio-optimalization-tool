import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier, BlackLittermanModel

def compute_black_litterman_portfolio(prices, views=None):
    """
    Compute the optimal portfolio weights using the Black-Litterman model
    
    Parameters:
    - prices: DataFrame of asset prices
    - views: Dictionary mapping assets to tuples of (expected return, confidence)
    
    Returns: weights, performance metrics, ef object, posterior returns, covariance matrix
    """
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    
    # Calculate market implied risk aversion manually
    market_prices = prices.mean(axis=1)
    market_returns = market_prices.pct_change().dropna()
    market_var = market_returns.var() * 252  # Annualized variance
    market_mean = market_returns.mean() * 252  # Annualized return
    risk_free_rate = 0.02  # Assume a risk-free rate 
    delta = (market_mean - risk_free_rate) / market_var  # Market risk aversion
    
    # Initialize Black-Litterman model with market implied prior
    market_prior = delta * S.dot(pd.Series(1/len(prices.columns), index=S.index))
    bl = BlackLittermanModel(S, pi=market_prior)
    
    # Add views if provided
    if views:
        view_dict = {}
        confidences = []
        for asset, (expected_return, confidence) in views.items():
            view_dict[asset] = expected_return
            confidences.append(confidence)
            
        confidence = np.array(confidences) if confidences else None
        bl.add_views(view_dict, confidence=confidence)
    
    # Get posterior estimate
    posterior_returns = bl.bl_returns()
    posterior_cov = bl.bl_cov()
    
    # Optimize portfolio
    ef = EfficientFrontier(posterior_returns, posterior_cov)
    ef.max_sharpe()
    weights = ef.clean_weights()
    
    # Get performance metrics
    performance = ef.portfolio_performance()
    
    return weights, performance, ef, posterior_returns, posterior_cov