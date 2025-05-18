import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier
from scipy.optimize import minimize

def compute_risk_parity_portfolio(prices, target_risk=0.1):
    """
    Compute portfolio weights using the risk parity approach
    
    Parameters:
    - prices: DataFrame of asset prices
    - target_risk: Target portfolio volatility
    
    Returns: weights, performance metrics, ef object, expected returns, covariance matrix
    """
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    
    # Number of assets
    n = len(mu)
    
    # Risk parity objective function
    def risk_budget_objective(weights, args):
        cov = args[0]  # Covariance matrix
        
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        risk_contribution = weights * np.dot(cov, weights) / portfolio_risk
        
        # Want all risk contributions to be equal
        target_risk_contribution = portfolio_risk / n
        diff = risk_contribution - target_risk_contribution
        
        return np.sum(np.square(diff))
    
    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(n))  # 0 <= weight <= 1
    
    # Initial guess (equal weights)
    init_weights = np.array([1 / n] * n)
    
    # Optimize
    result = minimize(
        risk_budget_objective,
        init_weights,
        args=[S],
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    
    if not result.success:
        raise Exception("Risk parity optimization failed")
    
    # Get optimal weights
    weights = dict(zip(mu.index, result.x))
    
    # Create efficient frontier object for convenience
    ef = EfficientFrontier(mu, S)
    ef._w = result.x  # Set the weights directly
    
    # Get performance
    performance = (
        np.sum(mu * result.x),  # Expected return
        np.sqrt(np.dot(result.x.T, np.dot(S, result.x))),  # Volatility
        np.sum(mu * result.x) / np.sqrt(np.dot(result.x.T, np.dot(S, result.x)))  # Sharpe
    )
    
    return weights, performance, ef, mu, S

def compute_constrained_portfolio(prices, min_weight=0.01, max_weight=0.25, sector_constraints=None):
    """
    Compute the optimal portfolio with constraints on weights and sector allocations
    
    Parameters:
    - prices: DataFrame of asset prices
    - min_weight: Minimum weight for each asset
    - max_weight: Maximum weight for each asset
    - sector_constraints: Dict mapping sectors to (tickers, min_alloc, max_alloc)
    
    Returns: weights, performance metrics, ef object, expected returns, covariance matrix
    """
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    
    # Optimize for maximum Sharpe ratio with constraints
    ef = EfficientFrontier(mu, S)
    
    # Add weight constraints
    ef.add_constraint(lambda w: w >= min_weight)
    ef.add_constraint(lambda w: w <= max_weight)
    
    # Add sector constraints if provided
    if sector_constraints:
        for sector, (tickers, min_alloc, max_alloc) in sector_constraints.items():
            # Create a binary vector with 1s for assets in this sector
            sector_indices = [ticker in tickers for ticker in mu.index]
            
            if any(sector_indices):  # Only add constraint if we have assets in this sector
                # Add minimum allocation constraint
                if min_alloc > 0:
                    ef.add_constraint(lambda w, sector_indices=sector_indices: 
                                     sum(w[i] for i, flag in enumerate(sector_indices) if flag) >= min_alloc)
                
                # Add maximum allocation constraint
                if max_alloc < 1:
                    ef.add_constraint(lambda w, sector_indices=sector_indices: 
                                     sum(w[i] for i, flag in enumerate(sector_indices) if flag) <= max_alloc)
    
    # Optimize
    ef.max_sharpe()
    weights = ef.clean_weights()
    
    # Get performance metrics
    performance = ef.portfolio_performance()
    
    return weights, performance, ef, mu, S