import pandas as pd
import numpy as np
from .markowitz import compute_optimal_portfolio as markowitz_optimize
from .risk_parity import compute_risk_parity_portfolio, compute_constrained_portfolio
from .hrp import compute_hrp_portfolio
from .black_litterman import compute_black_litterman_portfolio

def wrap_markowitz(prices, strategy='sharpe'):
    """Wrapper for Markowitz model"""
    try:
        weights, performance, ef = markowitz_optimize(prices, strategy=strategy)
        return weights, performance, ef
    except Exception as e:
        raise Exception(f"Markowitz optimization failed: {str(e)}")

def wrap_min_volatility(prices):
    """Wrapper for minimum volatility portfolio"""
    try:
        return wrap_markowitz(prices, strategy='min_volatility')
    except Exception as e:
        raise Exception(f"Minimum volatility optimization failed: {str(e)}")

def wrap_max_return(prices):
    """Wrapper for maximum return portfolio"""
    try:
        return wrap_markowitz(prices, strategy='max_return')
    except Exception as e:
        raise Exception(f"Maximum return optimization failed: {str(e)}")

def wrap_risk_parity(prices):
    """Wrapper for risk parity model"""
    try:
        weights, performance, ef, _, _ = compute_risk_parity_portfolio(prices)
        return weights, performance, ef
    except Exception as e:
        raise Exception(f"Risk parity optimization failed: {str(e)}")

def wrap_hrp(prices):
    """Wrapper for hierarchical risk parity model"""
    try:
        weights, performance, hrp, _, _ = compute_hrp_portfolio(prices)
        return weights, performance, None  # HRP doesn't provide efficient frontier
    except Exception as e:
        raise Exception(f"HRP optimization failed: {str(e)}")

def wrap_black_litterman(prices, views=None):
    """Wrapper for Black-Litterman model"""
    try:
        weights, performance, ef, _, _ = compute_black_litterman_portfolio(prices, views)
        return weights, performance, ef
    except Exception as e:
        raise Exception(f"Black-Litterman optimization failed: {str(e)}")

def wrap_constrained(prices, min_weight=0.01, max_weight=0.25, sector_constraints=None):
    """Wrapper for constrained optimization"""
    try:
        weights, performance, ef, _, _ = compute_constrained_portfolio(
            prices, min_weight, max_weight, sector_constraints
        )
        return weights, performance, ef
    except Exception as e:
        raise Exception(f"Constrained optimization failed: {str(e)}")

# Map of model names to their wrapper functions
MODEL_WRAPPERS = {
    'model_markowitz': wrap_markowitz,
    'model_min_vol': wrap_min_volatility,
    'model_max_ret': wrap_max_return,
    'model_risk_parity': wrap_risk_parity,
    'model_hrp': wrap_hrp,
    'model_black_litterman': wrap_black_litterman
} 