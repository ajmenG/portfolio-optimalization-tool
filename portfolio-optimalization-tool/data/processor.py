import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

def backtest_portfolio(prices, weights, benchmark_ticker="^GSPC"):
    """Backtest portfolio performance against a benchmark index"""
    # Ensure weights is in the right format and matches price columns
    if isinstance(weights, dict):
        # Convert dict to Series for easier handling
        weights_series = pd.Series(weights)
    else:
        weights_series = pd.Series(weights)
    
    # Make sure weights only include assets in the prices DataFrame
    weights_series = weights_series[weights_series.index.isin(prices.columns)]
    
    # Normalize weights to sum to 1 if they don't already
    if abs(weights_series.sum() - 1.0) > 0.0001:
        weights_series = weights_series / weights_series.sum()
    
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_hist = (daily_returns * weights_series).sum(axis=1)
    portfolio_cum = (1 + portfolio_hist).cumprod()
    
    # Get benchmark data for the same period with better error handling
    try:
        benchmark_data = yf.download(benchmark_ticker, 
                               start=prices.index[0], 
                               end=prices.index[-1],
                               progress=False)
        
        # Handle different data structures - some versions return MultiIndex columns
        if isinstance(benchmark_data.columns, pd.MultiIndex):
            # For MultiIndex columns
            benchmark = benchmark_data['Close'] if 'Close' in benchmark_data.columns else benchmark_data.iloc[:, 0]
        else:
            # For single-level columns
            if 'Close' in benchmark_data.columns:
                benchmark = benchmark_data['Close']
            elif 'Adj Close' in benchmark_data.columns:
                benchmark = benchmark_data['Adj Close']
            else:
                benchmark = benchmark_data.iloc[:, 0]  # Use first column as fallback
    except Exception as e:
        st.warning(f"Error fetching benchmark data: {e}. Using S&P 500 as fallback.")
        try:
            # Fallback to S&P 500
            benchmark_data = yf.download('^GSPC', 
                                  start=prices.index[0], 
                                  end=prices.index[-1],
                                  progress=False)
            benchmark = benchmark_data['Close']
        except:
            # Last resort - create a dummy benchmark
            benchmark = pd.Series(1, index=prices.index)
            st.error("Could not fetch any benchmark data. Using a placeholder.")
    
    # Ensure benchmark is a pandas Series and not a DataFrame or numpy array
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]
    elif isinstance(benchmark, np.ndarray):
        benchmark = pd.Series(benchmark.flatten(), index=prices.index[:len(benchmark)])
    
    # Make sure benchmark and portfolio have the same dates
    # Align the two series
    portfolio_cum, benchmark = portfolio_cum.align(benchmark, join='inner')
    
    # Check if we have valid benchmark data
    if len(benchmark) == 0:
        st.warning("No overlapping dates between portfolio and benchmark. Using dummy benchmark.")
        benchmark = pd.Series(1, index=portfolio_cum.index)
    
    # Normalize the benchmark to start from 1
    benchmark_cum = benchmark / benchmark.iloc[0]
    
    # Calculate metrics
    portfolio_return = portfolio_cum.iloc[-1] - 1
    benchmark_return = benchmark_cum.iloc[-1] - 1
    
    # Calculate portfolio volatility from historical returns
    portfolio_vol = portfolio_hist.std() * np.sqrt(252)  # Annualized volatility
    
    # Ensure we're working with a Series for benchmark volatility calculation
    if hasattr(benchmark, 'Close'):
        benchmark_returns = benchmark.Close.pct_change().dropna()
    else:
        benchmark_returns = benchmark.pct_change().dropna()
    
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    # Maximum drawdown
    rolling_max = portfolio_cum.cummax()
    drawdowns = (portfolio_cum / rolling_max) - 1
    max_drawdown = drawdowns.min()
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Portfolio': portfolio_cum,
        'Benchmark': benchmark_cum
    })
    
    # Return comparison dataframe and metrics dictionary
    return comparison, {
        'Portfolio Return': portfolio_return,
        'Benchmark Return': benchmark_return,
        'Outperformance': portfolio_return - benchmark_return,
        'Portfolio Volatility': portfolio_vol,
        'Benchmark Volatility': benchmark_vol,
        'Sharpe Ratio': sharpe,
        'Maximum Drawdown': max_drawdown
    }

def is_valid_ticker(ticker):
    """Check if a ticker symbol is valid more robustly"""
    if not ticker or not isinstance(ticker, str):
        return False
        
    # Clean the ticker symbol
    ticker = ticker.strip().upper()
    
    # Some basic format validation
    if not ticker:
        st.warning("Ticker cannot be empty")
        return False
    
    if len(ticker) > 10:
        st.warning(f"Ticker {ticker} is too long (max 10 characters)")
        return False
    
    if not any(c.isalpha() for c in ticker):
        st.warning(f"Ticker {ticker} must contain at least one letter")
        return False
    
    # Try to get data from yfinance
    for attempt in range(3):  # Add retry mechanism
        try:
            if debug:
                st.info(f"Checking if {ticker} exists... (attempt {attempt+1}/3)")
            stock = yf.Ticker(ticker)
            
            # Check multiple indicators to confirm validity
            validations = []
            
            # Method 1: Check info dictionary
            try:
                info = stock.info
                has_market_price = 'regularMarketPrice' in info or 'currentPrice' in info
                validations.append(has_market_price)
                if has_market_price:
                    st.info(f"Found market price for {ticker}")
            except Exception as e:
                st.warning(f"Could not get info for {ticker}: {str(e)}")
                validations.append(False)
            
            # Method 2: Check if we can get any price data
            try:
                hist = stock.history(period="1mo", interval="1d")
                has_history = not hist.empty
                validations.append(has_history)
                if has_history:
                    st.info(f"Found historical data for {ticker} with {len(hist)} records")
            except Exception as e:
                st.warning(f"Could not get history for {ticker}: {str(e)}")
                validations.append(False)
                
            # If any validation method succeeded, consider it valid
            is_valid = any(validations)
            if is_valid:
                st.success(f"Ticker {ticker} is valid!")
            else:
                st.error(f"Ticker {ticker} doesn't appear to be a valid tradable symbol")
            return is_valid
            
        except Exception as e:
            if attempt < 2:
                st.warning(f"Error validating ticker {ticker} (attempt {attempt+1}): {str(e)}. Retrying...")
                time.sleep(1)  # Wait before retry
                continue
            st.error(f"Error validating ticker {ticker}: {str(e)}")
            return False
    
    return False