import os
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import streamlit as st
import matplotlib.pyplot as plt

CACHE_DIR = "data"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_data(tickers, period="2y"):
    data = {}
    st.write("Fetching", tickers[0], f"({period}) from Yahoo Finance")
    
    for i, t in enumerate(tickers):
        ticker_data = yf.Ticker(t)
        hist = ticker_data.history(period=period)
        
        if isinstance(hist['Close'], pd.DataFrame):
            data[t] = hist['Close']
        else:
            data[t] = hist['Close'].to_frame(name=t)
        
        progress = int((i+1) / len(tickers) * 100)
        st.progress(progress)
    
    return pd.concat(data, axis=1)

def compute_optimal_portfolio(prices):
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    
    # Create efficient frontier object and optimize for max Sharpe ratio
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    weights = ef.clean_weights()
    exp_ret, vol, sharpe = ef.portfolio_performance(verbose=False)
    
    # Return weights, performance metrics, and the raw inputs (mu, S)
    # so we can create a fresh EfficientFrontier object for plotting
    return weights, (exp_ret, vol, sharpe), ef, mu, S

def plot_portfolio_performance(ef):
    fig, ax = plt.subplots()
    ef.plot_efficient_frontier(ax=ax)
    plt.title("Efficient Frontier")
    st.pyplot(fig)

def main():
    st.title("Portfolio Optimization Tool")
    
    tickers_input = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOG):")
    if tickers_input:
        tickers = [ticker.strip() for ticker in tickers_input.split(",")]
        period = st.selectbox("Select the data period:", ["1y", "2y", "5y"])
        
        prices = fetch_data(tickers, period)
        weights, performance, ef, mu, S = compute_optimal_portfolio(prices)
        
        st.write("Optimal Weights:", weights)
        st.write("Expected Return, Volatility, Sharpe Ratio:", performance)
        
        plot_portfolio_performance(ef)

if __name__ == "__main__":
    main()
