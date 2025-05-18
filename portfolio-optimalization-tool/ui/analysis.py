import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetcher import fetch_data, get_dividend_data, get_news_sentiment
from data.processor import backtest_portfolio
from models.markowitz import compute_optimal_portfolio
from models.black_litterman import compute_black_litterman_portfolio
from models.hrp import compute_hrp_portfolio
from models.risk_parity import compute_risk_parity_portfolio, compute_constrained_portfolio
from visualization.standard_plots import (
    plot_normalized_prices, plot_correlation_matrix, plot_portfolio_weights,
    plot_efficient_frontier, plot_monte_carlo, plot_strategy_comparison, plot_return_distribution
)
from visualization.interactive_plots import (
    create_backtest_chart, create_sector_pie_chart, create_risk_contribution_chart,
    create_dividend_chart, create_sentiment_chart, create_rebalancing_chart
)
from pypfopt import EfficientFrontier, risk_models, expected_returns
import yfinance as yf

def select_optimization_method():
    """UI for selecting optimization method and parameters"""
    st.write("### Portfolio Optimization Settings")
    optimization_tabs = st.tabs([
        "Modern Portfolio Theory", 
        "Black-Litterman Model", 
        "Hierarchical Risk Parity",
        "Risk Parity",
        "Constrained Optimization"
    ])
    
    # Default values
    optimization_method = "MPT"
    params = {}
    
    with optimization_tabs[0]:
        optimization_method = "MPT"
        st.write("""
        **Modern Portfolio Theory** optimizes the portfolio based on historical returns and covariance.
        It finds the optimal balance between risk and return based on the efficient frontier.
        """)
        
    with optimization_tabs[1]:
        optimization_method = "BL"
        st.write("""
        **Black-Litterman Model** combines market equilibrium with analyst views to create
        a more robust portfolio that addresses estimation error issues in MPT.
        """)
        
        # Allow user to input views on assets
        st.write("#### Market Views (Optional)")
        st.write("Add your views on expected returns for specific assets:")
        
        views = {}
        view_confidence = 0.5  # Default confidence
        
        # Add views on up to 5 assets
        for i in range(3):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            asset = col1.selectbox(f"Asset {i+1}", options=[""] + st.session_state.selected_tickers, key=f"bl_asset_{i}")
            if asset:
                expected_return = col2.slider(f"Expected Annual Return (%)", -100.0, 100.0, 10.0, key=f"bl_return_{i}") / 100
                confidence = col3.slider(f"Confidence", 0.1, 1.0, 0.5, key=f"bl_conf_{i}")
                
                if asset:
                    views[asset] = (expected_return, confidence)
        
        params["views"] = views
        
    with optimization_tabs[2]:
        optimization_method = "HRP"
        st.write("""
        **Hierarchical Risk Parity** uses machine learning to build a portfolio based on the hierarchical correlation structure
        of assets. It's designed to be more robust to estimation errors than traditional optimization approaches.
        """)
        
    with optimization_tabs[3]:
        optimization_method = "RP"
        st.write("""
        **Risk Parity** allocates assets to equalize risk contribution from each component,
        rather than focusing on return maximization. This approach can provide better diversification.
        """)
        
        params["risk_target"] = st.slider("Target Portfolio Volatility (%)", 5.0, 30.0, 10.0) / 100
        
    with optimization_tabs[4]:
        optimization_method = "Constrained"
        st.write("""
        **Constrained Optimization** applies the Modern Portfolio Theory with additional constraints
        on individual asset weights and sector allocations.
        """)
        
        col1, col2 = st.columns(2)
        params["min_weight"] = col1.slider("Minimum weight per asset (%)", 0, 20, 1) / 100
        params["max_weight"] = col2.slider("Maximum weight per asset (%)", 5, 100, 25) / 100
        
        # Sector constraints
        st.write("#### Sector Constraints")
        st.info("Define minimum and maximum allocations for each market sector")
        
        # Get sectors for selected tickers
        if st.session_state.selected_tickers:
            sectors = {}
            for ticker in st.session_state.selected_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    sector = stock.info.get('sector', 'Unknown')
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(ticker)
                except:
                    if 'Unknown' not in sectors:
                        sectors['Unknown'] = []
                    sectors['Unknown'].append(ticker)
            
            # Create UI elements for each sector
            sector_constraints = {}
            for sector, tickers in sectors.items():
                st.write(f"**{sector}** ({', '.join(tickers)})")
                col1, col2 = st.columns(2)
                min_alloc = col1.slider(f"Min % for {sector}", 0, 50, 0, key=f"min_{sector}") / 100
                max_alloc = col2.slider(f"Max % for {sector}", 0, 100, 100, key=f"max_{sector}") / 100
                sector_constraints[sector] = (tickers, min_alloc, max_alloc)
            
            params["sector_constraints"] = sector_constraints
    
    return optimization_method, params

def run_portfolio_analysis(selected_tickers, period, risk_profile, t, optimize_fn):
    """
    Run portfolio analysis with the selected tickers and parameters
    
    Parameters:
    - selected_tickers: List of ticker symbols
    - period: Time period for analysis
    - risk_profile: User's risk profile
    - t: Translation function
    - optimize_fn: Portfolio optimization function to use
    """
    if not selected_tickers:
        st.error(t('select_at_least_one'))
        return
    
    try:
        with st.spinner('Fetching data...'):
            prices = fetch_data(selected_tickers, period)
        
        if prices.empty or len(prices) == 0:
            st.error("Failed to fetch price data or data is empty. Please check ticker symbols.")
            return
        
        # Display price data
        st.write(f"### {t('price_data')}")
        st.dataframe(prices, use_container_width=True)
        
        # Display price chart - pełna szerokość
        st.write(f"### {t('price_chart')}")
        fig_prices = plot_normalized_prices(prices)
        st.pyplot(fig_prices, use_container_width=True)
        
        # Calculate returns and display correlation matrix - pełna szerokość
        returns = prices.pct_change().dropna()
        st.write(f"### {t('correlation_matrix')}")
        fig_corr = plot_correlation_matrix(returns)
        st.pyplot(fig_corr, use_container_width=True)
        
        # Optimize portfolio using the provided optimization function
        with st.spinner('Optimizing portfolio...'):
            try:
                weights, performance, ef = optimize_fn(prices)
                
                # Convert weights dictionary to DataFrame for plotting
                weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
                weights_df['Weight'] = weights_df['Weight'] * 100  # Convert to percentage
                
                # Układ w dwóch kolumnach dla wag i metryk wydajności
                col1, col2 = st.columns(2)
                
                # Display optimal portfolio weights
                with col1:
                    st.write(f"### {t('optimal_portfolio')}")
                    fig_weights = plot_portfolio_weights(weights_df)
                    st.pyplot(fig_weights, use_container_width=True)
                
                # Display performance metrics
                with col2:
                    st.write(f"### {t('performance_metrics')}")
                    st.write(f"Expected annual return: {performance[0]:.2%}")
                    st.write(f"Annual volatility: {performance[1]:.2%}")
                    st.write(f"Sharpe Ratio: {performance[2]:.2f}")
                
                # Plot efficient frontier if available - pełna szerokość
                if ef is not None:
                    st.write(f"### {t('efficient_frontier')}")
                    fig_ef = plot_efficient_frontier(ef, prices)
                    st.pyplot(fig_ef, use_container_width=True)
                
                # Run Monte Carlo simulation - pełna szerokość
                st.write(f"### {t('monte_carlo')}")
                fig_mc = plot_monte_carlo(prices, weights)
                st.pyplot(fig_mc, use_container_width=True)
                
                # Compare strategies - pełna szerokość
                st.write(f"### {t('strategies_comparison')}")
                mu = expected_returns.mean_historical_return(prices)
                S = risk_models.sample_cov(prices)
                
                # Get performance metrics for different strategies
                ef_sharpe = EfficientFrontier(mu, S)
                ef_sharpe.max_sharpe()
                sharpe_performance = ef_sharpe.portfolio_performance()
                
                ef_min_vol = EfficientFrontier(mu, S)
                ef_min_vol.min_volatility()
                min_vol_perf = ef_min_vol.portfolio_performance()
                
                # Maximum return portfolio
                max_ret_asset = mu.idxmax()
                # Create a new EfficientFrontier instance for max return
                ef_max_ret = EfficientFrontier(mu, S)
                # Create a dictionary with the max return asset weighted at 100%
                max_ret_weights = {asset: 1.0 if asset == max_ret_asset else 0.0 for asset in mu.index}
                # Properly set the weights using the API
                ef_max_ret.set_weights(max_ret_weights)
                max_ret_perf = ef_max_ret.portfolio_performance()
                
                fig_strat, _ = plot_strategy_comparison(sharpe_performance, min_vol_perf, max_ret_perf)
                st.pyplot(fig_strat, use_container_width=True)
                
                # Show return distribution - pełna szerokość
                st.write(f"### {t('return_distribution')}")
                fig_dist = plot_return_distribution(returns, weights)
                st.pyplot(fig_dist, use_container_width=True)
                
                # Run backtesting - pełna szerokość
                st.write(f"### {t('backtest_results')}")
                backtest_results = backtest_portfolio(prices, weights, benchmark_ticker="^GSPC")
                fig_backtest = create_backtest_chart(backtest_results, benchmark_name="S&P 500")
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Get and display dividend data - pełna szerokość
                st.write(f"### {t('dividend_analysis')}")
                dividend_data = get_dividend_data(selected_tickers, period)
                if any(data.get('has_dividends', False) for data in dividend_data.values()):
                    fig_div = create_dividend_chart(dividend_data, weights)
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    # Calculate portfolio yield
                    total_value = sum(weights.get(ticker, 0) * data.get('latest_price', 0) 
                                    for ticker, data in dividend_data.items())
                    total_dividend = sum(weights.get(ticker, 0) * data.get('total_dividend', 0) 
                                       for ticker, data in dividend_data.items())
                    if total_value > 0:
                        portfolio_yield = (total_dividend / total_value) * 100
                        st.write(f"### {t('portfolio_yield')}: {portfolio_yield:.2f}%")
                
            except Exception as e:
                st.error(f"Portfolio optimization failed: {str(e)}")
                st.write("Trying fallback to equal weights...")
                
                # Use equal weights as fallback
                weights = {ticker: 1.0/len(selected_tickers) for ticker in selected_tickers}
                weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
                weights_df['Weight'] = weights_df['Weight'] * 100  # Convert to percentage
                
                # Display equal weights
                st.write("### Equal Weight Portfolio")
                fig_weights = plot_portfolio_weights(weights_df)
                st.pyplot(fig_weights, use_container_width=True)
    
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.write(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        st.write("Traceback:", traceback.format_exc())
