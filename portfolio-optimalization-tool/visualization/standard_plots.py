import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

def plot_normalized_prices(prices):
    """Plot normalized price chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Check if DataFrame is empty
    if prices.empty or len(prices) == 0:
        ax.text(0.5, 0.5, 'No price data available', 
                horizontalalignment='center', 
                verticalalignment='center', fontsize=14)
        ax.set_title('Price Chart - No Data')
        plt.tight_layout()
        return fig
    
    try:
        # Normalize only if we have data
        normalized = prices.div(prices.iloc[0]) * 100
        
        # Use a colorful palette for better distinction
        palette = sns.color_palette("husl", len(normalized.columns))
        
        for i, column in enumerate(normalized.columns):
            ax.plot(normalized.index, normalized[column], label=column, linewidth=2, color=palette[i])
            
        # Improve chart formatting
        ax.set_title('Normalized Price (Base=100)', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        
        # Format legend for better readability
        if len(normalized.columns) > 10:
            # For many tickers, use a smaller font and place outside
            ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5), 
                      ncol=max(1, len(normalized.columns) // 20))
        else:
            ax.legend(fontsize=10)
            
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        import traceback
        st.error(f"Error in plot_normalized_prices: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        ax.text(0.5, 0.5, f'Error generating chart: {str(e)}', 
                horizontalalignment='center', 
                verticalalignment='center', fontsize=12, color='red')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(returns, figsize=(14, 10)):
    """Plot correlation heatmap"""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    return fig

def plot_portfolio_weights(weights_df, figsize=(14, 8)):
    """Plot portfolio weights bar chart"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(weights_df['Asset'], weights_df['Weight'])
    ax.set_ylabel('Weight (%)')
    ax.set_title('Portfolio Allocation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_efficient_frontier(ef, prices, figsize=(16, 10)):
    """Plot efficient frontier with optimal portfolios"""
    try:
        # Extract expected returns and covariance matrix
        mu = ef.expected_returns
        S = ef.cov_matrix
        
        # Check if we have valid data
        if mu is None or S is None or len(mu) == 0:
            raise ValueError("Missing expected returns or covariance matrix")
        
        # Convert mu to pandas Series if it's a numpy array
        if isinstance(mu, np.ndarray):
            mu = pd.Series(mu, index=prices.columns)
            
        # Convert S to pandas DataFrame if it's a numpy array    
        if isinstance(S, np.ndarray):
            S = pd.DataFrame(S, index=prices.columns, columns=prices.columns)
            
        # Get performance metrics of different portfolio strategies
        # Note: This may modify ef's weights, so we create a new instance for each calculation
        ef_sharpe = EfficientFrontier(mu, S)
        ef_sharpe.max_sharpe()
        performance = ef_sharpe.portfolio_performance()
        
        ef_min_vol = EfficientFrontier(mu, S)
        ef_min_vol.min_volatility()
        min_vol_perf = ef_min_vol.portfolio_performance()
        
        # Maximum return portfolio invests everything in the asset with highest expected return
        max_ret_asset = mu.idxmax()
        # Create a new EfficientFrontier instance for max return
        ef_max_ret = EfficientFrontier(mu, S)
        # Create a dictionary with the max return asset weighted at 100%
        max_ret_weights = {asset: 1.0 if asset == max_ret_asset else 0.0 for asset in mu.index}
        # Properly set the weights using the API
        ef_max_ret.set_weights(max_ret_weights)
        max_ret_perf = ef_max_ret.portfolio_performance()
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute efficient frontier curve
        risk_range = np.linspace(min_vol_perf[1] * 0.7, max_ret_perf[1] * 1.2, 100)
        ret_range = []
        
        for vol in risk_range:
            try:
                ef_temp = EfficientFrontier(mu, S)
                ef_temp.efficient_risk(vol)
                ret = ef_temp.portfolio_performance()[0]
                ret_range.append(ret)
            except:
                ret_range.append(np.nan)
        
        # Plot efficient frontier line
        valid_indices = ~np.isnan(np.array(ret_range))
        ax.plot(np.array(risk_range)[valid_indices] * 100, np.array(ret_range)[valid_indices] * 100, 
                'b-', linewidth=3, alpha=0.7, label="Efficient Frontier")
        
        # Mark individual assets
        for i, asset in enumerate(mu.index):
            asset_name = asset
            asset_risk = np.sqrt(S.loc[asset, asset]) * 100  # Convert to percentage
            asset_ret = mu[asset] * 100  # Convert to percentage
            ax.scatter(asset_risk, asset_ret, marker='o', s=100, alpha=0.7)
            ax.annotate(asset_name, 
                        (asset_risk, asset_ret),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10)
        
        # Mark optimal portfolios
        ax.scatter(min_vol_perf[1] * 100, min_vol_perf[0] * 100, marker="*", s=300, c="r", 
                  label=f"Min. Volatility: {min_vol_perf[1]*100:.1f}% risk, {min_vol_perf[0]*100:.1f}% return")
        ax.scatter(performance[1] * 100, performance[0] * 100, marker="*", s=300, c="g", 
                  label=f"Max Sharpe: {performance[1]*100:.1f}% risk, {performance[0]*100:.1f}% return")
        ax.scatter(max_ret_perf[1] * 100, max_ret_perf[0] * 100, marker="*", s=300, c="b", 
                  label=f"Max Return: {max_ret_perf[1]*100:.1f}% risk, {max_ret_perf[0]*100:.1f}% return")
        
        # Format plot
        ax.set_title('Efficient Frontier', fontsize=16)
        ax.set_xlabel('Volatility (Standard Deviation) %', fontsize=14)
        ax.set_ylabel('Expected Annual Return %', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=11)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        import traceback
        st.error(f"Error in efficient frontier plot: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error generating efficient frontier: {str(e)}', 
                horizontalalignment='center', 
                verticalalignment='center', fontsize=12, color='red')
        plt.tight_layout()
        return fig

def plot_monte_carlo(prices, weights, num_ports=1000, figsize=(16, 10)):
    """Plot Monte Carlo simulation of random portfolios"""
    try:
        # Check if we have valid inputs
        if prices.empty:
            raise ValueError("Missing price data")
            
        # Handle None weights
        if weights is None:
            st.warning("No weights provided for Monte Carlo simulation, using equal weights.")
            weights = {col: 1.0/len(prices.columns) for col in prices.columns}
            
        # Calculate expected returns and covariance matrix
        returns = prices.pct_change().dropna()
        mu = returns.mean() * 252  # Annualized returns
        S = returns.cov() * 252  # Annualized covariance
        
        # Convert mu to pandas Series if it's a numpy array
        if isinstance(mu, np.ndarray):
            mu = pd.Series(mu, index=prices.columns)
            
        # Convert S to pandas DataFrame if it's a numpy array
        if isinstance(S, np.ndarray):
            S = pd.DataFrame(S, index=prices.columns, columns=prices.columns)
        
        # Get performance of the optimal portfolio
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()
        performance = ef.portfolio_performance()
        
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
        
        # Generate random portfolios
        all_weights = np.zeros((num_ports, len(mu)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)
        
        # Convert S to numpy array for faster matrix calculations if it's a DataFrame
        S_np = S.values if isinstance(S, pd.DataFrame) else S
        
        for i in range(num_ports):
            # Random weights
            weights_rand = np.random.random(len(mu))
            weights_rand = weights_rand / np.sum(weights_rand)
            all_weights[i, :] = weights_rand
            
            # Expected return and volatility
            ret_arr[i] = np.sum(mu.values * weights_rand) if isinstance(mu, pd.Series) else np.sum(mu * weights_rand)
            vol_arr[i] = np.sqrt(np.dot(weights_rand.T, np.dot(S_np, weights_rand)))
            
            # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
            sharpe_arr[i] = ret_arr[i] / vol_arr[i] if vol_arr[i] > 0 else 0
        
        # Convert to percentages for better visualization
        ret_arr *= 100  # Convert to percentage
        vol_arr *= 100  # Convert to percentage
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a scatter plot with a colorbar indicating Sharpe ratio
        scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.5, s=15)
        
        # Mark the optimal portfolios
        ax.scatter(min_vol_perf[1] * 100, min_vol_perf[0] * 100, marker="*", s=200, c="r", 
                   label="Minimum Volatility")
        ax.scatter(performance[1] * 100, performance[0] * 100, marker="*", s=200, c="g", 
                   label="Maximum Sharpe")
        ax.scatter(max_ret_perf[1] * 100, max_ret_perf[0] * 100, marker="*", s=200, c="b", 
                   label="Maximum Return")
        
        # Format plot
        plt.colorbar(scatter, label='Sharpe Ratio')
        ax.set_title('Monte Carlo Simulation of Portfolio Optimization', fontsize=14)
        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        import traceback
        st.error(f"Error in Monte Carlo simulation: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error generating Monte Carlo simulation: {str(e)}', 
                horizontalalignment='center', 
                verticalalignment='center', fontsize=12, color='red')
        plt.tight_layout()
        return fig

def plot_strategy_comparison(performance, min_vol_perf, max_ret_perf, figsize=(16, 8)):
    """Plot comparison of different portfolio strategies"""
    strategies = pd.DataFrame({
        'Strategy': ['Maximum Sharpe', 'Minimum Volatility', 'Maximum Return'],
        'Expected Return (%)': [performance[0]*100, min_vol_perf[0]*100, max_ret_perf[0]*100],
        'Volatility (%)': [performance[1]*100, min_vol_perf[1]*100, max_ret_perf[1]*100],
        'Sharpe Ratio': [performance[2], min_vol_perf[2], max_ret_perf[2]]
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Return and volatility chart
    strategies.plot(x='Strategy', y=['Expected Return (%)', 'Volatility (%)'], kind='bar', ax=ax1)
    ax1.set_title('Expected Return vs. Volatility')
    ax1.set_ylabel('Percentage (%)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Sharpe ratio chart
    sns.barplot(x='Strategy', y='Sharpe Ratio', data=strategies, ax=ax2, palette='viridis')
    ax2.set_title('Sharpe Ratio by Strategy')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on Sharpe Ratio bars
    for i, v in enumerate(strategies['Sharpe Ratio']):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    return fig, strategies

def plot_return_distribution(returns, weights=None, figsize=(16, 8)):
    """Plot distribution of returns"""
    try:
        # Check if we have valid inputs
        if returns.empty:
            raise ValueError("Missing returns data")
            
        # Handle None weights
        if weights is None:
            st.warning("No weights provided for return distribution, using equal weights.")
            weights = {col: 1.0/len(returns.columns) for col in returns.columns}
            
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Individual asset return distributions
        returns_melted = returns.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Return')
        
        # Create a KDE plot for each asset
        sns.kdeplot(data=returns_melted, x='Return', hue='Asset', ax=ax1, fill=True, alpha=0.3)
        
        # Improve formatting
        ax1.set_title('Return Distributions by Asset', fontsize=14)
        ax1.set_xlabel('Daily Return', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Calculate portfolio returns
        if weights:
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            # Plot 2: Portfolio return distribution
            sns.histplot(portfolio_returns, kde=True, ax=ax2, bins=50, color='purple', alpha=0.6)
            
            # Add vertical lines for key statistics
            mean_return = portfolio_returns.mean()
            std_dev = portfolio_returns.std()
            
            ax2.axvline(mean_return, color='g', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_return:.4f}')
            ax2.axvline(mean_return + std_dev, color='r', linestyle='--', linewidth=1, 
                        label=f'+1 Std Dev: {mean_return + std_dev:.4f}')
            ax2.axvline(mean_return - std_dev, color='r', linestyle='--', linewidth=1, 
                        label=f'-1 Std Dev: {mean_return - std_dev:.4f}')
            
            # Improve formatting
            ax2.set_title('Portfolio Return Distribution', fontsize=14)
            ax2.set_xlabel('Daily Return', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_title('Portfolio Return Distribution (Weights Not Provided)', fontsize=14)
            ax2.text(0.5, 0.5, 'Portfolio weights required to calculate distribution', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        import traceback
        st.error(f"Error in return distribution plot: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error generating return distribution plot: {str(e)}', 
                horizontalalignment='center', 
                verticalalignment='center', fontsize=12, color='red')
        plt.tight_layout()
        return fig

def plot_portfolio_comparison(optimal_weights, min_vol_weights, max_ret_weights, optimal_perf, min_vol_perf, max_ret_perf):
    """Plot comprehensive comparison of different portfolio strategies"""
    fig = plt.figure(figsize=(18, 14))
    
    # Utwórz siatkę 2x2 dla różnych wykresów
    gs = fig.add_gridspec(2, 2)
    
    # 1. Wykres porównania wag portfeli (lewy górny)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Przygotuj dane do porównania wag
    all_assets = set()
    for weights in [optimal_weights, min_vol_weights, max_ret_weights]:
        all_assets.update(weights.keys())
    
    assets = sorted(list(all_assets))
    
    # Przygotuj dane do wykresu
    data = []
    for asset in assets:
        data.append({
            'Asset': asset,
            'Optimal Portfolio (%)': optimal_weights.get(asset, 0) * 100,
            'Min Volatility (%)': min_vol_weights.get(asset, 0) * 100,
            'Max Return (%)': max_ret_weights.get(asset, 0) * 100
        })
    
    weights_df = pd.DataFrame(data)
    weights_df = weights_df.set_index('Asset')
    
    # Rysuj wykres porównawczy wag
    weights_df.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Portfolio Weights Comparison', fontsize=14)
    ax1.set_ylabel('Weight (%)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # 2. Wykres wskaźników wydajności (prawy górny)
    ax2 = fig.add_subplot(gs[0, 1])
    
    performance_data = {
        'Strategy': ['Optimal Portfolio', 'Min Volatility', 'Max Return'],
        'Expected Return (%)': [optimal_perf[0]*100, min_vol_perf[0]*100, max_ret_perf[0]*100],
        'Volatility (%)': [optimal_perf[1]*100, min_vol_perf[1]*100, max_ret_perf[1]*100],
        'Sharpe Ratio': [optimal_perf[2], min_vol_perf[2], max_ret_perf[2]]
    }
    
    perf_df = pd.DataFrame(performance_data).set_index('Strategy')
    
    # Rysuj główne metryki dla wszystkich strategii
    perf_df[['Expected Return (%)', 'Volatility (%)']].plot(kind='bar', ax=ax2)
    ax2.set_title('Performance Metrics', fontsize=14)
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Dodaj tekst z wskaźnikami Sharpe bezpośrednio na wykresie
    for i, strategy in enumerate(perf_df.index):
        ax2.annotate(f'Sharpe: {perf_df.loc[strategy, "Sharpe Ratio"]:.2f}',
                     xy=(i, perf_df.loc[strategy, 'Expected Return (%)']),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # 3. Wykres porównawczy ryzyko-zwrot (dolny, na całą szerokość)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Rysuj punkty dla każdej strategii
    ax3.scatter(optimal_perf[1]*100, optimal_perf[0]*100, s=200, color='green', 
                label=f'Optimal Portfolio (Sharpe: {optimal_perf[2]:.2f})')
    ax3.scatter(min_vol_perf[1]*100, min_vol_perf[0]*100, s=200, color='blue', 
                label=f'Min Volatility (Sharpe: {min_vol_perf[2]:.2f})')
    ax3.scatter(max_ret_perf[1]*100, max_ret_perf[0]*100, s=200, color='red', 
                label=f'Max Return (Sharpe: {max_ret_perf[2]:.2f})')
    
    # Połącz punkty linią "frontu efektywnego"
    points = sorted([
        (min_vol_perf[1]*100, min_vol_perf[0]*100),
        (optimal_perf[1]*100, optimal_perf[0]*100),
        (max_ret_perf[1]*100, max_ret_perf[0]*100)
    ], key=lambda x: x[0])
    
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    ax3.plot(x_values, y_values, 'k--', alpha=0.5)
    
    ax3.set_title('Risk-Return Profile Comparison', fontsize=16)
    ax3.set_xlabel('Volatility (%)', fontsize=14)
    ax3.set_ylabel('Expected Return (%)', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left')
    
    plt.tight_layout(h_pad=3, w_pad=3)
    return fig, weights_df, perf_df