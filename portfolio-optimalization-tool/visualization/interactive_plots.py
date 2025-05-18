import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_backtest_chart(backtest_data_tuple, benchmark_name="S&P 500"):
    """Create interactive backtest comparison chart"""
    # Extract data from the tuple (comparison_df, metrics_dict)
    comparison_df, metrics_dict = backtest_data_tuple
    
    fig = px.line(
        comparison_df, 
        title=f"Portfolio vs {benchmark_name} Performance",
        labels={"value": "Growth of $1 invested", "variable": "Investment"},
        width=900,
        height=600
    )
    
    # Add hover templates for more detailed tooltips
    fig.update_traces(
        hovertemplate='<b>%{y:.2f}</b><br>Date: %{x}<extra></extra>'
    )
    
    # Add performance metrics as annotations
    annotations = []
    portfolio_return = metrics_dict['Portfolio Return']
    benchmark_return = metrics_dict['Benchmark Return']
    outperformance = metrics_dict['Outperformance']
    
    annotations.append(dict(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"Portfolio Return: {portfolio_return:.2%}",
        showarrow=False, font=dict(size=12, color="green"), bgcolor="white", borderpad=4
    ))
    
    annotations.append(dict(
        x=0.02, y=0.93, xref="paper", yref="paper",
        text=f"Benchmark Return: {benchmark_return:.2%}",
        showarrow=False, font=dict(size=12, color="blue"), bgcolor="white", borderpad=4
    ))
    
    annotations.append(dict(
        x=0.02, y=0.88, xref="paper", yref="paper",
        text=f"Outperformance: {outperformance:.2%}",
        showarrow=False, 
        font=dict(size=12, color="green" if outperformance > 0 else "red"), 
        bgcolor="white", borderpad=4
    ))
    
    # Add annotations to the figure
    fig.update_layout(
        hovermode="x unified",
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update line colors and styles
    fig.update_traces(
        line=dict(width=3),
        selector=dict(name="Portfolio")
    )
    
    fig.update_traces(
        line=dict(width=2, dash='dot'),
        selector=dict(name="Benchmark")
    )
    
    # Add responsive configuration
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add config for better display options
    fig.update_layout(
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def create_sector_pie_chart(sectors_summary):
    """Create interactive pie chart of sector allocations"""
    fig = px.pie(
        sectors_summary, 
        values='Allocation (%)', 
        names='Sector',
        title='Portfolio Allocation by Sector',
        hole=0.4,
        width=900,
        height=600
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # Add responsive configuration
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add config for better display options
    fig.update_layout(
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def create_risk_contribution_chart(risk_contrib_df):
    """Create bar chart of risk contributions by sector"""
    fig = px.bar(
        risk_contrib_df,
        x='Sector',
        y='Risk Contribution (%)',
        title='Portfolio Risk Contribution by Sector',
        color='Risk Contribution (%)',
        width=900,
        height=600
    )
    
    # Add responsive configuration
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add config for better display options
    fig.update_layout(
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def create_dividend_chart(dividend_data, weights=None):
    """Create bar chart of dividend yields
    
    Parameters:
    - dividend_data: Dictionary of ticker-level dividend data
    - weights: Optional dictionary of portfolio weights to filter tickers
    """
    # Process dividend data into DataFrame
    ticker_data = []
    
    # Filter tickers by portfolio weights if provided
    tickers_to_display = list(dividend_data.keys())
    if weights is not None:
        # Only include tickers with positive weights
        tickers_to_display = [ticker for ticker in tickers_to_display 
                             if ticker in weights and weights.get(ticker, 0) > 0]
        
    # Add visual indicator for whether ticker is in portfolio
    for ticker in tickers_to_display:
        data = dividend_data[ticker]
        if data.get('has_dividends', False):
            weight = weights.get(ticker, 0) * 100 if weights else 0  # Convert to percentage
            ticker_data.append({
                'Ticker': ticker,
                'Dividend Yield (%)': data.get('yield', 0),
                'Annual Dividend': data.get('total_dividend', 0),
                'Price': data.get('latest_price', 0),
                'Portfolio Weight (%)': weight
            })
    
    # Create DataFrame
    div_df = pd.DataFrame(ticker_data)
    
    if div_df.empty:
        # No dividend data, return empty figure with message
        fig = go.Figure(layout=dict(width=900, height=600))
        fig.add_annotation(
            text="No dividend data available for selected stocks in portfolio",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Sort by yield
    div_df = div_df.sort_values('Dividend Yield (%)', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        div_df,
        x='Ticker',
        y='Dividend Yield (%)',
        title='Dividend Yield by Stock in Portfolio',
        color='Dividend Yield (%)',
        color_continuous_scale=['#e6f7ff', '#0050b3'],
        hover_data=['Portfolio Weight (%)', 'Annual Dividend', 'Price'],
        width=900,
        height=600
    )
    
    # Add hover data
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Yield: %{y:.2f}%<br>Weight: %{customdata[0]:.2f}%<br>Annual Dividend: $%{customdata[1]:.2f}<br>Price: $%{customdata[2]:.2f}<extra></extra>',
        customdata=div_df[['Portfolio Weight (%)', 'Annual Dividend', 'Price']].values
    )
    
    # Add responsive configuration
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add config for better display options
    fig.update_layout(
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def create_sentiment_chart(sentiment_data):
    """Create bar chart of news sentiment by stock"""
    # Process sentiment data into DataFrame
    ticker_data = []
    for ticker, data in sentiment_data.items():
        if data.get('has_news', False):
            ticker_data.append({
                'Ticker': ticker,
                'Sentiment Score': data.get('avg_sentiment', 0),
                'News Count': len(data.get('news', []))
            })
    
    # Create DataFrame
    sent_df = pd.DataFrame(ticker_data)
    
    if sent_df.empty:
        # No sentiment data, return empty figure with message
        fig = go.Figure(layout=dict(width=900, height=600))
        fig.add_annotation(
            text="No news sentiment data available for selected stocks",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Sort by sentiment score
    sent_df = sent_df.sort_values('Sentiment Score', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        sent_df,
        x='Ticker',
        y='Sentiment Score',
        color='Sentiment Score',
        color_continuous_scale=['red', 'lightgray', 'green'],
        range_color=[-1, 1],
        title='News Sentiment by Stock',
        text='News Count',
        width=900,
        height=600
    )
    
    # Improve hover info
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.2f}<br>News articles: %{text}<extra></extra>'
    )
    
    # Add zero reference line
    fig.add_shape(
        type="line", line=dict(dash="dash", width=2, color="gray"),
        x0=-0.5, x1=len(sent_df)-0.5, y0=0, y1=0
    )
    
    # Add responsive configuration
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add config for better display options
    fig.update_layout(
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def create_rebalancing_chart(rebalance_df):
    """Create interactive chart comparing current vs target allocations"""
    if rebalance_df.empty:
        # No rebalancing data, return empty figure with message
        fig = go.Figure(layout=dict(width=900, height=600))
        fig.add_annotation(
            text="No rebalancing data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Sort by ticker for better visualization
    rebalance_df = rebalance_df.sort_values('Ticker')
    
    # Create figure
    fig = go.Figure()
    
    # Calculate differences for coloring
    rebalance_df['Difference'] = rebalance_df['Current (%)'] - rebalance_df['Target (%)']
    
    # Add current allocation bars
    fig.add_trace(go.Bar(
        name='Current',
        x=rebalance_df['Ticker'],
        y=rebalance_df['Current (%)'],
        marker_color='rgba(58, 71, 80, 0.6)',
        hovertemplate='<b>%{x}</b><br>Current: %{y:.2f}%<extra></extra>'
    ))
    
    # Add target allocation bars
    fig.add_trace(go.Bar(
        name='Target',
        x=rebalance_df['Ticker'],
        y=rebalance_df['Target (%)'],
        marker_color='rgba(246, 78, 139, 0.6)',
        hovertemplate='<b>%{x}</b><br>Target: %{y:.2f}%<extra></extra>'
    ))
    
    # Add difference annotations
    for i, row in rebalance_df.iterrows():
        # Skip small differences
        if abs(row['Difference']) < 0.5:
            continue
            
        # Add annotation to show difference
        fig.add_annotation(
            x=i,
            y=max(row['Current (%)'], row['Target (%)']) + 2,
            text=f"{row['Difference']:.1f}%",
            showarrow=False,
            font=dict(
                size=10,
                color='green' if row['Difference'] < 0 else 'red'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Current vs Target Allocation',
        barmode='group',
        yaxis_title='Allocation (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=900,
        height=600,
        # Add responsive configuration
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        # Add better hover options
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig