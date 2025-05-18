import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optimizer import fetch_data, compute_optimal_portfolio
from pypfopt import plotting, EfficientFrontier, risk_models, expected_returns

st.title("Modern Portfolio Theory Optimizer (Markowitz Model)")
st.info("""
This tool implements Harry Markowitz's Modern Portfolio Theory (1952) to find optimal asset allocations.
The model optimizes the tradeoff between risk and return using mean-variance optimization.
""")
# User input for stock tickers
tickers_input = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOG):")
period = st.selectbox("Select the data period:", ["1y", "2y", "3y", "4y", "5y", "10y"])

if st.button("Fetch Data"):
    if tickers_input:
        tickers = [ticker.strip() for ticker in tickers_input.split(",")]
        prices = fetch_data(tickers, period)

        # Display fetched data
        st.write("### Fetched Stock Price Data")
        st.dataframe(prices)

        # Display price chart - FIX: Upewnij się, że struktura DataFrame jest odpowiednia dla Streamlit
        st.write("### Price Chart")
        
        # Użyj matplotlib zamiast st.line_chart, aby uniknąć problemów z formatem danych
        fig, ax = plt.subplots(figsize=(10, 6))
        normalized = prices.div(prices.iloc[0]) * 100
        
        for column in normalized.columns:
            ax.plot(normalized.index, normalized[column], label=column)
            
        ax.set_title('Normalized Price (Base=100)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Alternatywnie, jeśli chcesz użyć st.line_chart, upewnij się, że dane mają prosty format
        # normalized_simple = normalized.reset_index()
        # st.line_chart(normalized_simple.set_index('Date'))
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Display correlation heatmap
        st.write("### Correlation Matrix")
        corr = returns.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
        # Compute optimal portfolio
        weights, performance, ef, mu, S = compute_optimal_portfolio(prices)

        # Display portfolio weights
        st.write("### Optimal Portfolio Weights (Maximum Sharpe Ratio)")
        weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])

        # Konwertuj klucze na stringi, jeśli są krotkami
        weights_df['Asset'] = weights_df['Asset'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        weights_df['Weight'] = weights_df['Weight'] * 100  # Convert to percentage

        # Sortuj według wagi, aby lepiej wizualizować istotne składniki
        weights_df = weights_df.sort_values('Weight', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(weights_df['Asset'], weights_df['Weight'])
        ax.set_ylabel('Weight (%)')
        ax.set_title('Portfolio Allocation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display tabular weights
        st.dataframe(weights_df)

        # Performance metrics
        st.write("### Performance Metrics")
        perf_df = pd.DataFrame({
            'Metric': ['Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
            'Value': [f"{performance[0]*100:.2f}%", f"{performance[1]*100:.2f}%", f"{performance[2]:.2f}"]
        })
        st.dataframe(perf_df)
        
        # Create minimum volatility portfolio
        min_vol_ef = EfficientFrontier(mu, S)
        min_vol_ef.min_volatility()
        min_vol_weights = min_vol_ef.clean_weights()
        min_vol_perf = min_vol_ef.portfolio_performance()
        
        st.write("### Minimum Volatility Portfolio")
        min_vol_df = pd.DataFrame(list(min_vol_weights.items()), columns=['Asset', 'Weight'])
        min_vol_df['Asset'] = min_vol_df['Asset'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        min_vol_df['Weight'] = min_vol_df['Weight'] * 100  # Convert to percentage
        st.dataframe(min_vol_df)
        
        st.write(f"Expected Annual Return: {min_vol_perf[0]*100:.2f}%")
        st.write(f"Annual Volatility: {min_vol_perf[1]*100:.2f}%")
        st.write(f"Sharpe Ratio: {min_vol_perf[2]:.2f}")
        
        # Create maximum return portfolio manually
        st.write("### Maximum Return Portfolio")

        # Find asset with highest return
        max_return_asset = mu.idxmax()
        max_ret_value = mu.max()

        # Create portfolio with 100% allocation to highest return asset
        max_ret_weights = {asset: 0 for asset in mu.index}
        max_ret_weights[max_return_asset] = 1.0

        # Calculate volatility for this portfolio
        max_ret_vol = np.sqrt(S.loc[max_return_asset, max_return_asset])
        max_ret_sharpe = max_ret_value / max_ret_vol

        # Display portfolio weights
        max_ret_df = pd.DataFrame(list(max_ret_weights.items()), columns=['Asset', 'Weight'])
        max_ret_df['Asset'] = max_ret_df['Asset'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
        max_ret_df['Weight'] = max_ret_df['Weight'] * 100  # Convert to percentage
        max_ret_df = max_ret_df.sort_values('Weight', ascending=False)
        st.dataframe(max_ret_df)

        max_ret_perf = (max_ret_value, max_ret_vol, max_ret_sharpe)
        st.write(f"Expected Annual Return: {max_ret_perf[0]*100:.2f}%")
        st.write(f"Annual Volatility: {max_ret_perf[1]*100:.2f}%")
        st.write(f"Sharpe Ratio: {max_ret_perf[2]:.2f}")

        # Efficient Frontier Plot - ulepszona wersja
        st.write("### Efficient Frontier")
        st.write("""
Granica efektywna pokazuje optymalne portfolia w przestrzeni ryzyko-zwrot.
- **Każdy punkt na granicy** reprezentuje portfel o maksymalnym możliwym zwrocie przy danym poziomie ryzyka
- **Gwiazdy** pokazują trzy optymalne strategie
""")

        # Większy wykres dla lepszej czytelności
        fig, ax = plt.subplots(figsize=(12, 8))

        # Stwórz czytelniejszą granicę efektywną
        ef_plot = EfficientFrontier(mu, S)
        risk_range = np.linspace(0.001, max_ret_vol*1.1, 100)
        ret_range = []

        for vol in risk_range:
            try:
                ef_plot = EfficientFrontier(mu, S)
                ef_plot.efficient_risk(vol)
                ret = ef_plot.portfolio_performance()[0]
                ret_range.append(ret)
            except:
                ret_range.append(np.nan)

        # Rysuj granicę efektywną jako linię, nie jako rozproszone punkty
        valid_indices = ~np.isnan(ret_range)
        ax.plot(risk_range[valid_indices], np.array(ret_range)[valid_indices], 
                'b-', linewidth=3, alpha=0.7, label="Granica efektywna")

        # Zaznacz pojedyncze składniki na wykresie z etykietami
        for i, asset in enumerate(mu.index):
            asset_name = asset[0] if isinstance(asset, tuple) else asset
            asset_risk = np.sqrt(S.loc[asset, asset])
            asset_ret = mu[asset]
            ax.scatter(asset_risk, asset_ret, marker='o', s=100, color=f'C{i}', alpha=0.7)
            ax.annotate(asset_name, 
                        (asset_risk, asset_ret),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10)

        # Oznacz optymalne portfele wyraźniej z dokładnymi wartościami
        ax.scatter(min_vol_perf[1], min_vol_perf[0], marker="*", s=300, c="r", 
                   label=f"Min. Volatility: {min_vol_perf[1]*100:.1f}% risk, {min_vol_perf[0]*100:.1f}% return")
        ax.scatter(performance[1], performance[0], marker="*", s=300, c="g", 
                   label=f"Max Sharpe: {performance[1]*100:.1f}% risk, {performance[0]*100:.1f}% return")
        ax.scatter(max_ret_perf[1], max_ret_perf[0], marker="*", s=300, c="b", 
                   label=f"Max Return: {max_ret_perf[1]*100:.1f}% risk, {max_ret_perf[0]*100:.1f}% return")

        # Ulepszenia formatowania
        ax.set_title('Efficient Frontier', fontsize=16)
        ax.set_xlabel('Volatility (Standard Deviation) %', fontsize=14)
        ax.set_ylabel('Expected Annual Return %', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Konwertuj osie na procenty
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

        # Dodaj legendę w lepszym miejscu
        ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)

        # Dodaj tekstowe wyjaśnienie
        st.write("""
**Jak interpretować ten wykres:**
- Każdy punkt na niebieskiej linii (granicy efektywnej) reprezentuje optymalny portfel
- Punkty oznaczone gwiazdkami to konkretne strategie:
  - **Czerwona gwiazda** (min. volatility): Portfel o najniższym możliwym ryzyku
  - **Zielona gwiazda** (max. Sharpe): Optymalny kompromis między ryzykiem a zwrotem
  - **Niebieska gwiazda** (max. return): Portfel o najwyższym możliwym zwrocie (całość w jednym aktywie)
- Pojedyncze punkty to indywidualne aktywa w portfoliu
""")
        
        # Monte Carlo simulation
        st.write("### Monte Carlo Simulation")
        num_ports = 1000
        all_weights = np.zeros((num_ports, len(mu)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)

        for i in range(num_ports):
            # Random weights
            weights = np.random.random(len(mu))
            weights = weights / np.sum(weights)
            all_weights[i, :] = weights
            
            # Expected return and volatility
            ret_arr[i] = np.sum(mu * weights)
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
            
            # Sharpe ratio
            if vol_arr[i] == 0:
                sharpe_arr[i] = 0  # Assign 0 if volatility is zero to avoid division by zero
            else:
                sharpe_arr[i] = ret_arr[i] / vol_arr[i]
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.5)
        
        # Mark the optimized portfolios
        ax.scatter(min_vol_perf[1], min_vol_perf[0], marker="*", s=200, c="r", label="Minimum Volatility")
        ax.scatter(performance[1], performance[0], marker="*", s=200, c="g", label="Maximum Sharpe")
        ax.scatter(max_ret_perf[1], max_ret_perf[0], marker="*", s=200, c="b", label="Maximum Return")
        
        plt.colorbar(scatter, label='Sharpe Ratio')
        ax.set_title('Monte Carlo Simulation of Portfolio Optimization')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Return')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Porównanie strategii - lepszy format
        st.write("### Portfolio Strategies Comparison")
        strategies = pd.DataFrame({
            'Strategy': ['Maximum Sharpe', 'Minimum Volatility', 'Maximum Return'],
            'Expected Return (%)': [performance[0]*100, min_vol_perf[0]*100, max_ret_perf[0]*100],
            'Volatility (%)': [performance[1]*100, min_vol_perf[1]*100, max_ret_perf[1]*100],
            'Sharpe Ratio': [performance[2], min_vol_perf[2], max_ret_perf[2]]
        })

        # Wyświetl dane w tabeli dla przejrzystości
        st.dataframe(strategies.set_index('Strategy').style.format({
            'Expected Return (%)': '{:.2f}%',
            'Volatility (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}'
        }))

        # Stwórz dwa osobne wykresy - jeden dla zwrotu i ryzyka, drugi dla Sharpe
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Wykres zwrotu i ryzyka
        strategies.plot(x='Strategy', y=['Expected Return (%)', 'Volatility (%)'], kind='bar', ax=ax1)
        ax1.set_title('Expected Return vs. Volatility')
        ax1.set_ylabel('Percentage (%)')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Wykres współczynnika Sharpe
        sns.barplot(x='Strategy', y='Sharpe Ratio', data=strategies, ax=ax2, palette='viridis')
        ax2.set_title('Sharpe Ratio by Strategy')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Dodaj wartości na słupkach Sharpe Ratio
        for i, v in enumerate(strategies['Sharpe Ratio']):
            ax2.text(i, v + 0.05, f'{v:.2f}', ha='center')

        plt.tight_layout()
        st.pyplot(fig)

        # Dodaj wykres radarowy dla pełniejszego porównania (opcjonalnie)
        st.write("### Radar Chart - Strategy Comparison")

        # Przygotuj dane dla wykresu radarowego
        categories = ['Return', 'Low Risk', 'Sharpe']
        N = len(categories)

        # Normalizuj wartości do skali 0-1 dla wykresu radarowego
        max_return = strategies['Expected Return (%)'].max()
        max_sharpe = strategies['Sharpe Ratio'].max()
        min_vol = strategies['Volatility (%)'].min()  # Niższa zmienność = lepiej

        values = np.zeros((len(strategies), N))
        for i, (_, row) in enumerate(strategies.iterrows()):
            values[i, 0] = row['Expected Return (%)'] / max_return  # Return
            values[i, 1] = min_vol / row['Volatility (%)']  # Odwrócone - niższe ryzyko jest lepsze
            values[i, 2] = row['Sharpe Ratio'] / max_sharpe  # Sharpe

        # Stwórz wykres radarowy
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Kąty dla każdej osi (w radianach)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Zamknij wykres

        # Dodaj osie
        plt.xticks(angles[:-1], categories)

        # Rysuj wykres dla każdej strategii
        for i, strategy in enumerate(strategies['Strategy']):
            values_strategy = values[i].tolist()
            values_strategy += values_strategy[:1]  # Zamknij wielokąt
            ax.plot(angles, values_strategy, linewidth=2, linestyle='solid', label=strategy)
            ax.fill(angles, values_strategy, alpha=0.1)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Strategy Comparison (Normalized)', size=15)
        st.pyplot(fig)

        # Dodanie wykresu historycznych zwrotów
        st.write("### Historical Returns")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Oblicz miesięczne zwroty
        monthly_returns = prices.resample('M').last().pct_change().dropna()
        
        for column in monthly_returns.columns:
            ax.plot(monthly_returns.index, monthly_returns[column], label=column)
            
        ax.set_title('Monthly Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Dodaj histogram rozkładu zwrotów
        st.write("### Return Distribution")
        returns_melted = returns.reset_index().melt(id_vars='Date', var_name='Stock', value_name='Return')

        # Tworzenie wykresu przy użyciu formatu długiego (poprawny sposób)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(data=returns_melted, x='Return', hue='Stock', kde=True, alpha=0.4, ax=ax)
        
        ax.set_title('Return Distribution')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)

        # Alternatywnie: stwórz osobne podwykresy dla każdej akcji
        st.write("### Individual Return Distributions")
        num_stocks = len(returns.columns)
        fig, axes = plt.subplots(nrows=min(3, num_stocks), 
                                 ncols=max(1, (num_stocks + 2) // 3), 
                                 figsize=(14, 10))
        axes = axes.flatten() if num_stocks > 1 else [axes]

        for i, column in enumerate(returns.columns):
            if i < len(axes):
                sns.histplot(returns[column], kde=True, ax=axes[i], color=f'C{i}')
                axes[i].set_title(f'{column} Returns')
                axes[i].set_xlabel('Daily Return')

        # Ukryj puste wykresy, jeśli takie istnieją
        for i in range(num_stocks, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.error("Please enter at least one stock ticker.")