# Portfolio Optimization Tool

## Overview
The Portfolio Optimization Tool is a powerful Streamlit application that implements Harry Markowitz's Modern Portfolio Theory (MPT) to help investors optimize their investment portfolios. The tool analyzes historical stock data to find optimal asset allocations that maximize returns while minimizing risk, visualizing the efficient frontier and various portfolio strategies.

![Portfolio Optimization](https://example.com/portfolio-optimization-image.png)

## Features
- **Stock Selection Options**:
  - Manual ticker input
  - Preset index selections (S&P 500, Nasdaq-100, Dow Jones)
  - Sector-based popular stocks
  - Custom ticker validation
- **Data Analysis**:
  - Historical price charts
  - Correlation matrices
  - Return distributions
  - Monte Carlo simulations
- **Portfolio Optimization Strategies**:
  - Maximum Sharpe Ratio (optimal risk/return)
  - Minimum Volatility
  - Maximum Return
- **Visualization**:
  - Efficient Frontier plotting
  - Portfolio allocation charts
  - Performance comparison
  - Return distribution analysis

## Files
- **app.py**: Main Streamlit application with basic UI and visualization
- **helper.py**: Enhanced UI components with additional features like index selection
- **optimizer.py**: Core functions for portfolio optimization and data fetching
- **requirements.txt**: Project dependencies

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd portfolio-optimization-tool
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
Run the application using the following command:
```bash
streamlit run app.py
```

For the enhanced version with more features:
```bash
streamlit run helper.py
```

### Using the Application
1. **Select Stocks**:
   - Enter stock tickers manually (e.g., AAPL, MSFT, GOOG)
   - Or use the ticker search functionality
   - Or select from predefined indices

2. **Set Parameters**:
   - Choose the historical data period (1y, 2y, 5y, etc.)

3. **Analyze Results**:
   - View the efficient frontier visualization
   - Compare different portfolio strategies
   - Analyze correlation between assets
   - Explore optimized portfolio weights

## Core Concepts

### Modern Portfolio Theory
The application implements Markowitz's Modern Portfolio Theory which:
- Quantifies the relationship between risk and return
- Demonstrates how diversification reduces portfolio risk
- Finds optimal portfolios that offer the highest expected return for a given level of risk

### Efficient Frontier
The curved line representing all optimal portfolios that offer:
- The highest expected return for a defined level of risk
- The lowest risk for a given level of expected return

### Sharpe Ratio
A measure of risk-adjusted return, calculated as:
```
Sharpe Ratio = (Expected Return - Risk Free Rate) / Portfolio Volatility
```
Higher Sharpe ratios indicate better risk-adjusted performance.

## Example Results
- **Maximum Sharpe Portfolio**: Optimal risk-adjusted return
- **Minimum Volatility Portfolio**: Lowest possible risk
- **Maximum Return Portfolio**: Highest possible return (typically concentrated in a single asset)

## Dependencies
The project relies on the following Python packages:
- streamlit >= 1.22.0
- pandas >= 1.3.5
- yfinance >= 0.2.18
- pypfopt >= 1.5.5
- matplotlib >= 3.5.3
- numpy >= 1.21.6
- seaborn >= 0.12.2
- scikit-learn >= 1.0.2
- plotly >= 5.13.0

## Limitations
- Historical performance does not guarantee future results
- Market conditions can change drastically
- Model assumes normal distribution of returns
- Transaction costs and taxes are not considered

## Contributing
Contributions are welcome! Please feel free to:
- Submit pull requests
- Report bugs or issues
- Suggest new features or enhancements
- Improve documentation

## License
This project is licensed under the MIT License. See the LICENSE file for more details.