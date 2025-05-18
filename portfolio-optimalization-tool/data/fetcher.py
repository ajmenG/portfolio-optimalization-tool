import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup
import re

@st.cache_data(ttl=3600)
def fetch_data(tickers, period="5y"):
    """Fetch historical price data for the given tickers"""
    if not tickers:
        st.error("No tickers provided")
        return pd.DataFrame()
        
    # Convert to list if a single ticker is provided
    if isinstance(tickers, str):
        tickers = [tickers]
        
    for attempt in range(3):  # Add retry mechanism
        try:
            data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            
            if data.empty:
                if attempt < 2:
                    time.sleep(2)  # Wait before retry
                    continue
                st.error("No data available for the selected tickers")
                return pd.DataFrame()
            
            # Handle multi-ticker vs single-ticker data
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    closes = data['Close']
                elif 'Adj Close' in data.columns.levels[0]:
                    closes = data['Adj Close']
                else:
                    avail_cols = data.columns.levels[0].tolist()
                    for col in ['Close', 'Adj Close', 'Price', 'Last']:
                        if col in avail_cols:
                            closes = data[col]
                            break
                    else:
                        st.error("Cannot determine price column")
                        return pd.DataFrame()
            else:
                if 'Close' in data.columns:
                    closes = pd.DataFrame(data['Close'])
                elif 'Adj Close' in data.columns:
                    closes = pd.DataFrame(data['Adj Close'])
                else:
                    st.error("No price columns found")
                    return pd.DataFrame()
                
                # Set column name for single ticker
                if len(tickers) == 1:
                    closes.columns = [tickers[0]]
            
            # Ensure datetime index and handle missing data
            closes.index = pd.to_datetime(closes.index)
            
            # Check for missing data and handle it
            missing_data = closes.isnull().sum()
            if missing_data.any():
                st.warning(f"Missing data detected for some tickers. Using forward/backward fill to handle missing values.")
                for ticker in missing_data[missing_data > 0].index:
                    st.info(f"{ticker} has {missing_data[ticker]} missing data points")
                
                closes = closes.fillna(method='ffill').fillna(method='bfill')
                
            return closes
            
        except Exception as e:
            if attempt < 2:
                time.sleep(2)  # Wait before retry
                continue
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """Get list of S&P 500 tickers"""
    for attempt in range(3):
        try:
            # Try from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to load Wikipedia page: {response.status_code}")
                
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            
            if not table:
                raise Exception("Failed to find the S&P 500 table on Wikipedia")
                
            tickers = []
            for row in table.find_all('tr')[1:]:  # Skip header
                cells = row.find_all('td')
                if len(cells) >= 2:
                    ticker = cells[0].text.strip()
                    # Remove any ".XX" suffixes (like .L for London)
                    ticker = re.sub(r'\.\w+$', '', ticker)
                    tickers.append(ticker)
            
            # Validate we got a reasonable number of tickers
            if len(tickers) < 400:  # S&P 500 should have ~500 tickers
                raise Exception(f"Found only {len(tickers)} tickers, expected around 500")
                
            return tickers
            
        except Exception as e:
            if attempt == 2:  # Last attempt, use fallback
                st.warning(f"Failed to fetch S&P 500 tickers: {str(e)}. Using fallback list.")
                # Fallback tickers - TOP 20 by market cap
                return ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "BRK-B", "LLY", "V", 
                       "UNH", "JPM", "XOM", "JNJ", "AVGO", "PG", "MA", "HD", "MRK", "CVX", "COST"]
            time.sleep(2)  # Wait before retry

@st.cache_data(ttl=86400)
def get_nasdaq100_tickers():
    """Get list of Nasdaq-100 tickers"""
    for attempt in range(3):
        try:
            # Try from Wikipedia
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to load Wikipedia page: {response.status_code}")
                
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            # Look for the right table (the one with current components)
            for table in tables:
                caption = table.find('caption')
                if caption and ('companies' in caption.text.lower() or 'component' in caption.text.lower()):
                    break
            else:
                table = tables[4] if len(tables) > 4 else tables[0]  # Default fallback
            
            tickers = []
            for row in table.find_all('tr')[1:]:  # Skip header
                cells = row.find_all('td')
                if len(cells) >= 2:
                    ticker_cell = cells[1] if len(cells) >= 3 else cells[0]
                    ticker = ticker_cell.text.strip()
                    # Remove any ".XX" suffixes (like .L for London)
                    ticker = re.sub(r'\.\w+$', '', ticker)
                    tickers.append(ticker)
            
            # Validate we got a reasonable number of tickers
            if len(tickers) < 80:  # Nasdaq-100 should have ~100 tickers
                raise Exception(f"Found only {len(tickers)} tickers, expected around 100")
                
            return tickers
            
        except Exception as e:
            if attempt == 2:  # Last attempt, use fallback
                st.warning(f"Failed to fetch Nasdaq-100 tickers: {str(e)}. Using fallback list.")
                # Fallback tickers - TOP 20 by market cap
                return ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ASML", 
                        "COST", "PEP", "ADBE", "CSCO", "AMD", "NFLX", "CMCSA", "TMUS", "INTC", 
                        "INTU", "QCOM"]
            time.sleep(2)  # Wait before retry

@st.cache_data(ttl=86400)
def get_dowjones_tickers():
    """Get list of Dow Jones Industrial Average tickers"""
    for attempt in range(3):
        try:
            # Try from Wikipedia
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to load Wikipedia page: {response.status_code}")
                
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            # Look for the right table (the one with current companies)
            for table in tables:
                caption = table.find('caption')
                if caption and 'components' in caption.text.lower():
                    break
            else:
                table = tables[1] if len(tables) > 1 else tables[0]  # Default fallback
            
            tickers = []
            for row in table.find_all('tr')[1:]:  # Skip header
                cells = row.find_all('td')
                if len(cells) >= 2:
                    ticker_cell = cells[0]
                    ticker = ticker_cell.text.strip()
                    # Remove any ".XX" suffixes (like .L for London)
                    ticker = re.sub(r'\.\w+$', '', ticker)
                    tickers.append(ticker)
            
            # Validate we got a reasonable number of tickers
            if len(tickers) < 25:  # Dow Jones should have 30 tickers
                raise Exception(f"Found only {len(tickers)} tickers, expected 30")
                
            return tickers
            
        except Exception as e:
            if attempt == 2:  # Last attempt, use fallback
                st.warning(f"Failed to fetch Dow Jones tickers: {str(e)}. Using fallback list.")
                # Fallback tickers - All 30 DJIA
                return ["AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", 
                        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", 
                        "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"]
            time.sleep(2)  # Wait before retry

@st.cache_data(ttl=86400)
def get_popular_tickers():
    """Get popular tickers organized by sector"""
    sectors = {
        "Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "INTC", "AMD", "ORCL"],
        "Finance": ["JPM", "BAC", "WFC", "C", "GS", "V", "MA", "AXP", "BLK", "MS"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV", "ABT", "LLY", "TMO", "DHR", "BMY"],
        "Consumer": ["PG", "KO", "PEP", "WMT", "MCD", "SBUX", "NKE", "HD", "COST", "TGT"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "OXY", "MPC", "VLO", "KMI"]
    }
    all_tickers = []
    for sector_tickers in sectors.values():
        all_tickers.extend(sector_tickers)
    return all_tickers

def get_dividend_data(tickers, period="5y"):
    """Get dividend history for tickers with improved handling"""
    dividend_data = {}
    has_any_dividends = False
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # First approach - try to get dividends from history
            hist = stock.history(period=period)
            if 'Dividends' in hist.columns:
                dividends = hist['Dividends']
                dividends = dividends[dividends > 0]  # Filter out zeros
            else:
                # Alternative approach using the dividends property
                dividends = stock.dividends
                if dividends is None:
                    dividends = pd.Series(dtype=float)
            
            # Get latest price
            try:
                latest_price = stock.history(period="1d")['Close'].iloc[-1]
                if pd.isna(latest_price):
                    latest_price = stock.fast_info['lastPrice']  # Alternative way
            except:
                try:
                    latest_price = stock.info.get('regularMarketPrice', 0)
                except:
                    latest_price = 0
            
            # Check if we actually have dividend data and a valid price
            if not dividends.empty and latest_price > 0:
                has_any_dividends = True
                
                # Get annual dividend amounts (sum by year)
                try:
                    annual_div = dividends.resample('Y').sum()
                    
                    # Calculate dividend yield using last 12 months
                    if len(dividends) >= 4:  # If we have at least a year of quarterly dividends
                        latest_annual_div = dividends.tail(4).sum()
                    else:
                        # Use what we have and annualize
                        latest_annual_div = dividends.sum()
                        if len(dividends) > 0:
                            # Attempt to annualize partial year data
                            days_covered = (dividends.index.max() - dividends.index.min()).days
                            if days_covered > 30:  # Avoid division by very small numbers
                                annualized_factor = 365 / days_covered
                                latest_annual_div *= annualized_factor
                    
                    # Calculate yield - protect against zero prices
                    if latest_price > 0:
                        div_yield = (latest_annual_div / latest_price) * 100
                    else:
                        div_yield = 0
                    
                    dividend_data[ticker] = {
                        'annual_dividends': annual_div,
                        'latest_price': latest_price,
                        'yield': div_yield,
                        'has_dividends': True,
                        'total_dividend': latest_annual_div
                    }
                except Exception as e:
                    st.warning(f"Error processing dividend data for {ticker}: {str(e)}")
                    dividend_data[ticker] = {'has_dividends': False}
            else:
                dividend_data[ticker] = {
                    'has_dividends': False,
                    'latest_price': latest_price,
                    'yield': 0,
                    'total_dividend': 0
                }
        except Exception as e:
            st.warning(f"Could not get dividend data for {ticker}: {str(e)}")
            dividend_data[ticker] = {'has_dividends': False}
    
    # Add a message if no dividends found
    if not has_any_dividends:
        st.warning("No dividend data found for any selected stocks. Try a longer time period or different stocks.")
    
    return dividend_data

def get_news_sentiment(ticker, days=30):
    """
    Get recent news sentiment for a ticker
    (Placeholder for actual sentiment analysis)
    """
    try:
        # Simulate news sentiment (in a real implementation, we would use an API)
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if news:
            # Simple mock sentiment analysis
            sentiment_scores = []
            news_items = []
            
            for item in news[:5]:  # Analyze top 5 news
                # Extract date
                date = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d')
                
                # Simulate sentiment analysis
                title = item['title'].lower()
                positive_words = ['up', 'rise', 'gain', 'positive', 'growth', 'profit', 'boost', 'success']
                negative_words = ['down', 'fall', 'loss', 'negative', 'drop', 'decline', 'fail', 'cut']
                
                # Count occurrences
                pos_count = sum(word in title for word in positive_words)
                neg_count = sum(word in title for word in negative_words)
                
                # Calculate sentiment (-1 to 1)
                sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1)
                sentiment = max(min(sentiment, 1.0), -1.0)  # Clamp between -1 and 1
                
                sentiment_scores.append(sentiment)
                news_items.append({
                    'date': date,
                    'title': item['title'],
                    'sentiment': sentiment,
                    'url': item.get('link', '')
                })
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                'avg_sentiment': avg_sentiment,
                'news': news_items,
                'has_news': True
            }
        else:
            return {'has_news': False}
    except Exception as e:
        return {'has_news': False, 'error': str(e)}