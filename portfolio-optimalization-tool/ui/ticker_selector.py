import streamlit as st
from data.fetcher import get_popular_tickers, get_sp500_tickers, get_nasdaq100_tickers, get_dowjones_tickers
from data.processor import is_valid_ticker

def create_ticker_selection(t):
    """
    Create UI for ticker selection
    
    Parameters:
    - t: translation function
    
    Returns:
    - selected_tickers: list of selected ticker symbols
    - period: string representing the selected time period
    """
    # Initialize session state for tickers if not exists
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ["AAPL", "MSFT", "GOOG"]
    
    # Create tabs for different selection methods
    tab1, tab2 = st.tabs([t('ticker_search'), t('predefined_packages')])
    
    with tab1:
        # Ticker autocomplete
        all_tickers = get_popular_tickers()
        
        # Ensure default tickers are in options
        default_tickers = ["AAPL", "MSFT", "GOOG"]
        options = list(set(all_tickers + default_tickers))
        
        # Make sure session state tickers are in options or add them
        for ticker in st.session_state.selected_tickers:
            if ticker not in options:
                options.append(ticker)
        
        # Filter defaults to only include tickers in options
        default_values = [ticker for ticker in st.session_state.selected_tickers if ticker in options]
        
        # Multiselect for tickers
        selected = st.multiselect(
            t('select_tickers'),
            options=options,
            default=default_values,
            help=t('add_custom_ticker')
        )
        
        # Update session state with selected tickers
        if selected:
            st.session_state.selected_tickers = selected
        
        # Custom ticker option
        custom_ticker = st.text_input(t('add_custom_ticker'))
        
        if st.button(t('check_add_ticker')):
            if custom_ticker and custom_ticker not in st.session_state.selected_tickers:
                with st.spinner(f"{t('checking')} {custom_ticker}..."):
                    if is_valid_ticker(custom_ticker):
                        # Add ticker to list
                        st.session_state.selected_tickers = st.session_state.selected_tickers + [custom_ticker]
                        st.success(f"{custom_ticker} {t('added_to_list')}")
                        
                        # Use proper rerun function depending on Streamlit version
                        try:
                            # Newer Streamlit version (>=1.20)
                            st.rerun()
                        except AttributeError:
                            # Older Streamlit version
                            st.experimental_rerun()
                    else:
                        st.error(f"{custom_ticker} {t('invalid_symbol')}")
    
    with tab2:
        index_option = st.selectbox(
            t('select_index'),
            [t('choose_index'), "S&P 500 (top 20)", "Nasdaq-100 (top 20)", "Dow Jones", t('popular_by_sector')]
        )
        
        if st.button(t('load_tickers')):
            with st.spinner(t('loading_tickers')):
                if index_option == "S&P 500 (top 20)":
                    tickers = []
                    try:
                        sp500_tickers = get_sp500_tickers()
                        tickers = sp500_tickers[:20] if len(sp500_tickers) >= 20 else sp500_tickers
                    except Exception as e:
                        st.error(f"Error fetching S&P 500: {str(e)}")
                    
                    if tickers:
                        st.session_state.selected_tickers = tickers
                        st.success(f"{len(tickers)} {t('tickers_loaded')} S&P 500")
                        st.rerun()
                
                elif index_option == "Nasdaq-100 (top 20)":
                    tickers = []
                    try:
                        nasdaq_tickers = get_nasdaq100_tickers()
                        tickers = nasdaq_tickers[:20] if len(nasdaq_tickers) >= 20 else nasdaq_tickers
                    except Exception as e:
                        st.error(f"Error fetching Nasdaq-100: {str(e)}")
                    
                    if tickers:
                        st.session_state.selected_tickers = tickers
                        st.success(f"{len(tickers)} {t('tickers_loaded')} Nasdaq-100")
                        st.rerun()
                
                elif index_option == "Dow Jones":
                    tickers = []
                    try:
                        tickers = get_dowjones_tickers()
                    except Exception as e:
                        st.error(f"Error fetching Dow Jones: {str(e)}")
                    
                    if tickers:
                        st.session_state.selected_tickers = tickers
                        st.success(f"{len(tickers)} {t('tickers_loaded')} Dow Jones")
                        st.rerun()
                
                elif index_option == t('popular_by_sector'):
                    sectors = {
                        "Tech": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                        "Finance": ["JPM", "V", "BAC", "GS"],
                        "Healthcare": ["JNJ", "PFE", "UNH", "MRK"],
                        "Consumer": ["PG", "KO", "WMT", "MCD"],
                        "Energy": ["XOM", "CVX", "COP", "SLB"]
                    }
                    tickers = []
                    for sector_tickers in sectors.values():
                        tickers.extend(sector_tickers)
                    
                    if tickers:
                        st.session_state.selected_tickers = tickers
                        st.success(f"{len(tickers)} {t('tickers_loaded')} {t('from_different_sectors')}")
                        st.rerun()
    
    # Display selected tickers
    if st.session_state.selected_tickers:
        st.write(f"### {t('selected_tickers')}")
        st.write(", ".join(st.session_state.selected_tickers))
    
    # Period selection
    st.write(f"### {t('analyze_period')}")
    period_option = st.radio(
        t('period_options'),
        [t('standard_period'), t('custom_period')]
    )
    
    if period_option == t('standard_period'):
        period = st.selectbox(
            t('select_period'),
            ["1y", "2y", "3y", "5y", "10y", "max"]
        )
    else:
        years = st.slider(
            t('select_years'),
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        period = f"{years}y"
    
    return st.session_state.selected_tickers, period