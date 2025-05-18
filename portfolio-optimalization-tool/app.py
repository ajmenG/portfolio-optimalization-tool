import streamlit as st
from ui.translations import get_translation, setup_language_selection
from ui.sidebar import create_sidebar
from ui.ticker_selector import create_ticker_selection
from ui.analysis import run_portfolio_analysis
from models.model_wrappers import MODEL_WRAPPERS

def main():
    # Set wide layout for better visualization of charts
    st.set_page_config(
        page_title="Portfolio Optimization Tool",
        page_icon="📈",
        layout="wide"
    )
    
    # Set up language and translation function
    lang = setup_language_selection()
    t = lambda key: get_translation(key, lang)
    
    # Set page title
    st.title(t('title'))
    
    # Set up sidebar - now returns both risk profile and optimization model
    risk_profile, optimization_model = create_sidebar(t)
    
    # Select tickers
    selected_tickers, period = create_ticker_selection(t)
    
    # Run portfolio analysis when button is clicked
    if st.button(t('analyze_portfolio')):
        if selected_tickers:
            # Get the selected optimization function
            optimize_fn = MODEL_WRAPPERS.get(optimization_model)
            if optimize_fn:
                run_portfolio_analysis(selected_tickers, period, risk_profile, t, optimize_fn)
            else:
                st.error(f"Unsupported optimization model: {optimization_model}")
        else:
            st.error(t('select_at_least_one'))

if __name__ == "__main__":
    main()