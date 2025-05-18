import streamlit as st
from typing import Dict, Optional

# Define translations
translations = {
    'en': {
        'title': "Portfolio Optimization Tool",
        'overview': "A powerful Streamlit application that implements Harry Markowitz's Modern Portfolio Theory (MPT) to help investors optimize their investment portfolios.",
        'ticker_search': "Ticker Search",
        'predefined_packages': "Predefined Packages",
        'select_tickers': "Select or type stock symbols:",
        'add_custom_ticker': "Or add a custom ticker:",
        'check_add_ticker': "Check and add ticker",
        'added_to_list': "added to the list",
        'invalid_symbol': "is not a valid symbol",
        'select_index': "Select stock index:",
        'load_tickers': "Load tickers from index",
        'tickers_loaded': "tickers loaded from",
        'selected_tickers': "Selected tickers:",
        'analyze_period': "Select analysis period:",
        'analyze_portfolio': "Analyze Portfolio",
        'price_data': "Price Data",
        'price_chart': "Price Chart",
        'correlation_matrix': "Correlation Matrix",
        'optimal_portfolio': "Optimal Portfolio Weights (Maximum Sharpe Ratio)",
        'performance_metrics': "Performance Metrics",
        'min_volatility': "Minimum Volatility Portfolio",
        'max_return': "Maximum Return Portfolio",
        'efficient_frontier': "Efficient Frontier",
        'monte_carlo': "Monte Carlo Simulation",
        'strategies_comparison': "Portfolio Strategies Comparison",
        'return_distribution': "Return Distribution",
        'select_at_least_one': "Select at least one stock ticker for analysis",
        'backtest_results': "Backtesting Results",
        'select_benchmark': "Select benchmark:",
        'portfolio_return': "Portfolio Return",
        'benchmark_return': "Benchmark Return",
        'outperformance': "Outperformance",
        'sector_analysis': "Sector Analysis",
        'sector_allocation': "Sector Allocation",
        'sector_risk': "Sector Risk Contribution",
        'dividend_analysis': "Dividend Analysis",
        'portfolio_yield': "Portfolio Dividend Yield",
        'dividend_details': "Dividend Details by Stock",
        'optimization_settings': "Portfolio Optimization Settings",
        'standard_optimization': "Standard Optimization",
        'constrained_optimization': "Constrained Optimization",
        'min_weight': "Minimum weight per asset (%)",
        'max_weight': "Maximum weight per asset (%)",
        'sector_constraints': "Sector Constraints",
        'sector_constraints_desc': "Define minimum and maximum allocations for each market sector",
        'min_allocation': "Min % for",
        'max_allocation': "Max % for",
        'rebalancing': "Portfolio Rebalancing",
        'rebalancing_info': "Enter your current portfolio holdings to get rebalancing recommendations.",
        'current_portfolio': "Enter Current Portfolio",
        'total_value': "Total Portfolio Value ($)",
        'current_holdings': "Enter your current holdings as percentages:",
        'generate_plan': "Generate Rebalancing Plan",
        'rebalancing_recommendations': "Rebalancing Recommendations",
        'current_vs_target': "Current vs Target Allocation",
        'portfolio_comparison': "Portfolio Comparison",
        'risk_profile': "Risk Profile",
        'risk_questionnaire': "Risk Assessment Questionnaire",
        'answer_questions': "Answer these questions to determine your risk profile:",
        'investment_horizon': "1. How long is your investment horizon?",
        'horizon_less_1y': "Less than 1 year",
        'horizon_1_3y': "1-3 years",
        'horizon_3_5y': "3-5 years",
        'horizon_5_10y': "5-10 years",
        'horizon_more_10y': "More than 10 years",
        'portfolio_drop': "2. What would you do if your portfolio dropped by 20%?",
        'drop_sell_all': "Sell everything to prevent further losses",
        'drop_sell_some': "Sell some investments to reduce risk",
        'drop_hold': "Do nothing and wait for recovery",
        'drop_buy': "Buy more at lower prices",
        'investment_goal': "3. Which statement best describes your investment goal?",
        'goal_preserve': "Preserve capital with minimal risk",
        'goal_income': "Generate income with moderate growth",
        'goal_balanced': "Achieve balanced growth and income",
        'goal_growth': "Maximize long-term growth",
        'investment_percent': "4. What percentage of your total assets are you investing?",
        'percent_less_10': "Less than 10%",
        'percent_10_25': "10-25%",
        'percent_25_50': "25-50%",
        'percent_50_75': "50-75%",
        'percent_more_75': "More than 75%",
        'experience': "5. How much investment experience do you have?",
        'exp_none': "None",
        'exp_limited': "Limited",
        'exp_moderate': "Moderate",
        'exp_extensive': "Extensive",
        'exp_professional': "Professional",
        'your_risk_profile': "Your Risk Profile:",
        'recommended_strategy': "Recommended strategy:",
        'optimization_model': "Optimization Model",
        'select_model': "Select optimization model:",
        'model_markowitz': "Markowitz (Maximum Sharpe Ratio)",
        'model_min_vol': "Minimum Volatility",
        'model_max_ret': "Maximum Return",
        'model_risk_parity': "Risk Parity",
        'model_hrp': "Hierarchical Risk Parity",
        'model_black_litterman': "Black-Litterman",
        'checking': "Checking",
        'choose_index': "Choose an index",
        'popular_by_sector': "Popular from each sector",
        'loading_tickers': "Loading tickers...",
        'from_different_sectors': "from different sectors",
        'period_options': "Period options:",
        'standard_period': "Standard",
        'custom_period': "Custom",
        'select_period': "Select period:",
        'select_years': "Select number of years of historical data:",
        'RDDT': "Recent"
    },
    'pl': {
        'title': "Narzędzie Optymalizacji Portfela",
        'overview': "Potężna aplikacja Streamlit, która implementuje Nowoczesną Teorię Portfela Harry'ego Markowitza (MPT), aby pomóc inwestorom zoptymalizować ich portfele inwestycyjne.",
        'ticker_search': "Wyszukiwarka tickerów",
        'predefined_packages': "Predefiniowane pakiety",
        'select_tickers': "Wybierz lub wpisz symbole akcji:",
        'add_custom_ticker': "Lub dodaj własny ticker:",
        'check_add_ticker': "Sprawdź i dodaj ticker",
        'added_to_list': "dodano do listy",
        'invalid_symbol': "nie jest poprawnym symbolem",
        'select_index': "Wybierz indeks giełdowy:",
        'load_tickers': "Załaduj tickery z indeksu",
        'tickers_loaded': "tickerów załadowanych z",
        'selected_tickers': "Wybrane tickery:",
        'analyze_period': "Wybierz okres analizy:",
        'analyze_portfolio': "Wykonaj analizę portfela",
        'price_data': "Dane cenowe",
        'price_chart': "Wykres cen",
        'correlation_matrix': "Macierz korelacji",
        'optimal_portfolio': "Optymalne wagi portfela (Maksymalny wskaźnik Sharpe'a)",
        'performance_metrics': "Wskaźniki efektywności",
        'min_volatility': "Portfel minimalnej zmienności",
        'max_return': "Portfel maksymalnego zwrotu",
        'efficient_frontier': "Granica efektywna",
        'monte_carlo': "Symulacja Monte Carlo",
        'strategies_comparison': "Porównanie strategii portfelowych",
        'return_distribution': "Rozkład zwrotów",
        'select_at_least_one': "Wybierz przynajmniej jeden ticker akcji do analizy",
        'backtest_results': "Wyniki backtestingu",
        'select_benchmark': "Wybierz benchmark:",
        'portfolio_return': "Zwrot z portfela",
        'benchmark_return': "Zwrot z benchmarku",
        'outperformance': "Wynik ponad benchmark",
        'sector_analysis': "Analiza sektorowa",
        'sector_allocation': "Alokacja sektorowa",
        'sector_risk': "Kontrybucja ryzyka sektorowego",
        'dividend_analysis': "Analiza dywidend",
        'portfolio_yield': "Stopa dywidendy portfela",
        'dividend_details': "Szczegóły dywidend według akcji",
        'optimization_settings': "Ustawienia optymalizacji portfela",
        'standard_optimization': "Standardowa optymalizacja",
        'constrained_optimization': "Optymalizacja z ograniczeniami",
        'min_weight': "Minimalna waga na składnik (%)",
        'max_weight': "Maksymalna waga na składnik (%)",
        'sector_constraints': "Ograniczenia sektorowe",
        'sector_constraints_desc': "Zdefiniuj minimalną i maksymalną alokację dla każdego sektora rynku",
        'min_allocation': "Min % dla",
        'max_allocation': "Max % dla",
        'rebalancing': "Rebalansowanie portfela",
        'rebalancing_info': "Wprowadź aktualny stan portfela, aby otrzymać rekomendacje rebalansowania.",
        'current_portfolio': "Wprowadź aktualny portfel",
        'total_value': "Całkowita wartość portfela ($)",
        'current_holdings': "Wprowadź aktualne pozycje jako procenty:",
        'generate_plan': "Wygeneruj plan rebalansowania",
        'rebalancing_recommendations': "Rekomendacje rebalansowania",
        'current_vs_target': "Aktualna vs docelowa alokacja",
        'portfolio_comparison': "Porównanie portfeli",
        'risk_profile': "Profil Ryzyka",
        'risk_questionnaire': "Kwestionariusz Oceny Ryzyka",
        'answer_questions': "Odpowiedz na pytania, aby określić swój profil ryzyka:",
        'investment_horizon': "1. Jaki jest Twój horyzont inwestycyjny?",
        'horizon_less_1y': "Mniej niż 1 rok",
        'horizon_1_3y': "1-3 lata",
        'horizon_3_5y': "3-5 lat",
        'horizon_5_10y': "5-10 lat",
        'horizon_more_10y': "Powyżej 10 lat",
        'portfolio_drop': "2. Co byś zrobił, gdyby Twój portfel stracił 20% wartości?",
        'drop_sell_all': "Sprzedał wszystko, aby zapobiec dalszym stratom",
        'drop_sell_some': "Sprzedał część inwestycji, aby zmniejszyć ryzyko",
        'drop_hold': "Nic nie robił i czekał na odbicie",
        'drop_buy': "Dokupił więcej po niższej cenie",
        'investment_goal': "3. Które stwierdzenie najlepiej opisuje Twój cel inwestycyjny?",
        'goal_preserve': "Zachowanie kapitału przy minimalnym ryzyku",
        'goal_income': "Generowanie dochodu z umiarkowanym wzrostem",
        'goal_balanced': "Zrównoważony wzrost i dochód",
        'goal_growth': "Maksymalizacja długoterminowego wzrostu",
        'investment_percent': "4. Jaki procent swoich aktywów inwestujesz?",
        'percent_less_10': "Mniej niż 10%",
        'percent_10_25': "10-25%",
        'percent_25_50': "25-50%",
        'percent_50_75': "50-75%",
        'percent_more_75': "Powyżej 75%",
        'experience': "5. Jakie masz doświadczenie inwestycyjne?",
        'exp_none': "Brak",
        'exp_limited': "Ograniczone",
        'exp_moderate': "Umiarkowane",
        'exp_extensive': "Duże",
        'exp_professional': "Profesjonalne",
        'your_risk_profile': "Twój Profil Ryzyka:",
        'recommended_strategy': "Rekomendowana strategia:",
        'optimization_model': "Model Optymalizacji",
        'select_model': "Wybierz model optymalizacji:",
        'model_markowitz': "Markowitz (Maksymalny wskaźnik Sharpe'a)",
        'model_min_vol': "Minimalna Zmienność",
        'model_max_ret': "Maksymalny Zwrot",
        'model_risk_parity': "Parytet Ryzyka",
        'model_hrp': "Hierarchiczny Parytet Ryzyka",
        'model_black_litterman': "Black-Litterman",
        'checking': "Sprawdzanie",
        'choose_index': "Wybierz indeks",
        'popular_by_sector': "Popularne z każdego sektora",
        'loading_tickers': "Ładowanie tickerów...",
        'from_different_sectors': "z różnych sektorów",
        'period_options': "Opcje okresu:",
        'standard_period': "Standardowy",
        'custom_period': "Niestandardowy",
        'select_period': "Wybierz okres:",
        'select_years': "Wybierz liczbę lat danych historycznych:",
        'RDDT': "Ostatni"
    }
}

# Define supported languages
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Polski": "pl"
}

class TranslationError(Exception):
    """Custom exception for translation errors"""
    pass

def validate_translations():
    """Validate that all translations have the same keys"""
    if not translations or not isinstance(translations, dict):
        raise TranslationError("Invalid translations dictionary")
        
    # Get keys from first language
    first_lang = next(iter(translations.values()))
    reference_keys = set(first_lang.keys())
    
    # Check all languages have the same keys
    for lang, trans in translations.items():
        current_keys = set(trans.keys())
        missing_keys = reference_keys - current_keys
        extra_keys = current_keys - reference_keys
        
        if missing_keys:
            raise TranslationError(f"Language {lang} is missing translations for: {missing_keys}")
        if extra_keys:
            raise TranslationError(f"Language {lang} has extra translations for: {extra_keys}")

def setup_language_selection() -> str:
    """
    Set up language selection in sidebar
    Returns: language code (en/pl)
    """
    try:
        st.sidebar.title("Settings / Ustawienia")
        selected_language = st.sidebar.selectbox(
            "Language / Język",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=1  # Default to Polish
        )
        
        # Map selection to language code
        lang_code = SUPPORTED_LANGUAGES.get(selected_language)
        if not lang_code:
            st.error(f"Unsupported language: {selected_language}")
            return "en"  # Fallback to English
            
        return lang_code
        
    except Exception as e:
        st.error(f"Error setting up language selection: {str(e)}")
        return "en"  # Fallback to English

def get_translation(key: str, lang: str) -> str:
    """
    Get translated text for a given key and language
    
    Parameters:
    - key: Translation key to look up
    - lang: Language code (en/pl)
    
    Returns:
    - Translated text or key if translation not found
    """
    try:
        if lang not in translations:
            st.warning(f"Unsupported language code: {lang}")
            lang = "en"  # Fallback to English
            
        return translations[lang].get(key, key)
        
    except Exception as e:
        st.error(f"Error getting translation: {str(e)}")
        return key  # Return key as fallback

# Validate translations on module import
try:
    validate_translations()
except TranslationError as e:
    st.error(f"Translation validation failed: {str(e)}")