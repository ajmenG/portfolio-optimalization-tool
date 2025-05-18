import streamlit as st

def create_sidebar(t):
    """
    Create the sidebar with risk assessment questionnaire
    
    Parameters:
    - t: translation function
    
    Returns:
    - risk_profile: string representing the user's risk profile
    - optimization_model: string representing the selected optimization model
    """
    st.sidebar.write(f"## {t('risk_profile')}")

    with st.sidebar.expander(t('risk_questionnaire')):
        st.write(t('answer_questions'))
        
        q1 = st.radio(
            t('investment_horizon'),
            [t('horizon_less_1y'), t('horizon_1_3y'), t('horizon_3_5y'), 
             t('horizon_5_10y'), t('horizon_more_10y')],
            index=2
        )
        
        q2 = st.radio(
            t('portfolio_drop'),
            [t('drop_sell_all'), t('drop_sell_some'), 
             t('drop_hold'), t('drop_buy')],
            index=2
        )
        
        q3 = st.radio(
            t('investment_goal'),
            [t('goal_preserve'), t('goal_income'), 
             t('goal_balanced'), t('goal_growth')],
            index=2
        )
        
        q4 = st.radio(
            t('investment_percent'),
            [t('percent_less_10'), t('percent_10_25'), t('percent_25_50'), 
             t('percent_50_75'), t('percent_more_75')],
            index=1
        )
        
        q5 = st.radio(
            t('experience'),
            [t('exp_none'), t('exp_limited'), t('exp_moderate'), 
             t('exp_extensive'), t('exp_professional')],
            index=2
        )
        
        # Score the responses
        q1_scores = {t('horizon_less_1y'): 1, t('horizon_1_3y'): 2, t('horizon_3_5y'): 3, 
                    t('horizon_5_10y'): 4, t('horizon_more_10y'): 5}
        q2_scores = {t('drop_sell_all'): 1, t('drop_sell_some'): 2, 
                    t('drop_hold'): 3, t('drop_buy'): 5}
        q3_scores = {t('goal_preserve'): 1, t('goal_income'): 2, 
                    t('goal_balanced'): 3, t('goal_growth'): 5}
        q4_scores = {t('percent_less_10'): 5, t('percent_10_25'): 4, t('percent_25_50'): 3, 
                    t('percent_50_75'): 2, t('percent_more_75'): 1}
        q5_scores = {t('exp_none'): 1, t('exp_limited'): 2, t('exp_moderate'): 3, 
                    t('exp_extensive'): 4, t('exp_professional'): 5}
        
        total_score = q1_scores[q1] + q2_scores[q2] + q3_scores[q3] + q4_scores[q4] + q5_scores[q5]
        
        # Determine risk profile
        if total_score < 10:
            risk_profile = "Conservative"
            risk_desc = t('goal_preserve')
            optimal_strategy = t('model_min_vol')
        elif total_score < 15:
            risk_profile = "Moderately Conservative"
            risk_desc = t('goal_income')
            optimal_strategy = t('model_min_vol')
        elif total_score < 20:
            risk_profile = "Balanced"
            risk_desc = t('goal_balanced')
            optimal_strategy = t('model_markowitz')
        elif total_score < 23:
            risk_profile = "Growth"
            risk_desc = t('goal_growth')
            optimal_strategy = t('model_markowitz')
        else:
            risk_profile = "Aggressive Growth"
            risk_desc = t('goal_growth')
            optimal_strategy = t('model_max_ret')
        
        st.write(f"### {t('your_risk_profile')} {risk_profile}")
        st.write(risk_desc)
        st.write(f"{t('recommended_strategy')} **{optimal_strategy}**")
        
        # Store in session state
        st.session_state.risk_profile = risk_profile
        st.session_state.recommended_strategy = optimal_strategy
    
    # Add optimization model selection
    st.sidebar.write(f"## {t('optimization_model')}")
    
    # Create a mapping between display names and model keys
    model_options = {
        t('model_markowitz'): 'model_markowitz',
        t('model_min_vol'): 'model_min_vol',
        t('model_max_ret'): 'model_max_ret',
        t('model_risk_parity'): 'model_risk_parity',
        t('model_hrp'): 'model_hrp',
        t('model_black_litterman'): 'model_black_litterman'
    }
    
    # Display selection with translated names
    selected_display_name = st.sidebar.selectbox(
        t('select_model'),
        list(model_options.keys()),
        index=0
    )
    
    # Convert display name back to model key
    optimization_model = model_options[selected_display_name]
    
    # Store in session state
    st.session_state.optimization_model = optimization_model
    
    return risk_profile, optimization_model