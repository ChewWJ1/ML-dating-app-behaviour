"""
SwipeIQ Streamlit Dashboard — Model Loader Utility
Handles caching and loading of joblib model files.
"""
import os
import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINE_PATH = os.path.join(ROOT_DIR, 'baseline_results.joblib')
TUNED_PATH = os.path.join(ROOT_DIR, 'tuned_results.joblib')

@st.cache_resource
def load_baseline_models():
    """Load the baseline_results.joblib file."""
    if not os.path.exists(BASELINE_PATH):
        st.error(f"Baseline models not found at {BASELINE_PATH}")
        return {}
    try:
        return joblib.load(BASELINE_PATH)
    except Exception as e:
        st.error(f"Error loading baseline models: {e}")
        return {}

@st.cache_resource
def load_tuned_models():
    """Load the tuned_results.joblib file."""
    if not os.path.exists(TUNED_PATH):
        return {}
    try:
        return joblib.load(TUNED_PATH)
    except Exception as e:
        st.error(f"Error loading tuned models: {e}")
        return {}

def preprocess_single_input(input_dict, full_cols, feature_columns, scaler):
    """
    Preprocess a single user's input dictionary to match the training feature shape.
    input_dict should have raw values for all necessary columns.
    """
    # Create a single-row dataframe
    df = pd.DataFrame([input_dict])
    
    # Ordinal encode
    income_map = {
        'Very Low': 0, 'Low': 0,
        'Lower-Middle': 1, 'Middle': 1, 'Upper-Middle': 1,
        'High': 2, 'Very High': 2
    }
    def map_education(val):
        val = str(val)
        if any(k in val for k in ['No Formal', 'High School', 'Diploma']): return 0
        elif any(k in val for k in ['Associate', 'Bachelor']): return 1
        elif any(k in val for k in ['Master', 'MBA', 'PhD', 'Postdoc']): return 2
        return 0
        
    if 'income_bracket' in df.columns:
        df['income_enc'] = df['income_bracket'].map(income_map).fillna(1)
    if 'education_level' in df.columns:
        df['education_enc'] = df['education_level'].apply(map_education)
        
    # One-hot encode manually to ensure exact column alignment
    nominal_cols = ['gender', 'sexual_orientation', 'location_type', 
                    'swipe_time_of_day', 'body_type', 'relationship_intent', 'zodiac_sign']
    
    # Create an empty dataframe with ALL preprocessed columns (113 cols) for correct scaling
    df_aligned = pd.DataFrame(0, index=[0], columns=full_cols)
    
    # Fill in numericals and ordinals
    for col in df.columns:
        if col in df_aligned.columns and col not in nominal_cols and col != 'interest_tags':
            df_aligned[col] = df[col]
            
    # Fill in one-hot columns
    for col in nominal_cols:
        if col in df.columns:
            val = df.at[0, col]
            dummy_col = f"{col}_{val}"
            if dummy_col in df_aligned.columns:
                df_aligned[dummy_col] = 1
                
    # Fill in multi-hot columns
    if 'interest_tags' in df.columns:
        tags = df.at[0, 'interest_tags']
        if isinstance(tags, list):
            for tag in tags:
                tag_col = f"interest_{tag}"
                if tag_col in df_aligned.columns:
                    df_aligned[tag_col] = 1
                    
    # Scale numericals
    numeric_cols = ['age', 'height_cm', 'weight_kg', 'app_usage_time_min',
                    'swipe_right_ratio', 'likes_received', 'mutual_matches',
                    'profile_pics_count', 'bio_length', 'message_sent_count',
                    'emoji_usage_rate', 'last_active_hour']
    numeric_cols = [c for c in numeric_cols if c in df_aligned.columns]
    
    # We must scale using the fitted scaler (which expects all 12 columns)
    if scaler and len(numeric_cols) > 0:
        try:
            df_numeric = df_aligned[numeric_cols]
            df_scaled = scaler.transform(df_numeric)
            df_aligned[numeric_cols] = df_scaled
        except Exception as e:
            st.warning(f"Scaling error: {e}")
            
    # Finally, filter down to ONLY the features the model was trained on
    # (after feature selection, which is ~67 columns)
    df_final = df_aligned[feature_columns]
            
    return df_final

def predict_single_user(model_dict, input_dict, full_cols, feature_columns, scaler):
    """
    Given a model_dict (from baseline_results), preprocess input and predict.
    """
    model = model_dict.get('model')
    if not model:
        return None, 0.0
        
    X_input = preprocess_single_input(input_dict, full_cols, feature_columns, scaler)
    
    # Ensure columns exactly match what model expects
    # (If the model was trained after feature selection, it expects 67 columns)
    # For now we'll assume the model uses all feature_columns. If feature selection was used,
    # the feature_columns list passed in must be exactly the 67 selected features.
    
    try:
        # predict_proba returns [[prob_0, prob_1]]
        probs = model.predict_proba(X_input)
        prob_1 = float(probs[0][1])
        prediction = 1 if prob_1 >= 0.5 else 0
        return prediction, prob_1
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0
