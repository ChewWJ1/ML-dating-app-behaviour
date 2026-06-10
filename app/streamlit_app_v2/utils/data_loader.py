"""
SwipeIQ Streamlit Dashboard — Data Loader Utility
Handles caching and preprocessing of the dataset.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import RobustScaler

# Define paths relative to the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Data path: check data/ subfolder first, then fallback to root
_DATA_PATH_PRIMARY = os.path.join(ROOT_DIR, 'data', 'dating_app_behavior_dataset_extended1.csv')
_DATA_PATH_FALLBACK = os.path.join(ROOT_DIR, 'dating_app_behavior_dataset_extended1.csv')
DATA_PATH = _DATA_PATH_PRIMARY if os.path.exists(_DATA_PATH_PRIMARY) else _DATA_PATH_FALLBACK

MODELS_DIR = os.path.join(ROOT_DIR, 'models')

_EDA_STATS_MODELS = os.path.join(MODELS_DIR, 'eda_stats.json')
_EDA_STATS_ROOT = os.path.join(ROOT_DIR, 'eda_stats.json')
EDA_STATS_PATH = _EDA_STATS_MODELS if os.path.exists(_EDA_STATS_MODELS) else _EDA_STATS_ROOT

_CV_STATS_MODELS = os.path.join(MODELS_DIR, 'cv_stats.json')
_CV_STATS_ROOT = os.path.join(ROOT_DIR, 'cv_stats.json')
CV_STATS_PATH = _CV_STATS_MODELS if os.path.exists(_CV_STATS_MODELS) else _CV_STATS_ROOT

_FI_STATS_MODELS = os.path.join(MODELS_DIR, 'feature_importances.json')
_FI_STATS_ROOT = os.path.join(ROOT_DIR, 'feature_importances.json')
FI_STATS_PATH = _FI_STATS_MODELS if os.path.exists(_FI_STATS_MODELS) else _FI_STATS_ROOT

# V5 models directory
MODELS_V5_DIR = os.path.join(ROOT_DIR, 'models_v5')


# ── Plot Path Helpers ──────────────────────────────────────────────────────────

def get_v8_plot_path(filename):
    """Return the full path to a V5 plot image."""
    return os.path.join(ROOT_DIR, 'assets', 'v8 plots', filename)


def get_notebook_plot_path(filename):
    """Return the full path to a notebook plot image."""
    return os.path.join(ROOT_DIR, 'assets', 'notebook_plots', filename)


def get_base_plot_path(filename):
    """Return the full path to a base plot image."""
    return os.path.join(ROOT_DIR, 'assets', 'plots', filename)


# ── V5 Artifact Loader ────────────────────────────────────────────────────────

@st.cache_resource
def load_v5_artifact(filename):
    """Load a joblib artifact from the models_v5 directory with error handling."""
    filepath = os.path.join(MODELS_V5_DIR, filename)
    if not os.path.exists(filepath):
        st.warning(f"V5 artifact not found: {filepath}")
        return None
    try:
        return joblib.load(filepath)
    except Exception as e:
        st.error(f"Error loading V5 artifact '{filename}': {e}")
        return None


# ── Data Loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_raw_data():
    """Load the raw extended dataset."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_eda_stats():
    """Load precomputed EDA statistics from JSON."""
    if not os.path.exists(EDA_STATS_PATH):
        return {}
    with open(EDA_STATS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_cv_stats():
    """Load cross-validation statistics from JSON."""
    if not os.path.exists(CV_STATS_PATH):
        return {}
    with open(CV_STATS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_feature_importances():
    """Load feature importance data from JSON."""
    if not os.path.exists(FI_STATS_PATH):
        return {}
    with open(FI_STATS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_preprocessed_data():
    """
    Replicates the notebook preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler, encoders
    We don't actually split train/test here to save memory, we just return the full X and y
    since this is mainly for the prediction pipeline.
    """
    df = load_raw_data().copy()
    if df.empty:
        return None, None, None, None
        
    # 1. Drop redundant columns
    cols_to_drop = ['app_usage_time_label', 'swipe_right_label']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    # 2. Binary target
    positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
    if 'match_outcome' in df.columns:
        y = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)
        df.drop(columns=['match_outcome'], inplace=True)
    else:
        y = pd.Series([0] * len(df))
        
    # 3. Ordinal Encoding mappings
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
        df['income_enc'] = df['income_bracket'].map(income_map)
        df.drop(columns=['income_bracket'], inplace=True)
    if 'education_level' in df.columns:
        df['education_enc'] = df['education_level'].apply(map_education)
        df.drop(columns=['education_level'], inplace=True)
        
    # 4. One-Hot Encoding
    nominal_cols = ['gender', 'sexual_orientation', 'location_type', 
                    'swipe_time_of_day', 'body_type', 'relationship_intent', 'zodiac_sign']
    nominal_cols = [c for c in nominal_cols if c in df.columns]
    
    # We must ensure consistent dummy columns for inference. We'll use get_dummies for now
    # and in the forecaster we align the columns.
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
    
    # 5. Multi-Hot Encoding for interest_tags
    if 'interest_tags' in df.columns:
        # Simplified manual multi-hot encoding for the top tags to avoid dependency on MultiLabelBinarizer
        tags_lists = df['interest_tags'].fillna('').str.split(', ')
        all_unique_tags = set(tag.strip() for tags in tags_lists for tag in tags if tag.strip())
        
        for tag in all_unique_tags:
            df[f'interest_{tag}'] = tags_lists.apply(lambda x: 1 if tag in x else 0)
            
        df.drop(columns=['interest_tags'], inplace=True)
        
    # 5.5 Feature Engineering [V8]
    if 'likes_received' in df.columns:
        df['engagement_score'] = df['likes_received'] * df['swipe_right_ratio'] * df['message_sent_count']
        df['profile_completeness'] = df['profile_pics_count'] * df['bio_length']
        df['activity_intensity'] = df['app_usage_time_min'] * df['emoji_usage_rate']
        df['selectivity_ratio'] = df['message_sent_count'] / (df['likes_received'] + 1)
        df['late_night_user'] = ((df['last_active_hour'] >= 22) | (df['last_active_hour'] <= 4)).astype(int)
        
        for col in ['likes_received', 'message_sent_count', 'bio_length', 'app_usage_time_min']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])

    # 6. RobustScaler
    numeric_cols = ['age', 'height_cm', 'weight_kg', 'app_usage_time_min',
                    'swipe_right_ratio', 'likes_received', 'profile_pics_count', 
                    'bio_length', 'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
                    'engagement_score', 'profile_completeness', 'activity_intensity', 'selectivity_ratio',
                    'likes_received_log', 'message_sent_count_log', 'bio_length_log', 'app_usage_time_min_log']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, y, list(df.columns), scaler
