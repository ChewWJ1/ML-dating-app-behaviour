"""
SwipeIQ Streamlit Dashboard — Model Loader Utility
Handles caching and loading of joblib model files.
"""
import os
import sys
import types
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin

# --- DYNAMIC INJECTION FOR PICKLED DEEP LEARNING MODEL CLASSES & TORCH BYPASS ---
# Serialized baseline_results.joblib contains custom PyTorch modules and wrappers.
# Here we dynamically define them, inject them into main namespaces, and mock
# PyTorch dynamically using import hooks to prevent c10.dll native load failures.

class DynamicMock(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
            
        # Case-based heuristic: Uppercase = class/constant; Lowercase = submodule
        if name[0].isupper():
            class DummyClass:
                def __init__(self, *args, **kwargs): pass
                def to(self, *args, **kwargs): return self
                def train(self, *args, **kwargs): return self
                def eval(self, *args, **kwargs): return self
            DummyClass.__name__ = name
            return DummyClass
        else:
            mock_sub = DynamicMock(f"{self.__name__}.{name}")
            sys.modules[f"{self.__name__}.{name}"] = mock_sub
            setattr(self, name, mock_sub)
            return mock_sub
            
    def __call__(self, *args, **kwargs):
        # Allow acting as a class constructor or callable function
        return self

class PyTorchSklearnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class=None, lr=0.005, epochs=10, batch_size=512, device='cpu', **kwargs):
        self.model_class = model_class
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.kwargs = kwargs
        self.model = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return np.zeros(len(X))
        
    def predict_proba(self, X):
        return np.vstack([np.ones(len(X)), np.zeros(len(X))]).T

class FTTransformer:
    def __init__(self, *args, **kwargs): pass
class SAINT:
    def __init__(self, *args, **kwargs): pass
class NODE:
    def __init__(self, *args, **kwargs): pass
class FeatureTokenizer:
    def __init__(self, *args, **kwargs): pass
class ObliviousDecisionTree:
    def __init__(self, *args, **kwargs): pass

def inject_classes():
    """Dynamically inject unpickling wrappers and custom PyTorch import hooks into active namespaces."""
    import importlib.abc
    import importlib.machinery
    
    # Define and install custom import hook if not present
    class TorchMockFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'torch' or fullname.startswith('torch.'):
                return importlib.machinery.ModuleSpec(fullname, TorchMockLoader(fullname))
            return None

    class TorchMockLoader(importlib.abc.Loader):
        def __init__(self, fullname):
            self.fullname = fullname
        def create_module(self, spec):
            mock_mod = DynamicMock(self.fullname)
            sys.modules[self.fullname] = mock_mod
            return mock_mod
        def exec_module(self, module):
            pass

    # Avoid duplicate finder injection
    has_finder = any(x.__class__.__name__ == 'TorchMockFinder' for x in sys.meta_path if hasattr(x, '__class__'))
    if not has_finder:
        sys.meta_path.insert(0, TorchMockFinder())
        
    if 'torch' not in sys.modules:
        sys.modules['torch'] = DynamicMock('torch')

    for mod_name in ['__main__', 'main']:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            for name, cls in [
                ('PyTorchSklearnClassifier', PyTorchSklearnClassifier),
                ('FTTransformer', FTTransformer),
                ('SAINT', SAINT),
                ('NODE', NODE),
                ('FeatureTokenizer', FeatureTokenizer),
                ('ObliviousDecisionTree', ObliviousDecisionTree)
            ]:
                setattr(mod, name, cls)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# V5 models directory
MODELS_V5_DIR = os.path.join(ROOT_DIR, 'models_v5')

# Baseline path: check models_v5/ first, then fallback to root
_BASELINE_V5 = os.path.join(MODELS_V5_DIR, 'baseline_results.joblib')
_BASELINE_ROOT = os.path.join(ROOT_DIR, 'baseline_results.joblib')
BASELINE_PATH = _BASELINE_V5 if os.path.exists(_BASELINE_V5) else _BASELINE_ROOT

# Tuned path: check models_v5/ first, then fallback to root
_TUNED_V5 = os.path.join(MODELS_V5_DIR, 'tuned_results.joblib')
_TUNED_ROOT = os.path.join(ROOT_DIR, 'tuned_results.joblib')
TUNED_PATH = _TUNED_V5 if os.path.exists(_TUNED_V5) else _TUNED_ROOT


@st.cache_resource
def load_baseline_models():
    """Load the baseline_results.joblib file."""
    if not os.path.exists(BASELINE_PATH):
        # Silently return empty dict to trigger graceful image fallback
        return {}
    try:
        inject_classes()
        import sys
        mod = sys.modules.get('main')
        with open(r'C:\Users\HP\.gemini\antigravity\scratch\debug_main.txt', 'w') as f:
            f.write(f"Main module type: {type(mod)}\n")
            f.write(f"Main module file: {getattr(mod, '__file__', 'None')}\n")
            f.write(f"Main module keys in sys.modules: {'main' in sys.modules}\n")
            f.write(f"Attributes: {dir(mod) if mod else 'None'}\n")
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
        inject_classes()
        return joblib.load(TUNED_PATH)
    except Exception as e:
        st.error(f"Error loading tuned models: {e}")
        return {}

@st.cache_resource
def load_cv_results():
    """Load the cross-validation results from models_v5/cv_results.joblib."""
    cv_path = os.path.join(MODELS_V5_DIR, 'cv_results.joblib')
    if not os.path.exists(cv_path):
        return {}
    try:
        inject_classes()
        return joblib.load(cv_path)
    except Exception as e:
        st.error(f"Error loading CV results: {e}")
        return {}

@st.cache_resource
def load_learning_curve_data():
    """Load the learning curve data from models_v5/learning_curve_data.joblib."""
    lc_path = os.path.join(MODELS_V5_DIR, 'learning_curve_data.joblib')
    if not os.path.exists(lc_path):
        return {}
    try:
        inject_classes()
        return joblib.load(lc_path)
    except Exception as e:
        st.error(f"Error loading learning curve data: {e}")
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
