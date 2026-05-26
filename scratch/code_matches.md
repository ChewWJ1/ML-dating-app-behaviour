Cell 2 contains 'optuna': # Install required packages (run once in Colab)
!pip install -q pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn mapie category-encoders boruta causal-learn opacus lime fairlearn...

Cell 3 contains 'PyTorch': # --- DYNAMIC HARDWARE AUTO-DETECTION ENGINE ---
import torch
import os

def get_best_pytorch_device():
    # 1. NVIDIA CUDA GPU Acceleration
    if torch.cuda.is_available():
        print("🚀 [Hardwa...

Cell 50 contains 'RobustScaler': from sklearn.preprocessing import RobustScaler
numeric_cols = [
    'age', 'height_cm', 'weight_kg',
    'app_usage_time_min', 'swipe_right_ratio',
    'likes_received', 'mutual_matches',
    'profile...

Cell 55 contains 'class ': X = df.drop(columns=['target'])
y = df['target']

print(f'Feature matrix X: {X.shape}')
print(f'Target vector  y: {y.shape}')
print(f'\nClass balance:')
print(y.value_counts().rename({0: 'Negative', 1...

Cell 74 contains 'class ': # --- Split on ORIGINAL selected features (primary — used for most models) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.2,
    random_state=RANDOM_STATE,...

Cell 75 contains 'class ': # Visualise class balance in train and test sets
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, split_y, title in zip(axes, [y_train, y_test], ['Training Set', 'Test Set']):
    vc = split_y...

Cell 77 contains 'class ': # Apply SMOTE to perfectly balance training set (50/50 split) natively in the pipeline
from imblearn.over_sampling import SMOTE
print("🔄 Applying SMOTE to balance class distribution in training set......

Cell 77 contains 'SMOTE': # Apply SMOTE to perfectly balance training set (50/50 split) natively in the pipeline
from imblearn.over_sampling import SMOTE
print("🔄 Applying SMOTE to balance class distribution in training set......

Cell 84 contains 'class ': # --- ADVANCED DIFFERENTIAL NEURAL ARCHITECTURES & SKLEARN WRAPPER ---
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import...

Cell 84 contains 'nn.Module': # --- ADVANCED DIFFERENTIAL NEURAL ARCHITECTURES & SKLEARN WRAPPER ---
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import...

Cell 84 contains 'PyTorch': # --- ADVANCED DIFFERENTIAL NEURAL ARCHITECTURES & SKLEARN WRAPPER ---
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import...

Cell 84 contains 'deep': # --- ADVANCED DIFFERENTIAL NEURAL ARCHITECTURES & SKLEARN WRAPPER ---
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import...

Cell 105 contains 'class ': import os, joblib
os.makedirs('models_v4_cache', exist_ok=True)
cache_dp = 'models_v4_cache/opacus.joblib'

if os.path.exists(cache_dp):
    print("⏭️  Loading cached Differential Privacy model...")...

Cell 105 contains 'nn.Module': import os, joblib
os.makedirs('models_v4_cache', exist_ok=True)
cache_dp = 'models_v4_cache/opacus.joblib'

if os.path.exists(cache_dp):
    print("⏭️  Loading cached Differential Privacy model...")...

Cell 109 contains 'class ': # pip install torch-geometric
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
i...

Cell 109 contains 'nn.Module': # pip install torch-geometric
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
i...

Cell 113 contains 'class ': import os, joblib
os.makedirs('models_v4_cache', exist_ok=True)
cache_scarf = 'models_v4_cache/scarf.joblib'

if os.path.exists(cache_scarf):
    print("⏭️  Loading cached SCARF representations...")...

Cell 113 contains 'nn.Module': import os, joblib
os.makedirs('models_v4_cache', exist_ok=True)
cache_scarf = 'models_v4_cache/scarf.joblib'

if os.path.exists(cache_scarf):
    print("⏭️  Loading cached SCARF representations...")...

Cell 117 contains 'deep': # Define parameter grids for top models
param_grids = {
    'Random Forest': {
        'n_estimators': [200, 300, 500, 800, 1000],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_spl...

Cell 119 contains 'MLP': # --- GPU-ACCELERATED OPTUNA HYPERPARAMETER SEARCH ENGINE ---
import optuna
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_search(X_tr, y_tr, X_te, y_te):
    prin...

Cell 119 contains 'optuna': # --- GPU-ACCELERATED OPTUNA HYPERPARAMETER SEARCH ENGINE ---
import optuna
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_search(X_tr, y_tr, X_te, y_te):
    prin...

Cell 119 contains 'deep': # --- GPU-ACCELERATED OPTUNA HYPERPARAMETER SEARCH ENGINE ---
import optuna
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_search(X_tr, y_tr, X_te, y_te):
    prin...

Cell 119 contains 'mlp': # --- GPU-ACCELERATED OPTUNA HYPERPARAMETER SEARCH ENGINE ---
import optuna
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_search(X_tr, y_tr, X_te, y_te):
    prin...

Cell 128 contains 'deep': ---
## ⚖️ Section 12: Ethical Considerations in Dating App ML

Machine learning models deployed in human-centric domains like dating apps raise critical ethical concerns that must be addressed:

1. **...

Cell 141 contains 'Friedman': from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import itertools
import numpy as np
import matplotlib.pyplot as plt

def h_statistic(model, X, feature_i, feature_j, grid_re...

Cell 146 contains 'conformal': import os, joblib
os.makedirs('models_v4_cache', exist_ok=True)
cache_mapie = 'models_v4_cache/mapie.joblib'

if os.path.exists(cache_mapie):
    print("⏭️  Loading cached MAPIE Conformal Prediction s...

Cell 150 contains 'class ': import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BayesianMLP(nn.Module):
    """MLP with dropout kept ON during inference for MC Dropout"""
    def __init__...

Cell 150 contains 'MLP': import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BayesianMLP(nn.Module):
    """MLP with dropout kept ON during inference for MC Dropout"""
    def __init__...

Cell 150 contains 'nn.Module': import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BayesianMLP(nn.Module):
    """MLP with dropout kept ON during inference for MC Dropout"""
    def __init__...

Cell 150 contains 'mlp': import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BayesianMLP(nn.Module):
    """MLP with dropout kept ON during inference for MC Dropout"""
    def __init__...

Cell 154 contains 'PyTorch': import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def fgsm_attack(model, X, y, epsilon, device):
    """Fast Gradient Sign Method — generates adversarial examples"...

Cell 159 contains 'class ': import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Step 1: Get...

Cell 159 contains 'nn.Module': import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Step 1: Get...
