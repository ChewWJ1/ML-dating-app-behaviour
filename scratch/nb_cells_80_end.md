Cell 80 (markdown): ## 🤖 Section 9: Baseline Establishment via AutoML Establishing an automated benchmark before crafting custom architectur
Cell 81 (markdown): ## 🧠 Section 10: Advanced Model Training Training traditional baselines, deep learning models, and graph-based networks.
Cell 82 (markdown): ### 10.1 Define & Train All Models
Cell 83 (code): import xgboost as xgb  # Detect best available device for XGBoost 
Cell 84 (code): # --- ADVANCED DIFFERENTIAL NEURAL ARCHITECTURES & SKLEARN WRAPPER --- import torch.nn as nn import torch.optim as optim
Cell 85 (code): # Train all models and collect results (With smart SVM bypass and selector) import joblib import os 
Cell 86 (markdown): ### 10.2 Model Comparison Table
Cell 87 (code): # Build comparison dataframe if results:     comparison = pd.DataFrame({ 
Cell 88 (code): # Visual comparison — bar chart of key metrics metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'R
Cell 89 (markdown): ### 10.3 Confusion Matrices
Cell 90 (code): num_models = len(results) n_cols = 4 n_rows = (num_models + n_cols - 1) // n_cols 
Cell 91 (markdown): ### 10.4 ROC Curves
Cell 92 (code): plt.figure(figsize=(10, 7)) colors = sns.color_palette('husl', len(results))  
Cell 93 (markdown): ### 10.5 Classification Reports
Cell 94 (code): for name, r in results.items():     print(f'\n{"="*60}')     print(f'{name}') 
Cell 95 (markdown): ### 10.6 Cross-Validation Scores (5-Fold)
Cell 96 (code): import os import joblib from sklearn.model_selection import cross_val_score 
Cell 97 (code): # 9.6 Statistical Significance Testing # Perform a paired t-test to check if the performance difference between the top 
Cell 98 (code): # Boxplot of cross-validation scores fig, ax = plt.subplots(figsize=(12, 5)) cv_df = pd.DataFrame(cv_results) 
Cell 99 (markdown): ### 10.7 Learning Curves — Top 3 Models
Cell 100 (code): # Identify top 3 models by test accuracy top3 = comparison.head(3).index.tolist() print(f'Top 3 models for learning curv
Cell 101 (markdown): --- ## 🔧 Section 10: Hyperparameter Tuning  We apply `RandomizedSearchCV` to the **top 3 performing models** (dynamicall
Cell 102 (code): from sklearn.model_selection import RandomizedSearchCV from scipy.stats import randint, uniform  
Cell 103 (markdown): ### 10.8 Differential Privacy 
Cell 104 (markdown): ## Flex 10: 🧊 Differential Privacy Training Given that dating app data is inherently sensitive (sexual orientation, rela
Cell 105 (code): import os, joblib os.makedirs('models_v4_cache', exist_ok=True) cache_dp = 'models_v4_cache/opacus.joblib' 
Cell 106 (markdown): ---  ## 🏆 Pick Your Flexes — Recommended Combinations  | If you want... | Pick these | Report sections | |---|---|---| |
Cell 107 (markdown): ### 10.9 Graph Neural Network (Node Classification) 
Cell 108 (markdown): ## Flex 7: 🕸️ Graph Neural Network — Users as a Social Network We constructed a k-nearest-neighbor similarity graph over
Cell 109 (code): # pip install torch-geometric import torch import torch.nn.functional as F 
Cell 110 (markdown): --- 
Cell 111 (markdown): ### 10.10 Self-Supervised Contrastive Learning (SCARF) 
Cell 112 (markdown): ## Flex 3: 🧪 Self-Supervised Contrastive Pre-Training (SCARF) We implemented SCARF, a self-supervised contrastive pre-tr
Cell 113 (code): import os, joblib os.makedirs('models_v4_cache', exist_ok=True) cache_scarf = 'models_v4_cache/scarf.joblib' 
Cell 114 (markdown): --- 
Cell 115 (markdown): ## 🎛️ Section 11: Hyperparameter Optimization GPU-accelerated randomized search and multi-objective Pareto optimization.
Cell 116 (markdown): ### 11.1 Define Search Spaces
Cell 117 (code): # Define parameter grids for top models param_grids = {     'Random Forest': { 
Cell 118 (markdown): ### 11.2 Run Hyperparameter Search (Top 3 Models)
Cell 119 (code): # --- GPU-ACCELERATED OPTUNA HYPERPARAMETER SEARCH ENGINE --- import optuna import logging 
Cell 120 (markdown): ### 11.3 Before vs After Tuning Comparison
Cell 121 (code): # Compare baseline vs tuned for the top 3 print(f'{"Model":<25} {"Metric":<12} {"Baseline":>10} {"Tuned":>10} {"Change":
Cell 122 (code): # Visual comparison — baseline vs tuned fig, axes = plt.subplots(1, 3, figsize=(16, 5))  
Cell 123 (markdown): ### 11.4 Best Tuned Model — Detailed Results
Cell 124 (code): # Select best overall model if tuned_results:     best_name = max(tuned_results, key=lambda n: tuned_results[n]['f1']) 
Cell 125 (code): # Confusion matrix for best model if best:     suffix = ' (Tuned)' if tuned_results else ' (Baseline)' 
Cell 126 (markdown): --- ## 📊 Section 11: Feature Importance Analysis
Cell 127 (code): # Feature importance from the best tree-based model # Try: Random Forest, XGBoost, or Decision Tree importance_model = N
Cell 128 (code): --- ## ⚖️ Section 12: Ethical Considerations in Dating App ML  
Cell 129 (markdown): --- ## ⚖️ Ethical Considerations in Dating App ML  Machine learning models deployed in human-centric domains like dating
Cell 130 (code): # Per-group accuracy breakdown (Testing for Demographic Parity) print('Accuracy Breakdown by Gender:') print('-' * 40) 
Cell 131 (markdown): --- ## 🏆 Section 12: Final Model Summary
Cell 132 (code): # Final comprehensive comparison: all baseline + all tuned print('=' * 80) print('FINAL MODEL COMPARISON') 
Cell 133 (code): # Final bar chart — all models ranked by F1 plt.figure(figsize=(14, 7)) colors_final = ['#4CAF50' if 'Tuned' in name els
Cell 134 (markdown): --- ## 🤖 Section 14: AutoML Comparison (FLAML & PyCaret) 
Cell 135 (code): from flaml import AutoML import sklearn.metrics import joblib 
Cell 136 (code): import pandas as pd try:     from pycaret.classification import setup, compare_models, pull 
Cell 137 (markdown): --- ## ✅ Section 15: Final Pipeline Summary & Hardware Optimisations  ### 🏆 Key Findings & Accomplishments:  1. **All 14
Cell 138 (markdown): ## 🔍 Section 12: Feature Importance & Interaction Analysis Extracting tree-based importance and calculating pairwise per
Cell 139 (markdown): ### 12.1 Permutation Feature Interaction (H-Statistic) 
Cell 140 (markdown): ## Flex 9: 🔬 Permutation Feature Interaction Detection We computed Friedman's H-statistic to quantify second-order featu
Cell 141 (code): from sklearn.inspection import PartialDependenceDisplay, partial_dependence import itertools import numpy as np 
Cell 142 (markdown): --- 
Cell 143 (markdown): ## 🛡️ Section 13: Advanced Model Robustness & Uncertainty Ensuring prediction reliability via conformal bounding, Bayesi
Cell 144 (markdown): ### 13.1 Conformal Prediction 
Cell 145 (markdown): ## Flex 1: 🎯 Conformal Prediction — Guaranteed Uncertainty Bands Rather than outputting point predictions, we implemente
Cell 146 (code): import os, joblib os.makedirs('models_v4_cache', exist_ok=True) cache_mapie = 'models_v4_cache/mapie.joblib' 
Cell 147 (markdown): > [!TIP] > **Report flex:** Include a table showing that your empirical coverage matches the theoretical guarantee (e.g.
Cell 148 (markdown): ### 13.2 Bayesian Uncertainty (MC Dropout) 
Cell 149 (markdown): ## Flex 6: 🌊 Bayesian Uncertainty Quantification (MC Dropout) We implemented Monte Carlo Dropout as an approximate Bayes
Cell 150 (code): import torch import torch.nn as nn import numpy as np 
Cell 151 (markdown): --- 
Cell 152 (markdown): ### 13.3 Adversarial Robustness (FGSM) 
Cell 153 (markdown): ## Flex 5: ⚔️ Adversarial Robustness Testing We evaluated model robustness against adversarial perturbations using the F
Cell 154 (code): import torch import torch.nn as nn import numpy as np 
Cell 155 (markdown): --- 
Cell 156 (markdown): ## 🚀 Section 14: Model Compression & Deployment Strategies Distilling knowledge from complex ensembles into lightweight,
Cell 157 (markdown): ### 14.1 Knowledge Distillation 
Cell 158 (markdown): ## Flex 4: 🎓 Knowledge Distillation — Complex → Simple We applied Hinton-style knowledge distillation to compress the kn
Cell 159 (code): import torch import torch.nn as nn import torch.nn.functional as F 
Cell 160 (markdown): --- 
Cell 161 (markdown): ## ⚖️ Section 15: Ethical Considerations & Demographic Parity Evaluating the fairness and bias of the champion model acr
Cell 162 (markdown): ## 🏆 Section 16: Final Model Summary Comprehensive evaluation of all trained models, ranking by F1 score and generating 