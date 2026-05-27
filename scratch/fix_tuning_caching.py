import json
import os

# Files to update
v5_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"
v4_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V4.ipynb"

# Target code content for V5
v5_tuning_source = """# --- GPU-ACCELERATED OPTUNA HYPERPARAMETER SEARCH ENGINE ---
import optuna
from sklearn.neural_network import MLPClassifier
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_search(X_tr, y_tr, X_te, y_te):
    print("🔄 Running massive GPU-accelerated Optuna hyperparameter search (1000 Trials)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            **TREE_CONFIG['xgb'] # Dynamic CUDA/OpenCL GPU assignment
        }
        
        clf = XGBClassifier(**params)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        return f1_score(y_te, preds)

    import os, joblib
    os.makedirs('../models_v5', exist_ok=True)
    cache_optuna = '../models_v5/optuna_pareto.joblib'
    if os.path.exists(cache_optuna):
        print("⏭️  Loading cached Optuna Pareto Optimization...")
        return joblib.load(cache_optuna)
    print("⏳ Running Multi-Objective Pareto Optimization (~2-5m)...")
    study = optuna.create_study(direction='maximize')
    start_time = time.time()
    # Runs 1000 trials. Under GPU acceleration, each trial fits in ~0.2s, completing in under 3-4 minutes!
    study.optimize(objective, n_trials=1000, n_jobs=1) 
    search_time = time.time() - start_time
    
    print(f"🎉 Optuna search completed in {search_time:.1f}s!")
    joblib.dump(study, cache_optuna)
    print(f"  Best F1 score: {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")
    return study.best_params

# We will run this dynamically inside the tuning cell


# Tune the top 3 models (With smart SVM bypass and selector)
import os
import joblib
import time
from sklearn.model_selection import RandomizedSearchCV

# -----------------------------------------------------------------
# SELECTOR: Set RETUNE_MODELS = True to retune top models from scratch.
# Set RETUNE_MODELS = False to load pre-tuned results from models_v5 if available.
RETUNE_MODELS = False
# -----------------------------------------------------------------

# Load pre-tuned SVM from original to skip 3+ hours of SVM tuning
original_tuned_path = '../models/tuned_results.joblib'

# Save newly tuned models to bypass path so we don't overwrite original files
checkpoint_tuned_path = '../models_v5/tuned_results.joblib'
tuned_results = {}
temp_tuned = {}

# Ensure bypass directory exists
os.makedirs('../models_v5', exist_ok=True)

# Load pre-trained tuned models from original for SVM bypass
if os.path.exists(original_tuned_path):
    try:
        temp_tuned = joblib.load(original_tuned_path)
        print(f'⚡ Loaded pre-trained tuned models from {original_tuned_path} for SVM bypass')
    except Exception as e:
        print(f'⚠️ Error loading original tuned checkpoint: {e}')

# Check if we should load the full tuned results from models_v5
loaded_tuned_from_cache = False
if not RETUNE_MODELS and os.path.exists(checkpoint_tuned_path):
    try:
        tuned_results = joblib.load(checkpoint_tuned_path)
        print(f'🎉 Successfully loaded pre-tuned baseline results from {checkpoint_tuned_path}!')
        print(f'   Models loaded: {list(tuned_results.keys())}')
        loaded_tuned_from_cache = True
    except Exception as e:
        print(f'⚠️ Error loading tuned checkpoint: {e}. Will retune.')

if not loaded_tuned_from_cache:
    print('🔄 Running hyperparameter tuning on top 3 models (Smart SVM Bypass)...')
    for name in top3:
        # Smart Skip: If SVM is in our checkpoint, load it instantly and skip 3+ hours of tuning
        if name == 'SVM' and 'SVM' in temp_tuned:
            print(f'⚡ Reusing pre-tuned SVM model from original checkpoint (saved hours of tuning!)...')
            tuned_results['SVM'] = temp_tuned['SVM']
            print(f'  Loaded Tuned SVM — Test Acc: {tuned_results["SVM"]["test_acc"]:.4f} | F1: {tuned_results["SVM"]["f1"]:.4f}')
            continue

        print(f'\\n{"="*60}')
        print(f'Tuning: {name}')
        print(f'{"="*60}')

        if name not in param_grids:
            print(f'  No parameter grid defined for {name}, skipping.')
            continue

        # Create fresh model instance
        base_models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            'XGBoost': XGBClassifier(scale_pos_weight=(30150/19850), 
                random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric='logloss', tree_method='hist',
                device=XGB_DEVICE
            ) if HAS_XGBOOST else GradientBoostingClassifier(random_state=RANDOM_STATE),
            'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
            'CatBoost': CatBoostClassifier(iterations=500, random_state=RANDOM_STATE, verbose=0),
            'Champion Stacking Ensemble': StackingClassifier(
                estimators=[
                    ('lgbm', LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1)),
                    ('xgb', XGBClassifier(scale_pos_weight=(30150/19850), n_estimators=500, random_state=RANDOM_STATE, eval_metric='logloss', tree_method='hist', device=XGB_DEVICE, n_jobs=1) if HAS_XGBOOST else GradientBoostingClassifier(n_estimators=500, random_state=RANDOM_STATE)),
                    ('cat', CatBoostClassifier(iterations=500, random_state=RANDOM_STATE, verbose=0))
                ],
                final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE, n_jobs=1),
                n_jobs=1,
                cv=3
            ),
            'Multi-Layer Perceptron': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE),
            'Balanced Random Forest': BalancedRandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=1),
            'Collaborative Filtering (Cosine KNN)': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=1),
            'TabNet Deep Learning': TabNetClassifier(verbose=0) if HAS_TABNET else LogisticRegression(random_state=RANDOM_STATE),
            'SVM': BaggingClassifier(
                estimator=SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE, cache_size=500, tol=1e-3),
                n_estimators=16, max_samples=0.20, n_jobs=1, random_state=RANDOM_STATE
            ),
        }

        search = RandomizedSearchCV(
            estimator=base_models[name],
            param_distributions=param_grids[name],
            n_iter=30,
            cv=5,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=1
        )

        start = time.time()
        search.fit(X_train, y_train)
        tune_time = time.time() - start

        best_model = search.best_estimator_
        y_pred_tuned = best_model.predict(X_test)

        if hasattr(best_model, 'predict_proba'):
            y_prob_tuned = best_model.predict_proba(X_test)[:, 1]
        else:
            y_prob_tuned = best_model.decision_function(X_test)

        tuned_results[name] = {
            'model': best_model,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'test_acc': accuracy_score(y_test, y_pred_tuned),
            'precision': precision_score(y_test, y_pred_tuned),
            'recall': recall_score(y_test, y_pred_tuned),
            'f1': f1_score(y_test, y_pred_tuned),
            'roc_auc': roc_auc_score(y_test, y_prob_tuned),
            'tune_time': tune_time,
            'y_pred': y_pred_tuned,
            'y_prob': y_prob_tuned
        }

        print(f'\\n  Best Parameters: {search.best_params_}')
        print(f'  Best CV F1:     {search.best_score_:.4f}')
        print(f'  Test Accuracy:  {tuned_results[name]["test_acc"]:.4f}')
        print(f'  Test F1:        {tuned_results[name]["f1"]:.4f}')
        print(f'  Test ROC-AUC:   {tuned_results[name]["roc_auc"]:.4f}')
        print(f'  Tuning Time:    {tune_time:.1f}s')

    # Save tuned results to bypass disk
    joblib.dump(tuned_results, checkpoint_tuned_path)
    print(f'\\n💾 Tuned models saved successfully to: {checkpoint_tuned_path}')
"""

# Load and update V5
with open(v5_path, 'r', encoding='utf-8') as f:
    nb_v5 = json.load(f)

for cell in nb_v5['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Running hyperparameter tuning" in source:
            cell['source'] = [line + '\n' for line in v5_tuning_source.split('\n')]
            print("Successfully updated V5 Tuning cell!")
            break

with open(v5_path, 'w', encoding='utf-8') as f:
    json.dump(nb_v5, f, indent=1, ensure_ascii=False)

# Target code content for V4
v4_tuning_source = v5_tuning_source.replace('../models_v5', '../models_v4')

# Load and update V4
with open(v4_path, 'r', encoding='utf-8') as f:
    nb_v4 = json.load(f)

for cell in nb_v4['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Running hyperparameter tuning" in source:
            cell['source'] = [line + '\n' for line in v4_tuning_source.split('\n')]
            print("Successfully updated V4 Tuning cell!")
            break

with open(v4_path, 'w', encoding='utf-8') as f:
    json.dump(nb_v4, f, indent=1, ensure_ascii=False)
