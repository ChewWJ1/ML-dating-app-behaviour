# Jupyter Notebook Comparison Report

## File Summary
| Metric | JR1 Notebook (Scratch) | Main Notebook (Notebooks) |
| --- | --- | --- |
| Total Cells | 114 | 114 |
| Code Cells | 62 | 62 |
| Markdown Cells | 52 | 52 |
| Code Cell Discrepancies | 9 | 9 |

## Section Outline Comparison
| JR1 Headers | Main Headers |
| --- | --- |
| # 💘 Tying the Data Knot: Predicting Meaningful Connections | # 💘 Tying the Data Knot: Predicting Meaningful Connections |
| ### WIA1006/WID3006 Machine Learning — Group Assignment | ### WIA1006/WID3006 Machine Learning — Group Assignment |
| ## 📦 Section 1: Install & Import Libraries | ## 📦 Section 1: Install & Import Libraries |
| ## 📂 Section 2: Data Loading | ## 📂 Section 2: Data Loading |
| ## 🔍 Section 3: Exploratory Data Analysis (EDA) | ## 🔍 Section 3: Exploratory Data Analysis (EDA) |
| ### 3.1 Basic Info & Statistics | ### 3.1 Basic Info & Statistics |
| ### 3.2 Missing Values & Duplicates | ### 3.2 Missing Values & Duplicates |
| ### 3.3 Target Variable — match_outcome | ### 3.3 Target Variable — match_outcome |
| ### 3.4 Categorical Feature Distributions | ### 3.4 Categorical Feature Distributions |
| ### 3.5 Numerical Feature Distributions | ### 3.5 Numerical Feature Distributions |
| ### 3.6 Numerical Features — Outlier Detection (Boxplots) | ### 3.6 Numerical Features — Outlier Detection (Boxplots) |
| ### 3.7 Feature vs Target — Numerical Features by Outcome | ### 3.7 Feature vs Target — Numerical Features by Outcome |
| ### 3.8 Feature vs Target — Categorical Features by Outcome | ### 3.8 Feature vs Target — Categorical Features by Outcome |
| ### 3.9 Correlation Heatmap (Numerical Features) | ### 3.9 Correlation Heatmap (Numerical Features) |
| ### 3.10 Interest Tags Analysis | ### 3.10 Interest Tags Analysis |
| ## 🧹 Section 4: Data Preprocessing | ## 🧹 Section 4: Data Preprocessing |
| ### 4.1 Create Working Copy & Drop Redundant Columns | ### 4.1 Create Working Copy & Drop Redundant Columns |
| ### 4.2 Create Binary Target Variable | ### 4.2 Create Binary Target Variable |
| ### 4.3 Encode Ordinal Feature — income_bracket (7 levels → 3 tiers) | ### 4.3 Encode Ordinal Feature — income_bracket (7 levels → 3 tiers) |
| ### 4.4 Encode Ordinal Feature — education_level (9 levels → 3 tiers) | ### 4.4 Encode Ordinal Feature — education_level (9 levels → 3 tiers) |
| ### 4.5 One-Hot Encode Nominal Categorical Features | ### 4.5 One-Hot Encode Nominal Categorical Features |
| ### 4.6 Multi-Hot Encode Interest Tags | ### 4.6 Multi-Hot Encode Interest Tags |
| ### 4.7 Normalize Numerical Features with StandardScaler | ### 4.7 Normalize Numerical Features with StandardScaler |
| ### 4.8 Final Preprocessed Dataset Overview | ### 4.8 Final Preprocessed Dataset Overview |
| ## 🎯 Section 5: Feature Selection | ## 🎯 Section 5: Feature Selection |
| ### 5.1 Prepare Feature Matrix & Target Vector | ### 5.1 Prepare Feature Matrix & Target Vector |
| ### 5.2 ANOVA F-Score Feature Selection (SelectKBest) | ### 5.2 ANOVA F-Score Feature Selection (SelectKBest) |
| ### 5.3 Mutual Information Feature Selection | ### 5.3 Mutual Information Feature Selection |
| ### 5.4 Select Final Feature Set | ### 5.4 Select Final Feature Set |
| ## 📐 Section 6: Dimensionality Reduction — PCA | ## 📐 Section 6: Dimensionality Reduction — PCA |
| ### 6.1 Explained Variance Analysis | ### 6.1 Explained Variance Analysis |
| ### 6.2 Apply PCA (retain 95% explained variance) | ### 6.2 Apply PCA (retain 95% explained variance) |
| ### 6.3 PCA Biplot — First Two Principal Components | ### 6.3 PCA Biplot — First Two Principal Components |
| ## ✂️ Section 7: Train / Test Split | ## ✂️ Section 7: Train / Test Split |
| ## ✅ Section 8: Pre-Training Checklist | ## ✅ Section 8: Pre-Training Checklist |
| ### Objects available for model training: | ### Objects available for model training: |
| ## 🤖 Section 9: Model Training | ## 🤖 Section 9: Model Training |
| ### 9.1 Define & Train All Models | ### 9.1 Define & Train All Models |
| ### 9.2 Model Comparison Table | ### 9.2 Model Comparison Table |
| ### 9.3 Confusion Matrices | ### 9.3 Confusion Matrices |
| ### 9.4 ROC Curves | ### 9.4 ROC Curves |
| ### 9.5 Classification Reports | ### 9.5 Classification Reports |
| ### 9.6 Cross-Validation Scores (5-Fold) | ### 9.6 Cross-Validation Scores (5-Fold) |
| ### 9.7 Learning Curves — Top 3 Models | ### 9.7 Learning Curves — Top 3 Models |
| ## 🔧 Section 10: Hyperparameter Tuning | ## 🔧 Section 10: Hyperparameter Tuning |
| ### 10.1 Define Search Spaces | ### 10.1 Define Search Spaces |
| ### 10.2 Run Hyperparameter Search (Top 3 Models) | ### 10.2 Run Hyperparameter Search (Top 3 Models) |
| ### 10.3 Before vs After Tuning Comparison | ### 10.3 Before vs After Tuning Comparison |
| ### 10.4 Best Tuned Model — Detailed Results | ### 10.4 Best Tuned Model — Detailed Results |
| ## 📊 Section 11: Feature Importance Analysis | ## 📊 Section 11: Feature Importance Analysis |
| ## ⚖️ Ethical Considerations in Dating App ML | ## ⚖️ Ethical Considerations in Dating App ML |
| ## 🏆 Section 12: Final Model Summary | ## 🏆 Section 12: Final Model Summary |
| ## AutoML Comparison: FLAML and PyCaret | ## AutoML Comparison: FLAML and PyCaret |
| ## ✅ Final Pipeline Summary & Hardware Optimisations | ## ✅ Final Pipeline Summary & Hardware Optimisations |
| ### 🏆 Key Findings & Accomplishments: | ### 🏆 Key Findings & Accomplishments: |
| ### ⚡ Hardware Acceleration & Speed Optimisations: | ### ⚡ Hardware Acceleration & Speed Optimisations: |
| ### 💾 Smart Checkpointing & Caching: | ### 💾 Smart Checkpointing & Caching: |

## Code Cell Differences
Found 9 cells with differing code:

### Code Cell 1
**JR1 (Scratch):**
```python
!pip install flaml pycaret shap
# Restart runtime after installation if you face any issues, though FLAML usually doesn't require it.
```
**Main (Notebooks):**
```python
!pip install flaml shap seaborn
# Pycaret has been removed to avoid pandas downgrade issues on Python 3.14
# Restart runtime after installation if you face any issues, though FLAML usually doesn't require it.
```

### Code Cell 3
**JR1 (Scratch):**
```python
# ------------------------------------------------------------------
# Dataset Path Setup (Local)
# ------------------------------------------------------------------
DATA_PATH = 'dating_app_behavior_dataset_extended1.csv'

df_raw = pd.read_csv(DATA_PATH)
print(f'Dataset loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns')
df_raw.head()
```
**Main (Notebooks):**
```python
# ------------------------------------------------------------------
# Dataset Path Setup (Local)
# ------------------------------------------------------------------
DATA_PATH = '../data/dating_app_behavior_dataset_extended1.csv'

df_raw = pd.read_csv(DATA_PATH)
print(f'Dataset loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns')
df_raw.head()
```

### Code Cell 37
**JR1 (Scratch):**
```python
# Define all models
# Robust self-contained fallback definitions for XGBoost in case cells are run out of order
if 'XGB_DEVICE' not in globals():
    XGB_DEVICE = 'cpu'
if 'HAS_XGBOOST' not in globals():
    try:
        from xgboost import XGBClassifier
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False
if 'RANDOM_STATE' not in globals():
    RANDOM_STATE = 42

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', 
        max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs', n_jobs=-1
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1
    ),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', 
        random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(class_weight='balanced', 
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(scale_pos_weight=(30150/19850), 
    n_estimators=500,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    tree_method='hist',
    device=XGB_DEVICE,
    n_jobs=-1
    ) if HAS_XGBOOST else GradientBoostingClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE
    ),
    'SVM': BaggingClassifier(
        estimator=SVC(class_weight='balanced', 
            kernel='rbf', probability=True, random_state=RANDOM_STATE,
            cache_size=1000, tol=1e-3
        ),
        n_estimators=16,   # Runs 16 models in parallel (1 per thread)
        max_samples=0.20,  # Each thread gets 20% of dataset (~8,000 samples) for higher accuracy
        n_jobs=-1,         # Maxes out all 16 CPU threads
        random_state=RANDOM_STATE
    ),
}

print(f'Models defined: {list(models.keys())}')
```
**Main (Notebooks):**
```python
import os
num_threads = os.cpu_count() or 1
# Define all models
# Robust self-contained fallback definitions for XGBoost in case cells are run out of order
if 'XGB_DEVICE' not in globals():
    XGB_DEVICE = 'cpu'
if 'HAS_XGBOOST' not in globals():
    try:
        from xgboost import XGBClassifier
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False
if 'RANDOM_STATE' not in globals():
    RANDOM_STATE = 42

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', 
        max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs', n_jobs=-1
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1
    ),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', 
        random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(class_weight='balanced', 
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(scale_pos_weight=(30150/19850), 
    n_estimators=500,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    tree_method='hist',
    device=XGB_DEVICE,
    n_jobs=-1
    ) if HAS_XGBOOST else GradientBoostingClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE
    ),
    'SVM': BaggingClassifier(
        estimator=SVC(class_weight='balanced', 
            kernel='rbf', probability=True, random_state=RANDOM_STATE,
            cache_size=1000, tol=1e-3
        ),
        n_estimators=num_threads,   # Runs 1 model per logical thread dynamically
        max_samples=0.20,  # Each thread gets 20% of dataset (~8,000 samples) for higher accuracy
        n_jobs=-1,         # Maxes out all CPU threads dynamically
        random_state=RANDOM_STATE
    ),
}

print(f'Models defined: {list(models.keys())}')
```

### Code Cell 38
**JR1 (Scratch):**
```python
# Train all models and collect results (With auto-save and auto-load to avoid repeating hours of training)
import joblib
import os

checkpoint_path = 'baseline_results.joblib'
results = {}

# Robust verification: Ensure the checkpoint exists, is not empty, and contains all models we defined
if os.path.exists(checkpoint_path):
    try:
        temp_results = joblib.load(checkpoint_path)
        # Check that it contains all the keys in our models dictionary
        if temp_results and isinstance(temp_results, dict) and all(m in temp_results for m in models.keys()):
            results = temp_results
            print(f'⚡ Successfully loaded all pre-trained baseline models and metrics from {checkpoint_path}!...')
            # Print loaded results
            for name, r in results.items():
                print(f'  Loaded {name} — Test Acc: {r["test_acc"]:.4f} | F1: {r["f1"]:.4f}')
        else:
            print('⚠️ Checkpoint file is incomplete or empty. Discarding and retraining all models...')
    except Exception as e:
        print(f'⚠️ Error loading checkpoint: {e}. Discarding and retraining all models...')

if not results:
    print('🔄 Training all 6 baseline models (this might take a while)...')
    for name, model in models.items():
        print(f'\n{"="*60}')
        print(f'Training: {name}')
        print(f'{"="*60}')

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Probability predictions (for ROC-AUC)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_prob)

        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'train_time': train_time,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        print(f'  Train Acc: {train_acc:.4f}  |  Test Acc: {test_acc:.4f}')
        print(f'  Precision: {precision:.4f}  |  Recall:   {recall:.4f}')
        print(f'  F1 Score:  {f1:.4f}       |  ROC-AUC:  {roc_auc:.4f}')
        print(f'  Train Time: {train_time:.2f}s')

    # Auto-save results to disk
    joblib.dump(results, checkpoint_path)
    print(f'\n💾 Pre-trained baseline models saved successfully to: {checkpoint_path}')

print(f'\n{"="*60}')
print('All baseline models loaded/trained successfully!')
print(f'{"="*60}')
```
**Main (Notebooks):**
```python
# Train all models and collect results (With auto-save and auto-load to avoid repeating hours of training)
import joblib
import os

checkpoint_path = '../models/baseline_results.joblib'
results = {}

# Robust verification: Ensure the checkpoint exists, is not empty, and contains all models we defined
if os.path.exists(checkpoint_path):
    try:
        temp_results = joblib.load(checkpoint_path)
        # Check that it contains all the keys in our models dictionary
        if temp_results and isinstance(temp_results, dict) and all(m in temp_results for m in models.keys()):
            results = temp_results
            print(f'⚡ Successfully loaded all pre-trained baseline models and metrics from {checkpoint_path}!...')
            # Print loaded results
            for name, r in results.items():
                print(f'  Loaded {name} — Test Acc: {r["test_acc"]:.4f} | F1: {r["f1"]:.4f}')
        else:
            print('⚠️ Checkpoint file is incomplete or empty. Discarding and retraining all models...')
    except Exception as e:
        print(f'⚠️ Error loading checkpoint: {e}. Discarding and retraining all models...')

if not results:
    print('🔄 Training all 6 baseline models (this might take a while)...')
    for name, model in models.items():
        print(f'\n{"="*60}')
        print(f'Training: {name}')
        print(f'{"="*60}')

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Probability predictions (for ROC-AUC)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_prob)

        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'train_time': train_time,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        print(f'  Train Acc: {train_acc:.4f}  |  Test Acc: {test_acc:.4f}')
        print(f'  Precision: {precision:.4f}  |  Recall:   {recall:.4f}')
        print(f'  F1 Score:  {f1:.4f}       |  ROC-AUC:  {roc_auc:.4f}')
        print(f'  Train Time: {train_time:.2f}s')

    # Auto-save results to disk
    joblib.dump(results, checkpoint_path)
    print(f'\n💾 Pre-trained baseline models saved successfully to: {checkpoint_path}')

print(f'\n{"="*60}')
print('All baseline models loaded/trained successfully!')
print(f'{"="*60}')
```

### Code Cell 44
**JR1 (Scratch):**
```python
import os
import joblib

cv_checkpoint_path = 'cv_results.joblib'
cv_results = {}

print('5-Fold Cross-Validation on Training Set:')
print(f'{"":<25} {"Mean Acc":>10} {"Std":>8} {"Min":>8} {"Max":>8}')
print('-' * 65)

if os.path.exists(cv_checkpoint_path):
    print(f'⚡ Loading pre-computed cross-validation scores from {cv_checkpoint_path} instantly!...')
    cv_results = joblib.load(cv_checkpoint_path)
    for name, cv_scores in cv_results.items():
        print(f'{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>8.4f} '
              f'{cv_scores.min():>8.4f} {cv_scores.max():>8.4f}')
else:
    print('🔄 Computing cross-validation scores for all models (using 16 threads)...')
    for name, model_info in results.items():
        model = model_info['model']
        
        # Tweak: Temporarily set inner model n_jobs=1 during cross_val_score 
        # to avoid thread oversubscription deadlock with the outer parallel cross_val_score(n_jobs=-1)
        original_n_jobs = getattr(model, 'n_jobs', None)
        if original_n_jobs is not None:
            model.n_jobs = 1
            
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='accuracy', n_jobs=-1)
        
        # Restore original n_jobs
        if original_n_jobs is not None:
            model.n_jobs = original_n_jobs
            
        cv_results[name] = cv_scores
        print(f'{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>8.4f} '
              f'{cv_scores.min():>8.4f} {cv_scores.max():>8.4f}')
              
    # Save scores to disk
    joblib.dump(cv_results, cv_checkpoint_path)
    print(f'\n💾 Cross-validation scores saved successfully to: {cv_checkpoint_path}')
```
**Main (Notebooks):**
```python
import os
import joblib

cv_checkpoint_path = '../models/cv_results.joblib'
cv_results = {}

print('5-Fold Cross-Validation on Training Set:')
print(f'{"":<25} {"Mean Acc":>10} {"Std":>8} {"Min":>8} {"Max":>8}')
print('-' * 65)

if os.path.exists(cv_checkpoint_path):
    print(f'⚡ Loading pre-computed cross-validation scores from {cv_checkpoint_path} instantly!...')
    cv_results = joblib.load(cv_checkpoint_path)
    for name, cv_scores in cv_results.items():
        print(f'{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>8.4f} '
              f'{cv_scores.min():>8.4f} {cv_scores.max():>8.4f}')
else:
    print('🔄 Computing cross-validation scores for all models (using 16 threads)...')
    for name, model_info in results.items():
        model = model_info['model']
        
        # Tweak: Temporarily set inner model n_jobs=1 during cross_val_score 
        # to avoid thread oversubscription deadlock with the outer parallel cross_val_score(n_jobs=-1)
        original_n_jobs = getattr(model, 'n_jobs', None)
        if original_n_jobs is not None:
            model.n_jobs = 1
            
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='accuracy', n_jobs=-1)
        
        # Restore original n_jobs
        if original_n_jobs is not None:
            model.n_jobs = original_n_jobs
            
        cv_results[name] = cv_scores
        print(f'{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>8.4f} '
              f'{cv_scores.min():>8.4f} {cv_scores.max():>8.4f}')
              
    # Save scores to disk
    joblib.dump(cv_results, cv_checkpoint_path)
    print(f'\n💾 Cross-validation scores saved successfully to: {cv_checkpoint_path}')
```

### Code Cell 47
**JR1 (Scratch):**
```python
# Identify top 3 models by test accuracy
top3 = comparison.head(3).index.tolist()
print(f'Top 3 models for learning curves: {top3}')

import joblib
import os

lc_checkpoint_path = 'learning_curve_data.joblib'
lc_data = {}

if os.path.exists(lc_checkpoint_path):
    print(f'⚡ Loading computed learning curve data from {lc_checkpoint_path} instantly!...')
    lc_data = joblib.load(lc_checkpoint_path)
else:
    print('🔄 Learning curve data not found. Running cross-validation across sizes (this might take a while)...')
    for name in top3:
        model = results[name]['model']
        print(f'  Computing learning curve for: {name}')
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=5, scoring='accuracy', n_jobs=-1
        )
        lc_data[name] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores
        }
    # Save computed learning curves to disk
    joblib.dump(lc_data, lc_checkpoint_path)
    print(f'💾 Computed learning curve data saved successfully to: {lc_checkpoint_path}')

# Plot learning curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(top3):
    data = lc_data[name]
    train_sizes = data['train_sizes']
    train_scores = data['train_scores']
    val_scores = data['val_scores']

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#4CAF50')
    axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#F44336')
    axes[i].plot(train_sizes, train_mean, 'o-', color='#4CAF50', label='Training', linewidth=2)
    axes[i].plot(train_sizes, val_mean, 'o-', color='#F44336', label='Validation', linewidth=2)
    axes[i].set_title(name, fontweight='bold')
    axes[i].set_xlabel('Training Set Size')
    axes[i].set_ylabel('Accuracy')
    axes[i].legend(loc='lower right', fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Learning Curves — Top 3 Models', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
```
**Main (Notebooks):**
```python
# Identify top 3 models by test accuracy
top3 = comparison.head(3).index.tolist()
print(f'Top 3 models for learning curves: {top3}')

import joblib
import os

lc_checkpoint_path = '../models/learning_curve_data.joblib'
lc_data = {}

if os.path.exists(lc_checkpoint_path):
    print(f'⚡ Loading computed learning curve data from {lc_checkpoint_path} instantly!...')
    lc_data = joblib.load(lc_checkpoint_path)
else:
    print('🔄 Learning curve data not found. Running cross-validation across sizes (this might take a while)...')
    for name in top3:
        model = results[name]['model']
        print(f'  Computing learning curve for: {name}')
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=5, scoring='accuracy', n_jobs=-1
        )
        lc_data[name] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores
        }
    # Save computed learning curves to disk
    joblib.dump(lc_data, lc_checkpoint_path)
    print(f'💾 Computed learning curve data saved successfully to: {lc_checkpoint_path}')

# Plot learning curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(top3):
    data = lc_data[name]
    train_sizes = data['train_sizes']
    train_scores = data['train_scores']
    val_scores = data['val_scores']

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#4CAF50')
    axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#F44336')
    axes[i].plot(train_sizes, train_mean, 'o-', color='#4CAF50', label='Training', linewidth=2)
    axes[i].plot(train_sizes, val_mean, 'o-', color='#F44336', label='Validation', linewidth=2)
    axes[i].set_title(name, fontweight='bold')
    axes[i].set_xlabel('Training Set Size')
    axes[i].set_ylabel('Accuracy')
    axes[i].legend(loc='lower right', fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Learning Curves — Top 3 Models', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Code Cell 50
**JR1 (Scratch):**
```python
# Tune the top 3 models (With auto-save and auto-load to avoid repeating hours of tuning)
import os
import joblib
checkpoint_tuned_path = 'tuned_results.joblib'
tuned_results = {}

if os.path.exists(checkpoint_tuned_path):
    print(f'⚡ Loading pre-trained tuned models and metrics from {checkpoint_tuned_path} instantly!...')
    tuned_results = joblib.load(checkpoint_tuned_path)
    for name, r in tuned_results.items():
        print(f'  Loaded Tuned {name} — Test Acc: {r["test_acc"]:.4f} | F1: {r["f1"]:.4f}')
else:
    print('🔄 Tuned results not found. Running hyperparameter tuning on top 3 models (this might take a while)...')
    for name in top3:
        print(f'\n{"="*60}')
        print(f'Tuning: {name}')
        print(f'{"="*60}')

        if name not in param_grids:
            print(f'  No parameter grid defined for {name}, skipping.')
            continue

        # Create fresh model instance
        # Note: We do NOT use n_jobs=-1 inside base estimators here because RandomizedSearchCV
        # is already running in parallel with n_jobs=-1. This avoids CPU thread conflicts.
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
            'SVM': BaggingClassifier(
            estimator=SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE, cache_size=500, tol=1e-3),
            n_estimators=16, max_samples=0.20, n_jobs=-1, random_state=RANDOM_STATE
        ),
        }

        search = RandomizedSearchCV(
            estimator=base_models[name],
            param_distributions=param_grids[name],
            n_iter=30,
            cv=5,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=-1,
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

        print(f'\n  Best Parameters: {search.best_params_}')
        print(f'  Best CV F1:     {search.best_score_:.4f}')
        print(f'  Test Accuracy:  {tuned_results[name]["test_acc"]:.4f}')
        print(f'  Test F1:        {tuned_results[name]["f1"]:.4f}')
        print(f'  Test ROC-AUC:   {tuned_results[name]["roc_auc"]:.4f}')
        print(f'  Tuning Time:    {tune_time:.1f}s')

    # Auto-save tuned results to disk
    joblib.dump(tuned_results, checkpoint_tuned_path)
    print(f'\n💾 Pre-trained tuned models saved successfully to: {checkpoint_tuned_path}')

print(f'\n{"="*60}')
print('Hyperparameter tuning complete!')
print(f'{"="*60}')

```
**Main (Notebooks):**
```python
num_threads = os.cpu_count() or 1
# Tune the top 3 models (With auto-save and auto-load to avoid repeating hours of tuning)
import os
import joblib
checkpoint_tuned_path = '../models/tuned_results.joblib'
tuned_results = {}

if os.path.exists(checkpoint_tuned_path):
    print(f'⚡ Loading pre-trained tuned models and metrics from {checkpoint_tuned_path} instantly!...')
    tuned_results = joblib.load(checkpoint_tuned_path)
    for name, r in tuned_results.items():
        print(f'  Loaded Tuned {name} — Test Acc: {r["test_acc"]:.4f} | F1: {r["f1"]:.4f}')
else:
    print('🔄 Tuned results not found. Running hyperparameter tuning on top 3 models (this might take a while)...')
    for name in top3:
        print(f'\n{"="*60}')
        print(f'Tuning: {name}')
        print(f'{"="*60}')

        if name not in param_grids:
            print(f'  No parameter grid defined for {name}, skipping.')
            continue

        # Create fresh model instance
        # Note: We do NOT use n_jobs=-1 inside base estimators here because RandomizedSearchCV
        # is already running in parallel with n_jobs=-1. This avoids CPU thread conflicts.
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
            'SVM': BaggingClassifier(
            estimator=SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE, cache_size=500, tol=1e-3),
            n_estimators=16, max_samples=0.20, n_jobs=-1, random_state=RANDOM_STATE
        ),
        }

        search = RandomizedSearchCV(
            estimator=base_models[name],
            param_distributions=param_grids[name],
            n_iter=30,
            cv=5,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=-1,
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

        print(f'\n  Best Parameters: {search.best_params_}')
        print(f'  Best CV F1:     {search.best_score_:.4f}')
        print(f'  Test Accuracy:  {tuned_results[name]["test_acc"]:.4f}')
        print(f'  Test F1:        {tuned_results[name]["f1"]:.4f}')
        print(f'  Test ROC-AUC:   {tuned_results[name]["roc_auc"]:.4f}')
        print(f'  Tuning Time:    {tune_time:.1f}s')

    # Auto-save tuned results to disk
    joblib.dump(tuned_results, checkpoint_tuned_path)
    print(f'\n💾 Pre-trained tuned models saved successfully to: {checkpoint_tuned_path}')

print(f'\n{"="*60}')
print('Hyperparameter tuning complete!')
print(f'{"="*60}')

```

### Code Cell 60
**JR1 (Scratch):**
```python
from flaml import AutoML
import sklearn.metrics

try:
    print("--- Starting FLAML ---")
    automl = AutoML()
    
    # FLAML settings (120 seconds budget for demonstration)
    automl_settings = {
        "time_budget": 3600, 
        "metric": 'accuracy',
        "task": 'classification',
        "n_jobs": -1, # Force FLAML to use all 24 CPU threads
        "log_file_name": 'flaml.log',
        "verbose": 0 # Set to 3 for detailed logs
    }
    
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    
    print("\nBest FLAML model found:", automl.best_estimator)
    print("Best FLAML hyperparameters:", automl.best_config)
    
    # Evaluate
    flaml_predictions = automl.predict(X_test)
    flaml_accuracy = sklearn.metrics.accuracy_score(y_test, flaml_predictions)
    print(f"\nFLAML Test Accuracy: {flaml_accuracy:.4f}")
except Exception as e:
    print(f"FLAML Error: {e}")
```
**Main (Notebooks):**
```python
from flaml import AutoML
import sklearn.metrics
import joblib
import os

try:
    print("--- Starting FLAML ---")
    flaml_checkpoint_path = '../models/flaml_results.joblib'
    
    if os.path.exists(flaml_checkpoint_path):
        print("⚡ Loading pre-trained FLAML model from ../models/flaml_results.joblib instantly!...")
        automl = joblib.load(flaml_checkpoint_path)
    else:
        automl = AutoML()
        
        # FLAML settings (120 seconds budget for demonstration)
        automl_settings = {
            "time_budget": 3600, 
            "metric": 'accuracy',
            "task": 'classification',
            "n_jobs": -1, # Force FLAML to use all 24 CPU threads
            "log_file_name": 'flaml.log',
            "verbose": 0 # Set to 3 for detailed logs
        }
        
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        
        # Ensure models directory exists
        os.makedirs('../models', exist_ok=True)
        joblib.dump(automl, flaml_checkpoint_path)
    
    print("\nBest FLAML model found:", automl.best_estimator)
    print("Best FLAML hyperparameters:", automl.best_config)
    
    # Evaluate
    flaml_predictions = automl.predict(X_test)
    flaml_accuracy = sklearn.metrics.accuracy_score(y_test, flaml_predictions)
    print(f"\nFLAML Test Accuracy: {flaml_accuracy:.4f}")
except Exception as e:
    print(f"FLAML Error: {e}")
```

### Code Cell 61
**JR1 (Scratch):**
```python
import pandas as pd
from pycaret.classification import setup, compare_models, pull

try:
    print("\n--- Starting PyCaret ---")
    
    # PyCaret works best with pandas DataFrames containing both features and the target label
    if isinstance(X_train, pd.DataFrame):
        train_df = X_train.copy()
    else:
        # If X_train is a numpy array (e.g. from StandardScaler), convert it back to DF
        train_df = pd.DataFrame(X_train)
        
    # Add the target column
    train_df['target'] = list(y_train)
    
    # Initialize setup
    # We disable html so it prints nicely in the standard colab output instead of taking over the cell UI
    clf1 = setup(data=train_df, target='target', session_id=123, verbose=False, html=False, use_gpu=True)
    
    # Compare all standard models and return the best one
    print("Comparing models...")
    best_pycaret_model = compare_models() # No budget, run exhaustively
    
    print("\nBest PyCaret Model:", best_pycaret_model)
    
    # Show the results grid
    results_grid = pull()
    print("\nPyCaret Leaderboard:")
    display(results_grid.head(5))
    
except Exception as e:
    print(f"PyCaret Error: {e}")
```
**Main (Notebooks):**
```python
import pandas as pd
try:
    from pycaret.classification import setup, compare_models, pull
    print("\n--- Starting PyCaret ---")
    
    # PyCaret works best with pandas DataFrames containing both features and the target label
    if isinstance(X_train, pd.DataFrame):
        train_df = X_train.copy()
    else:
        # If X_train is a numpy array (e.g. from StandardScaler), convert it back to DF
        train_df = pd.DataFrame(X_train)
        
    # Add the target column
    train_df['target'] = list(y_train)
    
    # Initialize setup
    # We disable html so it prints nicely in the standard colab output instead of taking over the cell UI
    clf1 = setup(data=train_df, target='target', session_id=123, verbose=False, html=False, use_gpu=True)
    
    # Compare all standard models and return the best one
    print("Comparing models...")
    best_pycaret_model = compare_models() # No budget, run exhaustively
    
    print("\nBest PyCaret Model:", best_pycaret_model)
    
    # Show the results grid
    results_grid = pull()
    print("\nPyCaret Leaderboard:")
    display(results_grid.head(5))
    
except Exception as e:
    print(f"PyCaret Error: {e}")
```

## Evaluation Metrics / Outputs Comparison

### JR1 Metrics & Key Outputs

#### Cell 1 Code Snippet:
```python
!pip install flaml pycaret shap
# Restart runtime after installation if you face any issues, though FLAML usually doesn't require it.
```
**Output:**
```
error: subprocess-exited-with-error
  
  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 2
  ╰─> [106 lines of output]
      + meson setup C:\Users\User\AppData\Local\Temp\pip-install-lv7kq3m2\pandas_36ced063ec98477e837461cd3be1b215 C:\Users\User\AppData\Local\Temp\pip-install-lv7kq3m2\pandas_36ced063ec98477e837461cd3be1b215\.mesonpy-2hoamj67\build -Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md --vsenv --native-file=C:\Users\User\AppData\Local\Temp\pip-install-lv7kq3m2\pandas_36ced063ec98477e837461cd3be1b215\.mesonpy-2hoamj67\build\meson-python-native-file.ini
      The Meson build system
      Version: 1.2.1
      Source dir: C:\Users\User\AppData\Local\Temp\pip-install-lv7kq3m2\pandas_36ced063ec98477e837461cd3be1b215
      Build dir: C:\Users\User\AppData\Local\Temp\pip-install-lv7kq3m2\pandas_36ced063ec98477e837461cd3be1b215\.mesonpy-2hoamj67\build
      Build type: native build
      Project name: pandas
      Project version: 2.1.4
      Activating VS 18.4.2
      C compiler for the host machine: cl (msvc 19.50.35728 "Microsoft (R) C/C++ Optimizing Compiler Version 19.50.35728 for x64")
      C linker for the host machine: link link 14.50.35728.0
      C++ compiler for the host machine: cl (msvc 19.50.35728 "Microsoft (R) C/C++ Optimizing Compiler Version 19.50.35728 for x64")
      C++ linker for the host machine: link link 14.50.35728.0
      Cython compiler for the host machine: cython (cython 0.29.37)
      Host machine cpu family: x86_64
      Host machine cpu: x86_64
      Program python found: YES (C:\Users\User\OneDrive\Documents\ML-dating-app-behaviour\.venv\Scripts\python.exe)
      Run-time dependency python found: YES 3.14
      Build targets in project: 53
      
      pandas 2.1.4
      
        User defined options
          Native files: C:\Users\User\AppData\Local\Temp\pip-install-lv7kq3m2\pandas_36ced063ec98477e837461cd3be1b215\.mesonpy-2hoamj67\build\meson-python-native-file.ini
          buildtype   : release
          vsenv       : True
          b_ndebug    : if-release
          b_vscrt     : md
      
      Found ninja.EXE-1.13.0.git.kitware.jobserver-pipe-1 at C:\Users\User\AppData\Local\Temp\pip-build-env-4sjrfjlv\normal\Scripts\ninja.EXE
      
      Visual Studio environment is needed to run Ninja. It is recommended to use Meson wrapper:
      C:\Users\User\AppData\Local\Temp\pip-build-env-4sjrfjlv\overlay\Scripts\meson compile -C .
      + meson compile
      Activating VS 18.4.2
      INFO: automatically activated MSVC compiler environment
      INFO: autodetecting backend as ninja
      INFO: calculating backend command to run: C:\Users\User\AppData\Local\Temp\pip-build-env-4sjrfjlv\normal\Scripts\ninja.EXE
      [1/151] Generating pandas/_libs/khash_primitive_helper_pxi with a custom command
      [2/151] Generating pandas/_libs/index_class_helper_pxi with a custom command
      [3/151] Generating pandas/_libs/algos_take_helper_pxi with a custom command
      [4/151] Generating pandas/_libs/sparse_op_helper_pxi with a custom command
      [5/151] Generating pandas/_libs/hashtable_func_helper_pxi with a custom command
      [6/151] Generating pandas/_libs/algos_common_helper_pxi with a custom command
      [7/151] Generating pandas/_libs/intervaltree_helper_pxi with a custom command
      [8/151] Generating pandas/_libs/hashtable_class_helper_pxi with a custom command
      [9/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/indexing.pyx
      [10/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/base.pyx
      [11/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/ccalendar.pyx
      [12/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/dtypes.pyx
      [13/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/nattype.pyx
      [14/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/np_datetime.pyx
      [15/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/arrays.pyx
      [16/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/vectorized.pyx
      [17/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/hashing.pyx
      [18/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/timezones.pyx
      [19/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/fields.pyx
      [20/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/strptime.pyx
      [21/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/tzconversion.pyx
      [22/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/ops_dispatch.pyx
      [23/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/parsing.pyx
      [24/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/conversion.pyx
      [25/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/internals.pyx
      [26/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/properties.pyx
      [27/151] Compiling C object pandas/_libs/tslibs/base.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_base.pyx.c.obj
      [31mFAILED: [code=2] [0mpandas/_libs/tslibs/base.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_base.pyx.c.obj
      "cl" "-Ipandas\_libs\tslibs\base.cp314-win_amd64.pyd.p" "-Ipandas\_libs\tslibs" "-I..\..\pandas\_libs\tslibs" "-I..\..\..\..\pip-build-env-4sjrfjlv\overlay\Lib\site-packages\numpy\core\include" "-I..\..\pandas\_libs\include" "-IC:\Users\User\AppData\Roaming\uv\python\cpython-3.14-windows-x86_64-none\Include" "-DNDEBUG" "/MD" "/nologo" "/showIncludes" "/utf-8" "-w" "/O2" "/Gw" "-DNPY_NO_DEPRECATED_API=0" "-DNPY_TARGET_VERSION=NPY_1_21_API_VERSION" "/Fdpandas\_libs\tslibs\base.cp314-win_amd64.pyd.p\meson-generated_pandas__libs_tslibs_base.pyx.c.pdb" /Fopandas/_libs/tslibs/base.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_base.pyx.c.obj "/c" pandas/_libs/tslibs/base.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/base.pyx.c
      pandas/_libs/tslibs/base.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/base.pyx.c(5397): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      pandas/_libs/tslibs/base.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/base.pyx.c(5631): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      [28/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/missing.pyx
      [29/151] Compiling C object pandas/_libs/tslibs/ccalendar.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_ccalendar.pyx.c.obj
      [31mFAILED: [code=2] [0mpandas/_libs/tslibs/ccalendar.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_ccalendar.pyx.c.obj
      "cl" "-Ipandas\_libs\tslibs\ccalendar.cp314-win_amd64.pyd.p" "-Ipandas\_libs\tslibs" "-I..\..\pandas\_libs\tslibs" "-I..\..\..\..\pip-build-env-4sjrfjlv\overlay\Lib\site-packages\numpy\core\include" "-I..\..\pandas\_libs\include" "-IC:\Users\User\AppData\Roaming\uv\python\cpython-3.14-windows-x86_64-none\Include" "-DNDEBUG" "/MD" "/nologo" "/showIncludes" "/utf-8" "-w" "/O2" "/Gw" "-DNPY_NO_DEPRECATED_API=0" "-DNPY_TARGET_VERSION=NPY_1_21_API_VERSION" "/Fdpandas\_libs\tslibs\ccalendar.cp314-win_amd64.pyd.p\meson-generated_pandas__libs_tslibs_ccalendar.pyx.c.pdb" /Fopandas/_libs/tslibs/ccalendar.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_ccalendar.pyx.c.obj "/c" pandas/_libs/tslibs/ccalendar.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/ccalendar.pyx.c
      pandas/_libs/tslibs/ccalendar.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/ccalendar.pyx.c(7376): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      pandas/_libs/tslibs/ccalendar.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/ccalendar.pyx.c(7686): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      [30/151] Compiling C object pandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_dtypes.pyx.c.obj
      [31mFAILED: [code=2] [0mpandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_dtypes.pyx.c.obj
      "cl" "-Ipandas\_libs\tslibs\dtypes.cp314-win_amd64.pyd.p" "-Ipandas\_libs\tslibs" "-I..\..\pandas\_libs\tslibs" "-I..\..\..\..\pip-build-env-4sjrfjlv\overlay\Lib\site-packages\numpy\core\include" "-I..\..\pandas\_libs\include" "-IC:\Users\User\AppData\Roaming\uv\python\cpython-3.14-windows-x86_64-none\Include" "-DNDEBUG" "/MD" "/nologo" "/showIncludes" "/utf-8" "-w" "/O2" "/Gw" "-DNPY_NO_DEPRECATED_API=0" "-DNPY_TARGET_VERSION=NPY_1_21_API_VERSION" "/Fdpandas\_libs\tslibs\dtypes.cp314-win_amd64.pyd.p\meson-generated_pandas__libs_tslibs_dtypes.pyx.c.pdb" /Fopandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/meson-generated_pandas__libs_tslibs_dtypes.pyx.c.obj "/c" pandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/dtypes.pyx.c
      pandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/dtypes.pyx.c(16379): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      pandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/dtypes.pyx.c(16575): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      pandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/dtypes.pyx.c(16923): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      pandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/dtypes.pyx.c(17195): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      pandas/_libs/tslibs/dtypes.cp314-win_amd64.pyd.p/pandas/_libs/tslibs/dtypes.pyx.c(17429): error C2198: 'int _PyLong_AsByteArray(PyLongObject *,unsigned char *,size_t,int,int,int)': too few arguments for call
      [31/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/offsets.pyx
      [32/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/timestamps.pyx
      [33/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/timedeltas.pyx
      [34/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslibs/period.pyx
      [35/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/byteswap.pyx
      [36/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/testing.pyx
      [37/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/ops.pyx
      [38/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/index.pyx
      [39/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/reshape.pyx
      [40/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/parsers.pyx
      [41/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/sas.pyx
      [42/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/window/indexers.pyx
      [43/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/lib.pyx
      [44/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/tslib.pyx
      [45/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/interval.pyx
      [46/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/writers.pyx
      [47/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/window/aggregations.pyx
      [48/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/join.pyx
      [49/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/sparse.pyx
      [50/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/hashtable.pyx
      [51/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/groupby.pyx
      [52/151] Compiling Cython source C:/Users/User/AppData/Local/Temp/pip-install-lv7kq3m2/pandas_36ced063ec98477e837461cd3be1b215/pandas/_libs/algos.pyx
      ninja: build stopped: subcommand failed.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> pandas

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

#### Cell 23 Code Snippet:
```python
print(f'Final dataset shape: {df.shape}')
print(f'Total features: {df.shape[1] - 1}  |  Target column: target')
print(f'\nMissing values after preproc...
```
**Output:**
```
Final dataset shape: (50000, 114)
Total features: 113  |  Target column: target

Missing values after preprocessing: 0

Data types:
int64      100
float64     14
Name: count, dtype: int64
```

#### Cell 38 Code Snippet:
```python
# Train all models and collect results (With auto-save and auto-load to avoid repeating hours of training)
import joblib
import os

checkpoint_path = ...
```
**Output:**
```
🔄 Training all 6 baseline models (this might take a while)...

============================================================
Training: Logistic Regression
============================================================
  Train Acc: 0.5190  |  Test Acc: 0.4999
  Precision: 0.3958  |  Recall:   0.4935
  F1 Score:  0.4393       |  ROC-AUC:  0.5015
  Train Time: 0.11s

============================================================
Training: KNN
============================================================
  Train Acc: 0.7025  |  Test Acc: 0.5366
  Precision: 0.3951  |  Recall:   0.3149
  F1 Score:  0.3504       |  ROC-AUC:  0.5021
  Train Time: 0.02s

============================================================
Training: Decision Tree
============================================================
  Train Acc: 1.0000  |  Test Acc: 0.5236
  Precision: 0.4000  |  Recall:   0.4000
  F1 Score:  0.4000       |  ROC-AUC:  0.5025
  Train Time: 0.90s

============================================================
Training: Random Forest
============================================================
  Train Acc: 1.0000  |  Test Acc: 0.6019
  Precision: 0.3514  |  Recall:   0.0033
  F1 Score:  0.0065       |  ROC-AUC:  0.5098
  Train Time: 3.79s

============================================================
Training: XGBoost
============================================================
  Train Acc: 0.9720  |  Test Acc: 0.5335
  Precision: 0.4085  |  Recall:   0.3909
  F1 Score:  0.3995       |  ROC-AUC:  0.5052
  Train Time: 1.42s

============================================================
Training: SVM
============================================================
  Train Acc: 0.6030  |  Test Acc: 0.6030
  Precision: 0.0000  |  Recall:   0.0000
  F1 Score:  0.0000       |  ROC-AUC:  0.5093
  Train Time: 78.81s

💾 Pre-trained baseline models saved successfully to: baseline_results.joblib

============================================================
All baseline models loaded/trained successfully!
============================================================
```

#### Cell 39 Code Snippet:
```python
# Build comparison dataframe
if results:
    comparison = pd.DataFrame({
        name: {
            'Train Accuracy': r['train_acc'],
            'Te...
```
**Output:**
```
Model Comparison (sorted by Test Accuracy):
                     Train Accuracy  Test Accuracy  Precision  Recall  F1 Score  ROC-AUC  Train Time (s)  Overfit Gap
SVM                          0.6030         0.6030     0.0000  0.0000    0.0000   0.5093         78.8053       0.0000
Random Forest                1.0000         0.6019     0.3514  0.0033    0.0065   0.5098          3.7859       0.3981
KNN                          0.7024         0.5366     0.3951  0.3149    0.3504   0.5021          0.0175       0.1659
XGBoost                      0.9720         0.5335     0.4085  0.3909    0.3995   0.5052          1.4164       0.4385
Decision Tree                1.0000         0.5236     0.4000  0.4000    0.4000   0.5025          0.9049       0.4764
Logistic Regression          0.5190         0.4999     0.3958  0.4935    0.4393   0.5015          0.1059       0.0191
```

#### Cell 43 Code Snippet:
```python
for name, r in results.items():
    print(f'\n{"="*60}')
    print(f'{name}')
    print(f'{"="*60}')
    print(classification_report(y_test, r['y_pred...
```
**Output:**
```
============================================================
Logistic Regression
============================================================
              precision    recall  f1-score   support

    Negative       0.60      0.50      0.55      6030
    Positive       0.40      0.49      0.44      3970

    accuracy                           0.50     10000
   macro avg       0.50      0.50      0.49     10000
weighted avg       0.52      0.50      0.51     10000


============================================================
KNN
============================================================
              precision    recall  f1-score   support

    Negative       0.60      0.68      0.64      6030
    Positive       0.40      0.31      0.35      3970

    accuracy                           0.54     10000
   macro avg       0.50      0.50      0.50     10000
weighted avg       0.52      0.54      0.52     10000


============================================================
Decision Tree
============================================================
              precision    recall  f1-score   support

    Negative       0.60      0.60      0.60      6030
    Positive       0.40      0.40      0.40      3970

    accuracy                           0.52     10000
   macro avg       0.50      0.50      0.50     10000
weighted avg       0.52      0.52      0.52     10000


============================================================
Random Forest
============================================================
              precision    recall  f1-score   support

    Negative       0.60      1.00      0.75      6030
    Positive       0.35      0.00      0.01      3970

    accuracy                           0.60     10000
   macro avg       0.48      0.50      0.38     10000
weighted avg       0.50      0.60      0.46     10000


============================================================
XGBoost
============================================================
              precision    recall  f1-score   support

    Negative       0.61      0.63      0.62      6030
    Positive       0.41      0.39      0.40      3970

    accuracy                           0.53     10000
   macro avg       0.51      0.51      0.51     10000
weighted avg       0.53      0.53      0.53     10000


============================================================
SVM
============================================================
              precision    recall  f1-score   support

    Negative       0.60      1.00      0.75      6030
    Positive       0.00      0.00      0.00      3970

    accuracy                           0.60     10000
   macro avg       0.30      0.50      0.38     10000
weighted avg       0.36      0.60      0.45     10000
```

#### Cell 50 Code Snippet:
```python
# Tune the top 3 models (With auto-save and auto-load to avoid repeating hours of tuning)
import os
import joblib
checkpoint_tuned_path = 'tuned_resul...
```
**Output:**
```
🔄 Tuned results not found. Running hyperparameter tuning on top 3 models (this might take a while)...

============================================================
Tuning: SVM
============================================================
Fitting 5 folds for each of 30 candidates, totalling 150 fits

  Best Parameters: {'estimator__kernel': 'poly', 'estimator__gamma': 0.01, 'estimator__C': 100}
  Best CV F1:     0.0000
  Test Accuracy:  0.6030
  Test F1:        0.0000
  Test ROC-AUC:   0.5001
  Tuning Time:    8931.0s

============================================================
Tuning: Random Forest
============================================================
Fitting 5 folds for each of 30 candidates, totalling 150 fits

  Best Parameters: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 10}
  Best CV F1:     0.3848
  Test Accuracy:  0.5335
  Test F1:        0.3978
  Test ROC-AUC:   0.5076
  Tuning Time:    1150.6s

============================================================
Tuning: KNN
============================================================
Fitting 5 folds for each of 30 candidates, totalling 150 fits

  Best Parameters: {'weights': 'uniform', 'n_neighbors': 3, 'metric': 'euclidean'}
  Best CV F1:     0.3688
  Test Accuracy:  0.5392
  Test F1:        0.3780
  Test ROC-AUC:   0.5089
  Tuning Time:    84.3s

💾 Pre-trained tuned models saved successfully to: tuned_results.joblib

============================================================
Hyperparameter tuning complete!
============================================================
```

#### Cell 51 Code Snippet:
```python
# Compare baseline vs tuned for the top 3
print(f'{"Model":<25} {"Metric":<12} {"Baseline":>10} {"Tuned":>10} {"Change":>10}')
print('-' * 70)

for na...
```
**Output:**
```
Model                     Metric         Baseline      Tuned     Change
----------------------------------------------------------------------
SVM                       Accuracy         0.6030     0.6030 +   0.0000
SVM                       F1 Score         0.0000     0.0000 +   0.0000
SVM                       ROC-AUC          0.5093     0.5001   -0.0093

Random Forest             Accuracy         0.6019     0.5335   -0.0684
Random Forest             F1 Score         0.0065     0.3978 +   0.3913
Random Forest             ROC-AUC          0.5098     0.5076   -0.0022

KNN                       Accuracy         0.5366     0.5392 +   0.0026
KNN                       F1 Score         0.3504     0.3780 +   0.0275
KNN                       ROC-AUC          0.5021     0.5089 +   0.0068
```

#### Cell 53 Code Snippet:
```python
# Select best overall model
if tuned_results:
    best_name = max(tuned_results, key=lambda n: tuned_results[n]['f1'])
    best = tuned_results[best_n...
```
**Output:**
```
Best Model (Tuned): Random Forest
Best Parameters: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 10}

Test Accuracy: 0.5335
Test F1 Score: 0.3978
Test ROC-AUC:  0.5076

Classification Report:
              precision    recall  f1-score   support

    Negative       0.61      0.63      0.62      6030
    Positive       0.41      0.39      0.40      3970

    accuracy                           0.53     10000
   macro avg       0.51      0.51      0.51     10000
weighted avg       0.53      0.53      0.53     10000
```

#### Cell 57 Code Snippet:
```python
# Per-group accuracy breakdown (Testing for Demographic Parity)
print('Accuracy Breakdown by Gender:')
print('-' * 40)

# Retrieve the raw (unscaled/u...
```
**Output:**
```
Accuracy Breakdown by Gender:
----------------------------------------
Transgender              : 0.4826 (N=1722)
Prefer Not to Say        : 0.4973 (N=1645)
Female                   : 0.4896 (N=1630)
Non-binary               : 0.5066 (N=1664)
Genderfluid              : 0.4991 (N=1679)
Male                     : 0.5247 (N=1660)
```

#### Cell 58 Code Snippet:
```python
# Final comprehensive comparison: all baseline + all tuned
print('=' * 80)
print('FINAL MODEL COMPARISON')
print('=' * 80)

all_results = {}
for name,...
```
**Output:**
```
================================================================================
FINAL MODEL COMPARISON
================================================================================
                                Accuracy      F1  Precision  Recall  ROC-AUC  Train Time
Logistic Regression (Baseline)    0.4999  0.4393     0.3958  0.4935   0.5015      0.1059
Decision Tree (Baseline)          0.5236  0.4000     0.4000  0.4000   0.5025      0.9049
XGBoost (Baseline)                0.5335  0.3995     0.4085  0.3909   0.5052      1.4164
Random Forest (Tuned)             0.5335  0.3978     0.4080  0.3882   0.5076   1150.6441
KNN (Tuned)                       0.5392  0.3780     0.4072  0.3526   0.5089     84.2607
KNN (Baseline)                    0.5366  0.3504     0.3951  0.3149   0.5021      0.0175
Random Forest (Baseline)          0.6019  0.0065     0.3514  0.0033   0.5098      3.7859
SVM (Tuned)                       0.6030  0.0000     0.0000  0.0000   0.5001   8931.0402
SVM (Baseline)                    0.6030  0.0000     0.0000  0.0000   0.5093     78.8053

Best overall model: Logistic Regression (Baseline)
  F1 Score:  0.4393
  ROC-AUC:   0.5015
  Accuracy:  0.4999
```

#### Cell 61 Code Snippet:
```python
import pandas as pd
from pycaret.classification import setup, compare_models, pull

try:
    print("\n--- Starting PyCaret ---")
    
    # PyCaret wo...
```
**Output:**
```
Model  Accuracy     AUC  Recall   Prec.  \
svm                   SVM - Linear Kernel    0.6030  0.5010  0.0000  0.0000   
dummy                    Dummy Classifier    0.6030  0.5000  0.0000  0.0000   
ridge                    Ridge Classifier    0.6029  0.5071  0.0000  0.0000   
lr                    Logistic Regression    0.6028  0.5071  0.0000  0.0000   
ada                  Ada Boost Classifier    0.6028  0.5072  0.0040  0.4817   
lda          Linear Discriminant Analysis    0.6028  0.5071  0.0000  0.0000   
gbc          Gradient Boosting Classifier    0.6012  0.5054  0.0054  0.3619   
lightgbm  Light Gradient Boosting Machine    0.5952  0.4972  0.0412  0.4030   
rf               Random Forest Classifier    0.5920  0.5074  0.0550  0.4010   
et                 Extra Trees Classifier    0.5836  0.5034  0.0966  0.3984   
xgboost         Extreme Gradient Boosting    0.5631  0.5036  0.2104  0.4038   
nb                            Naive Bayes    0.5489  0.5058  0.2697  0.3997   
qda       Quadratic Discriminant Analysis    0.5447  0.5025  0.2961  0.4008   
knn                K Neighbors Classifier    0.5359  0.4920  0.3073  0.3920   
dt               Decision Tree Classifier    0.5147  0.4964  0.4073  0.3928   

              F1   Kappa     MCC  TT (Sec)  
svm       0.0000  0.0000  0.0000     0.941  
dummy     0.0000  0.0000  0.0000     0.106  
ridge     0.0000 -0.0002 -0.0037     0.181  
lr        0.0000 -0.0004 -0.0057     0.623  
ada       0.0080  0.0012  0.0073     1.781  
lda       0.0000 -0.0005 -0.0079     0.249  
gbc       0.0106 -0.0013 -0.0065     5.840  
lightgbm  0.0747  0.0014  0.0027     3.090  
rf        0.0966  0.0007  0.0016     1.407  
et        0.1554  0.0010  0.0012     1.616  
xgboost   0.2766  0.0062  0.0069     1.526  
nb        0.3219  0.0026  0.0029     0.236  
qda       0.3405  0.0047  0.0049     0.330  
knn       0.3444 -0.0066 -0.0068     1.212  
dt        0.3999 -0.0072 -0.0072     0.974  

Best PyCaret Model: SGDClassifier(alpha=0.0001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.001, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='hinge',
              max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
              power_t=0.5, random_state=123, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False)

PyCaret Leaderboard:
```

#### Cell 61 Code Snippet:
```python
import pandas as pd
from pycaret.classification import setup, compare_models, pull

try:
    print("\n--- Starting PyCaret ---")
    
    # PyCaret wo...
```
**Output:**
```
Model  Accuracy     AUC  Recall   Prec.     F1   Kappa  \
svm     SVM - Linear Kernel    0.6030  0.5010   0.000  0.0000  0.000  0.0000   
dummy      Dummy Classifier    0.6030  0.5000   0.000  0.0000  0.000  0.0000   
ridge      Ridge Classifier    0.6029  0.5071   0.000  0.0000  0.000 -0.0002   
lr      Logistic Regression    0.6028  0.5071   0.000  0.0000  0.000 -0.0004   
ada    Ada Boost Classifier    0.6028  0.5072   0.004  0.4817  0.008  0.0012   

          MCC  TT (Sec)  
svm    0.0000     0.941  
dummy  0.0000     0.106  
ridge -0.0037     0.181  
lr    -0.0057     0.623  
ada    0.0073     1.781
```

### Main Metrics & Key Outputs

#### Cell 23 Code Snippet:
```python
print(f'Final dataset shape: {df.shape}')
print(f'Total features: {df.shape[1] - 1}  |  Target column: target')
print(f'\nMissing values after preproc...
```
**Output:**
```
Final dataset shape: (50000, 114)
Total features: 113  |  Target column: target

Missing values after preprocessing: 0

Data types:
int32      99
float64    14
int64       1
Name: count, dtype: int64
```

#### Cell 38 Code Snippet:
```python
# Train all models and collect results (With auto-save and auto-load to avoid repeating hours of training)
import joblib
import os

checkpoint_path = ...
```
**Output:**
```
⚡ Successfully loaded all pre-trained baseline models and metrics from ../models/baseline_results.joblib!...
  Loaded Logistic Regression — Test Acc: 0.6030 | F1: 0.0000
  Loaded KNN — Test Acc: 0.5366 | F1: 0.3504
  Loaded Decision Tree — Test Acc: 0.5112 | F1: 0.3998
  Loaded Random Forest — Test Acc: 0.5990 | F1: 0.0461
  Loaded XGBoost — Test Acc: 0.5607 | F1: 0.2941
  Loaded SVM — Test Acc: 0.6030 | F1: 0.0000

============================================================
All baseline models loaded/trained successfully!
============================================================
```

#### Cell 39 Code Snippet:
```python
# Build comparison dataframe
if results:
    comparison = pd.DataFrame({
        name: {
            'Train Accuracy': r['train_acc'],
            'Te...
```
**Output:**
```
Model Comparison (sorted by Test Accuracy):
                     Train Accuracy  Test Accuracy  Precision  Recall  F1 Score  ROC-AUC  Train Time (s)  Overfit Gap
Logistic Regression          0.6030         0.6030     0.0000  0.0000    0.0000   0.5014          0.2501       0.0000
SVM                          0.6048         0.6030     0.0000  0.0000    0.0000   0.5143       1983.4667       0.0018
Random Forest                1.0000         0.5990     0.4145  0.0244    0.0461   0.5140          3.0006       0.4010
XGBoost                      0.8528         0.5607     0.4061  0.2305    0.2941   0.5092          0.7225       0.2921
KNN                          0.7024         0.5366     0.3951  0.3149    0.3504   0.5021          0.0179       0.1659
Decision Tree                1.0000         0.5112     0.3900  0.4101    0.3998   0.4939          0.6506       0.4888
```

#### Cell 43 Code Snippet:
```python
for name, r in results.items():
    print(f'\n{"="*60}')
    print(f'{name}')
    print(f'{"="*60}')
    print(classification_report(y_test, r['y_pred...
```
**Output:**
```
============================================================
Logistic Regression
============================================================
              precision    recall  f1-score   support

    Negative       0.60      1.00      0.75      6030
    Positive       0.00      0.00      0.00      3970

    accuracy                           0.60     10000
   macro avg       0.30      0.50      0.38     10000
weighted avg       0.36      0.60      0.45     10000


============================================================
KNN
============================================================
              precision    recall  f1-score   support

    Negative       0.60      0.68      0.64      6030
    Positive       0.40      0.31      0.35      3970

    accuracy                           0.54     10000
   macro avg       0.50      0.50      0.50     10000
weighted avg       0.52      0.54      0.52     10000


============================================================
Decision Tree
============================================================
              precision    recall  f1-score   support

    Negative       0.60      0.58      0.59      6030
    Positive       0.39      0.41      0.40      3970

    accuracy                           0.51     10000
   macro avg       0.49      0.49      0.49     10000
weighted avg       0.52      0.51      0.51     10000


============================================================
Random Forest
============================================================
              precision    recall  f1-score   support

    Negative       0.60      0.98      0.75      6030
    Positive       0.41      0.02      0.05      3970

    accuracy                           0.60     10000
   macro avg       0.51      0.50      0.40     10000
weighted avg       0.53      0.60      0.47     10000


============================================================
XGBoost
============================================================
              precision    recall  f1-score   support

    Negative       0.61      0.78      0.68      6030
    Positive       0.41      0.23      0.29      3970

    accuracy                           0.56     10000
   macro avg       0.51      0.50      0.49     10000
weighted avg       0.53      0.56      0.53     10000


============================================================
SVM
============================================================
              precision    recall  f1-score   support

    Negative       0.60      1.00      0.75      6030
    Positive       0.00      0.00      0.00      3970

    accuracy                           0.60     10000
   macro avg       0.30      0.50      0.38     10000
weighted avg       0.36      0.60      0.45     10000
```

#### Cell 50 Code Snippet:
```python
num_threads = os.cpu_count() or 1
# Tune the top 3 models (With auto-save and auto-load to avoid repeating hours of tuning)
import os
import joblib
ch...
```
**Output:**
```
⚡ Loading pre-trained tuned models and metrics from ../models/tuned_results.joblib instantly!...
  Loaded Tuned Logistic Regression — Test Acc: 0.6030 | F1: 0.0000
  Loaded Tuned SVM — Test Acc: 0.6030 | F1: 0.0000
  Loaded Tuned Random Forest — Test Acc: 0.5919 | F1: 0.0969

============================================================
Hyperparameter tuning complete!
============================================================
```

#### Cell 51 Code Snippet:
```python
# Compare baseline vs tuned for the top 3
print(f'{"Model":<25} {"Metric":<12} {"Baseline":>10} {"Tuned":>10} {"Change":>10}')
print('-' * 70)

for na...
```
**Output:**
```
Model                     Metric         Baseline      Tuned     Change
----------------------------------------------------------------------
Logistic Regression       Accuracy         0.6030     0.6030 +   0.0000
Logistic Regression       F1 Score         0.0000     0.0000 +   0.0000
Logistic Regression       ROC-AUC          0.5014     0.5019 +   0.0006

SVM                       Accuracy         0.6030     0.6030 +   0.0000
SVM                       F1 Score         0.0000     0.0000 +   0.0000
SVM                       ROC-AUC          0.5143     0.5025   -0.0118

Random Forest             Accuracy         0.5990     0.5919   -0.0071
Random Forest             F1 Score         0.0461     0.0969 +   0.0508
Random Forest             ROC-AUC          0.5140     0.5111   -0.0029
```

#### Cell 53 Code Snippet:
```python
# Select best overall model
if tuned_results:
    best_name = max(tuned_results, key=lambda n: tuned_results[n]['f1'])
    best = tuned_results[best_n...
```
**Output:**
```
Best Model (Tuned): Random Forest
Best Parameters: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': None}

Test Accuracy: 0.5919
Test F1 Score: 0.0969
Test ROC-AUC:  0.5111

Classification Report:
              precision    recall  f1-score   support

    Negative       0.60      0.95      0.74      6030
    Positive       0.40      0.06      0.10      3970

    accuracy                           0.59     10000
   macro avg       0.50      0.50      0.42     10000
weighted avg       0.52      0.59      0.48     10000
```

#### Cell 57 Code Snippet:
```python
# Per-group accuracy breakdown (Testing for Demographic Parity)
print('Accuracy Breakdown by Gender:')
print('-' * 40)

# Retrieve the raw (unscaled/u...
```
**Output:**
```
Accuracy Breakdown by Gender:
----------------------------------------
Transgender              : 0.5970 (N=1722)
Prefer Not to Say        : 0.6109 (N=1645)
Female                   : 0.6067 (N=1630)
Non-binary               : 0.6226 (N=1664)
Genderfluid              : 0.6069 (N=1679)
Male                     : 0.5741 (N=1660)
```

#### Cell 58 Code Snippet:
```python
# Final comprehensive comparison: all baseline + all tuned
print('=' * 80)
print('FINAL MODEL COMPARISON')
print('=' * 80)

all_results = {}
for name,...
```
**Output:**
```
================================================================================
FINAL MODEL COMPARISON
================================================================================
                                Accuracy      F1  Precision  Recall  ROC-AUC  Train Time
Decision Tree (Baseline)          0.5112  0.3998     0.3900  0.4101   0.4939      0.6506
KNN (Baseline)                    0.5366  0.3504     0.3951  0.3149   0.5021      0.0179
XGBoost (Baseline)                0.5607  0.2941     0.4061  0.2305   0.5092      0.7225
Random Forest (Tuned)             0.5919  0.0969     0.3989  0.0552   0.5111   2081.2140
Random Forest (Baseline)          0.5990  0.0461     0.4145  0.0244   0.5140      3.0006
Logistic Regression (Baseline)    0.6030  0.0000     0.0000  0.0000   0.5014      0.2501
SVM (Baseline)                    0.6030  0.0000     0.0000  0.0000   0.5143   1983.4667
Logistic Regression (Tuned)       0.6030  0.0000     0.0000  0.0000   0.5019     17.5811
SVM (Tuned)                       0.6030  0.0000     0.0000  0.0000   0.5025  17551.3808

Best overall model: Decision Tree (Baseline)
  F1 Score:  0.3998
  ROC-AUC:   0.4939
  Accuracy:  0.5112
```

#### Cell 60 Code Snippet:
```python
from flaml import AutoML
import sklearn.metrics
import joblib
import os

try:
    print("--- Starting FLAML ---")
    flaml_checkpoint_path = '../mode...
```
**Output:**
```
--- Starting FLAML ---

Best FLAML model found: rf
Best FLAML hyperparameters: {'n_estimators': 4, 'max_leaves': 11, 'max_features': 1.0, 'criterion': 'entropy'}

FLAML Test Accuracy: 0.6028
```

#### Cell 61 Code Snippet:
```python
import pandas as pd
try:
    from pycaret.classification import setup, compare_models, pull
    print("\n--- Starting PyCaret ---")
    
    # PyCaret...
```
**Output:**
```
Model  Accuracy     AUC  Recall   Prec.  \
svm                   SVM - Linear Kernel    0.6030  0.5010  0.0000  0.0000   
dummy                    Dummy Classifier    0.6030  0.5000  0.0000  0.0000   
ridge                    Ridge Classifier    0.6029  0.5071  0.0000  0.0000   
lr                    Logistic Regression    0.6028  0.5071  0.0000  0.0000   
ada                  Ada Boost Classifier    0.6028  0.5072  0.0040  0.4817   
lda          Linear Discriminant Analysis    0.6028  0.5071  0.0000  0.0000   
gbc          Gradient Boosting Classifier    0.6012  0.5054  0.0054  0.3619   
lightgbm  Light Gradient Boosting Machine    0.5952  0.4972  0.0412  0.4030   
rf               Random Forest Classifier    0.5920  0.5074  0.0550  0.4010   
et                 Extra Trees Classifier    0.5836  0.5034  0.0966  0.3984   
xgboost         Extreme Gradient Boosting    0.5631  0.5036  0.2104  0.4038   
nb                            Naive Bayes    0.5489  0.5058  0.2697  0.3997   
qda       Quadratic Discriminant Analysis    0.5447  0.5025  0.2961  0.4008   
knn                K Neighbors Classifier    0.5359  0.4920  0.3073  0.3920   
dt               Decision Tree Classifier    0.5147  0.4964  0.4073  0.3928   

              F1   Kappa     MCC  TT (Sec)  
svm       0.0000  0.0000  0.0000     0.593  
dummy     0.0000  0.0000  0.0000     0.093  
ridge     0.0000 -0.0002 -0.0037     0.140  
lr        0.0000 -0.0004 -0.0057     0.219  
ada       0.0080  0.0012  0.0073     1.512  
lda       0.0000 -0.0005 -0.0079     0.224  
gbc       0.0106 -0.0013 -0.0065     4.805  
lightgbm  0.0747  0.0014  0.0027     1.605  
rf        0.0966  0.0007  0.0016     1.507  
et        0.1554  0.0010  0.0012     1.477  
xgboost   0.2766  0.0062  0.0069     1.413  
nb        0.3219  0.0026  0.0029     0.174  
qda       0.3405  0.0047  0.0049     0.304  
knn       0.3444 -0.0066 -0.0068     0.683  
dt        0.3999 -0.0072 -0.0072     0.605  

Best PyCaret Model: SGDClassifier(alpha=0.0001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.001, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='hinge',
              max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
              power_t=0.5, random_state=123, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False)

PyCaret Leaderboard:
```

#### Cell 61 Code Snippet:
```python
import pandas as pd
try:
    from pycaret.classification import setup, compare_models, pull
    print("\n--- Starting PyCaret ---")
    
    # PyCaret...
```
**Output:**
```
Model  Accuracy     AUC  Recall   Prec.     F1   Kappa  \
svm     SVM - Linear Kernel    0.6030  0.5010   0.000  0.0000  0.000  0.0000   
dummy      Dummy Classifier    0.6030  0.5000   0.000  0.0000  0.000  0.0000   
ridge      Ridge Classifier    0.6029  0.5071   0.000  0.0000  0.000 -0.0002   
lr      Logistic Regression    0.6028  0.5071   0.000  0.0000  0.000 -0.0004   
ada    Ada Boost Classifier    0.6028  0.5072   0.004  0.4817  0.008  0.0012   

          MCC  TT (Sec)  
svm    0.0000     0.593  
dummy  0.0000     0.093  
ridge -0.0037     0.140  
lr    -0.0057     0.219  
ada    0.0073     1.512
```