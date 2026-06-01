"""Fix remaining 3 issues found by deep audit."""
import json

NB_PATH = r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes = []

# ─────────────────────────────────────────────────────────────
# FIX 1: Cell 144 — FLAML cell was mangled by regex. Rewrite cleanly.
# ─────────────────────────────────────────────────────────────
FLAML_CLEAN = '''from flaml import AutoML
import sklearn.metrics
import joblib
import os

try:
    print("--- Starting FLAML ---")
    flaml_checkpoint_path = '../models_v7/flaml_results.joblib'
    os.makedirs('../models_v7', exist_ok=True)

    if os.path.exists(flaml_checkpoint_path):
        print(f"Loading pre-trained FLAML model from {flaml_checkpoint_path}...")
        automl = joblib.load(flaml_checkpoint_path)
    else:
        print("Training FLAML AutoML from scratch on V7 features...")
        automl = AutoML()
        automl.fit(
            X_train, y_train,
            task="classification",
            metric="f1",
            time_budget=120,  # 2 minutes budget
            n_jobs=1,
            verbose=0,
            seed=RANDOM_STATE
        )
        joblib.dump(automl, flaml_checkpoint_path)
        print(f"FLAML model saved to {flaml_checkpoint_path}")

    print(f"\\nBest FLAML model found: {automl.best_estimator}")
    print(f"Best FLAML hyperparameters: {automl.best_config}")

    # Evaluate
    flaml_predictions = automl.predict(X_test)
    flaml_accuracy = sklearn.metrics.accuracy_score(y_test, flaml_predictions)
    flaml_f1 = sklearn.metrics.f1_score(y_test, flaml_predictions)
    flaml_auc = sklearn.metrics.roc_auc_score(y_test, automl.predict_proba(X_test)[:, 1])
    print(f"\\nFLAML Test Accuracy: {flaml_accuracy:.4f}")
    print(f"FLAML Test F1:       {flaml_f1:.4f}")
    print(f"FLAML Test ROC-AUC:  {flaml_auc:.4f}")
except Exception as e:
    print(f"FLAML Error: {e}")
'''

cell_144 = nb['cells'][144]
cell_144['source'] = [line + '\n' for line in FLAML_CLEAN.split('\n')]
if cell_144['source']:
    cell_144['source'][-1] = cell_144['source'][-1].rstrip('\n')
fixes.append('Cell 144: FLAML — rewritten cleanly to train from scratch on V7 features')

# ─────────────────────────────────────────────────────────────
# FIX 2: Cell 149 — H-statistic hardcoded fallback loop
# ─────────────────────────────────────────────────────────────
src = ''.join(nb['cells'][149].get('source', []))
if "for name in ['Random Forest (Tuned)'" in src:
    new_src = src.replace(
        "for name in ['Random Forest (Tuned)', 'Random Forest', 'XGBoost (Tuned)', 'XGBoost', 'LightGBM']:",
        "for name in [best_name]:  # Use the dynamically selected champion model"
    )
    nb['cells'][149]['source'] = [line + '\n' for line in new_src.split('\n')]
    if nb['cells'][149]['source']:
        nb['cells'][149]['source'][-1] = nb['cells'][149]['source'][-1].rstrip('\n')
    fixes.append('Cell 149: H-statistic — replaced hardcoded fallback with best_name')

# ─────────────────────────────────────────────────────────────
# FIX 3: Cell 173 — DiCE hardcoded fallback loop
# ─────────────────────────────────────────────────────────────
src = ''.join(nb['cells'][173].get('source', []))
if "for name in ['Random Forest', 'Random Forest (Tuned)'" in src:
    new_src = src.replace(
        "for name in ['Random Forest', 'Random Forest (Tuned)', 'XGBoost', 'LightGBM']:",
        "for name in [best_name]:  # Use the dynamically selected champion model"
    )
    nb['cells'][173]['source'] = [line + '\n' for line in new_src.split('\n')]
    if nb['cells'][173]['source']:
        nb['cells'][173]['source'][-1] = nb['cells'][173]['source'][-1].rstrip('\n')
    fixes.append('Cell 173: DiCE — replaced hardcoded fallback with best_name')

# ─────────────────────────────────────────────────────────────
# NOTE: Cell 2 syntax error is a !pip install line (Colab magic).
# Python's compile() can't parse shell commands. This is a false positive.
# ─────────────────────────────────────────────────────────────

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'\n=== FIXES APPLIED: {len(fixes)} ===')
for f in fixes:
    print(f'  [OK] {f}')
print('\n  [SKIP] Cell 2: !pip install — false positive (Colab shell magic, not Python syntax)')
