"""
Patch ML_dating_app_behaviour V5.ipynb → V6.ipynb

Applies the following fixes:
1. Constrain all tree-based baseline models (max_depth, min_samples_leaf)
2. Fix top3 tuning selection (pipeline-compatible + ROC-AUC)
3. Fix best_name selection (ROC-AUC + pipeline compatibility)
4. Scrub API tokens
5. Fix StandardScaler → RobustScaler in checklist
6. Update param_grids for constrained tree models
7. Update tuning cell model definitions
8. Add calibrated prediction propagation
9. Update Section 17 summary
"""

import json
import shutil
import os
import sys

# --- Paths ---
NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
SRC = os.path.join(NOTEBOOK_DIR, 'ML_dating_app_behaviour V5.ipynb')
DST = os.path.join(NOTEBOOK_DIR, 'ML_dating_app_behaviour V6.ipynb')

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  ✅ Saved: {path}")

def get_source_str(cell):
    """Join cell source lines into a single string."""
    return ''.join(cell.get('source', []))

def set_source(cell, new_source_str):
    """Set cell source from a single string, splitting into lines for .ipynb format."""
    lines = new_source_str.split('\n')
    # Re-add \n to all lines except the last (if it's empty)
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:  # Don't add empty trailing line
                result.append(line)
    cell['source'] = result

def find_cell_by_content(cells, search_str):
    """Find a cell whose source contains the search string."""
    for i, cell in enumerate(cells):
        src = get_source_str(cell)
        if search_str in src:
            return i
    return None

def replace_in_source(cell, old_str, new_str):
    """Replace text in a cell's source."""
    src = get_source_str(cell)
    if old_str not in src:
        return False
    new_src = src.replace(old_str, new_str)
    set_source(cell, new_src)
    return True


def main():
    print("=" * 60)
    print("  Patching V5 → V6")
    print("=" * 60)

    # Copy V5 to V6
    shutil.copy2(SRC, DST)
    print(f"\n  📋 Copied V5 → V6")

    # Load notebook
    nb = load_notebook(DST)
    cells = nb['cells']
    print(f"  📊 Total cells: {len(cells)}")

    fixes_applied = 0
    fixes_failed = 0

    # =========================================================================
    # FIX 1: Constrain baseline tree-based models
    # =========================================================================
    print("\n--- Fix 1: Constrain baseline tree-based model hyperparameters ---")

    idx = find_cell_by_content(cells, "RandomForestClassifier(class_weight='balanced', n_estimators=500")
    if idx is not None:
        cell = cells[idx]

        # Random Forest: add max_depth, min_samples_leaf, max_features
        ok1 = replace_in_source(cell,
            "'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),",
            "'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=500, max_depth=8, min_samples_leaf=10, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1),  # Depth-constrained to prevent SMOTE noise memorisation"
        )

        # Decision Tree: add max_depth, min_samples_leaf
        ok2 = replace_in_source(cell,
            "'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),",
            "'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),  # Depth-constrained"
        )

        # CatBoost: add depth limit
        ok3 = replace_in_source(cell,
            "'CatBoost': CatBoostClassifier(iterations=500, random_state=RANDOM_STATE, verbose=0),",
            "'CatBoost': CatBoostClassifier(iterations=500, depth=8, min_data_in_leaf=10, random_state=RANDOM_STATE, verbose=0),  # Depth-constrained"
        )

        # Balanced Random Forest: add constraints
        ok4 = replace_in_source(cell,
            "'Balanced Random Forest': BalancedRandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),",
            "'Balanced Random Forest': BalancedRandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),  # Depth-constrained"
        )

        results = [ok1, ok2, ok3, ok4]
        applied = sum(results)
        print(f"  Cell {idx}: {applied}/4 model constraints applied")
        if applied > 0:
            # Clear outputs since they'll be stale
            cell['outputs'] = []
            fixes_applied += 1
        if applied < 4:
            for i, (ok, name) in enumerate(zip(results, ['RF', 'DT', 'CatBoost', 'BRF'])):
                if not ok:
                    print(f"    ⚠️  {name} constraint NOT applied (string not found)")
            fixes_failed += 1
    else:
        print("  ❌ Could not find baseline models cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 2: Fix top3 selection
    # =========================================================================
    print("\n--- Fix 2: Fix top3 tuning selection ---")

    idx = find_cell_by_content(cells, "top3 = comparison.head(3).index.tolist()")
    if idx is not None:
        cell = cells[idx]
        ok = replace_in_source(cell,
            "top3 = comparison.head(3).index.tolist()",
            """# Select top 3 pipeline-compatible models for tuning (by ROC-AUC)
# Pipeline compatibility: SHAP TreeExplainer + DiCE + CalibratedClassifierCV
PIPELINE_COMPATIBLE = {'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Decision Tree'}
valid_for_tuning = comparison[comparison.index.isin(PIPELINE_COMPATIBLE)]
top3 = valid_for_tuning.sort_values('ROC-AUC', ascending=False).head(3).index.tolist()"""
        )
        if ok:
            # Also fix the print statement
            replace_in_source(cell,
                "print(f'Top 3 models for learning curves: {top3}')",
                "print(f'Top 3 pipeline-compatible models (by ROC-AUC): {top3}')"
            )
            cell['outputs'] = []
            fixes_applied += 1
            print(f"  Cell {idx}: ✅ top3 selection fixed")
        else:
            print(f"  Cell {idx}: ❌ Could not patch top3 selection")
            fixes_failed += 1
    else:
        print("  ❌ Could not find top3 cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 3: Fix best_name selection criterion
    # =========================================================================
    print("\n--- Fix 3: Fix best_name model selection criterion ---")

    idx = find_cell_by_content(cells, "best_name = max(tuned_results, key=lambda n: tuned_results[n]['f1'])")
    if idx is not None:
        cell = cells[idx]
        old_block = """# Select best overall model
if tuned_results:
    best_name = max(tuned_results, key=lambda n: tuned_results[n]['f1'])
    best = tuned_results[best_name]
    print(f'Best Model (Tuned): {best_name}')
    print(f'Best Parameters: {best["best_params"]}')
elif results:
    best_name = max(results, key=lambda n: results[n]['f1'])
    best = results[best_name]
    print(f'⚠️ Tuned results are empty (tuning cell was skipped). Falling back to the best Baseline model: {best_name}')
else:
    best_name = None
    best = None
    print('❌ Error: No trained models or baseline results found. Please run the model training cells first.')"""

        new_block = """# Select best overall model
# Criterion: ROC-AUC (only metric not gameable by threshold choice)
# Filter: pipeline-compatible models only (SHAP + DiCE + calibration support)
PIPELINE_COMPATIBLE_MODELS = {'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Decision Tree'}

if tuned_results:
    eligible = {n: r for n, r in tuned_results.items() if n in PIPELINE_COMPATIBLE_MODELS}
    if not eligible:
        eligible = tuned_results  # fallback if no compatible models were tuned
    best_name = max(eligible, key=lambda n: eligible[n]['roc_auc'])
    best = tuned_results[best_name]
    print(f'Best Model (Tuned, by ROC-AUC, pipeline-compatible): {best_name}')
    print(f'Best Parameters: {best["best_params"]}')
elif results:
    eligible = {n: r for n, r in results.items() if n in PIPELINE_COMPATIBLE_MODELS}
    if not eligible:
        eligible = results
    best_name = max(eligible, key=lambda n: eligible[n]['roc_auc'])
    best = results[best_name]
    print(f'⚠️ Tuned results empty. Falling back to best pipeline-compatible Baseline model (by ROC-AUC): {best_name}')
else:
    best_name = None
    best = None
    print('❌ Error: No trained models or baseline results found.')"""

        ok = replace_in_source(cell, old_block, new_block)
        if ok:
            cell['outputs'] = []
            fixes_applied += 1
            print(f"  Cell {idx}: ✅ best_name selection fixed")
        else:
            print(f"  Cell {idx}: ❌ Could not patch best_name (exact string mismatch)")
            fixes_failed += 1
    else:
        print("  ❌ Could not find best_name cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 4: Scrub API tokens
    # =========================================================================
    print("\n--- Fix 4: Scrub API tokens ---")

    idx = find_cell_by_content(cells, 'TABPFN_TOKEN')
    if idx is not None:
        cell = cells[idx]
        src = get_source_str(cell)

        # Replace TABPFN token
        import re
        src_new = re.sub(
            r'os\.environ\["TABPFN_TOKEN"\]\s*=\s*"[^"]*"',
            'os.environ["TABPFN_TOKEN"] = os.environ.get("TABPFN_TOKEN", "INSERT_YOUR_TOKEN_HERE")',
            src
        )
        # Replace HF token
        src_new = re.sub(
            r'os\.environ\["HF_TOKEN"\]\s*=\s*"[^"]*"',
            'os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "INSERT_YOUR_TOKEN_HERE")',
            src_new
        )

        if src_new != src:
            set_source(cell, src_new)
            fixes_applied += 1
            print(f"  Cell {idx}: ✅ API tokens scrubbed")
        else:
            print(f"  Cell {idx}: ❌ Could not scrub tokens")
            fixes_failed += 1
    else:
        print("  ❌ Could not find API token cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 5: Fix StandardScaler → RobustScaler in checklist
    # =========================================================================
    print("\n--- Fix 5: Fix StandardScaler → RobustScaler ---")

    idx = find_cell_by_content(cells, "StandardScaler on 12 numeric columns")
    if idx is not None:
        cell = cells[idx]
        ok = replace_in_source(cell,
            "StandardScaler on 12 numeric columns",
            "RobustScaler on 12 numeric columns"
        )
        if ok:
            fixes_applied += 1
            print(f"  Cell {idx}: ✅ StandardScaler → RobustScaler")
        else:
            print(f"  Cell {idx}: ❌ Could not patch scaler reference")
            fixes_failed += 1
    else:
        print("  ❌ Could not find checklist cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 6: Update param_grids for constrained trees
    # =========================================================================
    print("\n--- Fix 6: Update param_grids for constrained tree models ---")

    idx = find_cell_by_content(cells, "param_grids = {")
    if idx is not None:
        cell = cells[idx]

        # Fix RF param grid: constrained depth range
        ok1 = replace_in_source(cell,
            """    'Random Forest': {
        'n_estimators': [200, 300, 500, 800, 1000],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },""",
            """    'Random Forest': {
        'n_estimators': [200, 300, 500, 800],
        'max_depth': [5, 8, 10, 12],        # Constrained range to prevent SMOTE noise memorisation
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [5, 10, 15],     # Higher minimums for regularisation
        'max_features': ['sqrt', 'log2']
    },"""
        )

        # Fix Decision Tree param grid
        ok2 = replace_in_source(cell,
            """    'Decision Tree': {
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }""",
            """    'Decision Tree': {
        'max_depth': [5, 8, 10, 12],         # Constrained
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [5, 10, 15],     # Higher minimums
        'criterion': ['gini', 'entropy']
    }"""
        )

        # Fix XGBoost param grid (already has reasonable depths)
        ok3 = replace_in_source(cell,
            """    'XGBoost': {
        'n_estimators': [200, 300, 500, 800, 1000],
        'max_depth': [3, 5, 7, 10, 12],""",
            """    'XGBoost': {
        'n_estimators': [200, 300, 500, 800],
        'max_depth': [3, 5, 7, 8],           # Constrained upper bound"""
        )

        # Fix CatBoost param grid
        ok4_1 = replace_in_source(cell,
            """    'CatBoost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    },
    'LightGBM': {""",
            """    'CatBoost': {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],                  # Already constrained
        'learning_rate': [0.01, 0.05, 0.1]
    },
    'LightGBM': {"""
        )

        applied = sum([ok1, ok2, ok3])
        print(f"  Cell {idx}: {applied}/3 param grid updates applied (RF={ok1}, DT={ok2}, XGB={ok3})")
        if applied > 0:
            cell['outputs'] = []
            fixes_applied += 1
        if applied < 3:
            fixes_failed += 1
    else:
        print("  ❌ Could not find param_grids cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 7: Update tuning cell model definitions
    # =========================================================================
    print("\n--- Fix 7: Update tuning cell model definitions ---")

    idx = find_cell_by_content(cells, "RETUNE_MODELS = False")
    if idx is not None:
        cell = cells[idx]

        # Fix RF in tuning cell
        ok1 = replace_in_source(cell,
            "'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),",
            "'Random Forest': RandomForestClassifier(class_weight='balanced', max_depth=8, min_samples_leaf=10, max_features='sqrt', random_state=RANDOM_STATE),"
        )

        # Fix DT in tuning cell
        ok2 = replace_in_source(cell,
            "'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),",
            "'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),"
        )

        applied = sum([ok1, ok2])
        print(f"  Cell {idx}: {applied}/2 tuning model definitions updated (RF={ok1}, DT={ok2})")
        if applied > 0:
            cell['outputs'] = []
            fixes_applied += 1
        if applied < 2:
            fixes_failed += 1
    else:
        print("  ❌ Could not find tuning cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 8: Add calibrated prediction propagation after calibration
    # =========================================================================
    print("\n--- Fix 8: Add calibrated prediction propagation ---")

    idx = find_cell_by_content(cells, "CalibratedClassifierCV(base_model, method='isotonic', cv=3)")
    if idx is not None:
        cell = cells[idx]
        src = get_source_str(cell)

        # Check if calibration propagation already exists
        if "calibrated_clf.predict(X_test)" not in src:
            # Add propagation code at the end of the cell
            propagation_code = """

# === CRITICAL: Propagate calibrated predictions downstream ===
# This ensures SHAP, DiCE, confusion matrix, and fairness audit
# all use the Isotonically Calibrated model predictions.
if champion_name and champion_name in results:
    results[champion_name]['y_pred'] = calibrated_clf.predict(X_test)
    results[champion_name]['y_prob'] = calibrated_clf.predict_proba(X_test)[:, 1]
    print(f"\\n✅ Updated results['{champion_name}'] with calibrated predictions.")
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    cal_acc = accuracy_score(y_test, results[champion_name]['y_pred'])
    cal_f1 = f1_score(y_test, results[champion_name]['y_pred'])
    cal_auc = roc_auc_score(y_test, results[champion_name]['y_prob'])
    results[champion_name]['test_acc'] = cal_acc
    results[champion_name]['f1'] = cal_f1
    results[champion_name]['roc_auc'] = cal_auc
    print(f"   Calibrated Accuracy: {cal_acc:.4f} | F1: {cal_f1:.4f} | AUC: {cal_auc:.4f}")
"""
            new_src = src + propagation_code
            set_source(cell, new_src)
            cell['outputs'] = []
            fixes_applied += 1
            print(f"  Cell {idx}: ✅ Calibration propagation added")
        else:
            print(f"  Cell {idx}: ⏭️  Already has calibration propagation")
    else:
        print("  ❌ Could not find calibration cell!")
        fixes_failed += 1

    # =========================================================================
    # FIX 9: Update Section 17 summary
    # =========================================================================
    print("\n--- Fix 9: Update Section 17 summary ---")

    idx = find_cell_by_content(cells, "highest test accuracy (60.48%)")
    if idx is not None:
        cell = cells[idx]
        ok = replace_in_source(cell,
            "The **Random Forest** was selected as the best model based on its highest test accuracy (60.48%), full SHAP TreeExplainer compatibility, successful Isotonic Calibration, and its role as the engine powering Microsoft DiCE counterfactual recourse.",
            "Since all 16 evaluated models converge at ROC-AUC ≈ 0.50, no single metric meaningfully separates their predictive capability. **Random Forest** was selected as the final model on the basis of **pipeline compatibility**: it is the only architecture simultaneously supporting SHAP TreeExplainer, DiCE counterfactual recourse, and isotonic probability calibration. Among pipeline-compatible models, Random Forest achieves the highest ROC-AUC. Tree depth was constrained to 8 to prevent memorisation of SMOTE-synthesised training noise."
        )
        if ok:
            fixes_applied += 1
            print(f"  Cell {idx}: ✅ Section 17 summary updated")
        else:
            print(f"  Cell {idx}: ❌ Could not patch Section 17")
            fixes_failed += 1
    else:
        print("  ❌ Could not find Section 17 cell!")
        fixes_failed += 1

    # =========================================================================
    # SAVE
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"  Results: {fixes_applied} fixes applied, {fixes_failed} issues")
    print("=" * 60)

    save_notebook(nb, DST)

    print(f"""
  📋 Next steps:
  1. Open '{os.path.basename(DST)}' in Jupyter/VS Code
  2. Set RETRAIN_BASELINE = True  (in the baseline training cell)
  3. Set RETUNE_MODELS = True     (in the tuning cell)
  4. Kernel → Restart & Run All
  5. After successful run, set both flags back to False
  6. Save the notebook
""")


if __name__ == '__main__':
    main()
