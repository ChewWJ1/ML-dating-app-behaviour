import json
import re

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Track Isolation Forest cell
iso_forest_cell = None
iso_forest_idx = -1

for i, cell in enumerate(nb.get("cells", [])):
    source_str = "".join(cell.get("source", []))
    if "iso_forest = IsolationForest(contamination=0.05" in source_str:
        iso_forest_cell = cell
        iso_forest_idx = i
        break

if iso_forest_idx != -1:
    # Pop it from its current location
    nb["cells"].pop(iso_forest_idx)

# Now iterate and clean syntax errors
for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            
            # 1. Fix dangling commas from removed 'mutual_matches'
            # Look for multiple commas with spaces in between
            line = re.sub(r",\s*,", ",", line)
            # Remove comma before closing bracket
            line = re.sub(r",\s*\]", "]", line)

            # 2. Fix broken XGBoost dictionary entry 1
            if "'XGBoost': XGBClassifier(n_estimators=500, eval_metric='logloss', tree_method='hist'" in line and "**TREE_CONFIG" in line:
                line = "    'XGBoost': XGBClassifier(n_estimators=500, eval_metric='logloss', tree_method='hist', device=TREE_CONFIG['xgb'].get('device', 'cpu'), scale_pos_weight=1, random_state=RANDOM_STATE, n_jobs=-1),  # Neutral: SMOTE already balances classes 50/50\n"

            # 3. Fix broken XGBoost tuple entry 2
            if "('xgb', XGBClassifier(n_estimators=500, eval_metric='logloss', tree_method='hist'" in line and "GradientBoostingClassifier" in line:
                line = "                    ('xgb', XGBClassifier(n_estimators=500, eval_metric='logloss', tree_method='hist', device=TREE_CONFIG['xgb'].get('device', 'cpu'), scale_pos_weight=1, random_state=RANDOM_STATE, n_jobs=1)) if HAS_XGBOOST else ('gbm', GradientBoostingClassifier(n_estimators=500, random_state=RANDOM_STATE)),\n"

            # 4. Remove any rogue XGB_DEVICE references which were incorrectly injected
            if "device=XGB_DEVICE" in line and "XGB_DEVICE" not in "".join(cell["source"]):
                # The previous agent injected XGB_DEVICE but didn't define it. TREE_CONFIG['xgb']['device'] should be used.
                line = line.replace("device=XGB_DEVICE", "device=TREE_CONFIG['xgb'].get('device', 'cpu')")

            # 5. Fix indentation error from SMOTE fix
            if "    X_train, y_train = X_train_raw, y_train_raw  # Keep unsmoted globally for CV" in line:
                line = line.replace("    X_train, y_train = X_train_raw, y_train_raw", "X_train, y_train = X_train_raw, y_train_raw")

            # 6. Fix indentation error from Calibration fix
            if "                calibrated_clf = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')" in line:
                line = line.replace("                calibrated_clf =", "        calibrated_clf =")

            new_source.append(line)
            
        cell["source"] = new_source

# Now insert the Isolation Forest properly after Section 7 Train/Test Split
# We look for the cell where X_train is defined after PCA ablation or similar.
# In the original notebook, Section 7 ends with:
# print(f"Training features (X_train): {X_train.shape}")
# print(f"Testing features (X_test):   {X_test.shape}")
# We'll search for this string.
insert_idx = -1
for i, cell in enumerate(nb.get("cells", [])):
    source_str = "".join(cell.get("source", []))
    if "Training features (X_train):" in source_str and "Testing features (X_test):" in source_str:
        insert_idx = i
        break

if insert_idx != -1 and iso_forest_cell:
    nb["cells"].insert(insert_idx + 1, iso_forest_cell)
    print("Moved Isolation Forest guardrail strictly to after Section 7 Train/Test split.")
else:
    print("Could not find exact insertion point for Isolation Forest. Searching for alternative...")
    # Alternative: X_train_raw, X_test, y_train_raw, y_test = train_test_split
    for i, cell in enumerate(nb.get("cells", [])):
        source_str = "".join(cell.get("source", []))
        if "X_train, X_test, y_train, y_test = train_test_split" in source_str:
            insert_idx = i
            break
    if insert_idx != -1 and iso_forest_cell:
        nb["cells"].insert(insert_idx + 1, iso_forest_cell)
        print("Moved Isolation Forest guardrail strictly to after Section 7 Train/Test split (alternative point).")


with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("V8 Notebook successfully cleaned and repaired!")
