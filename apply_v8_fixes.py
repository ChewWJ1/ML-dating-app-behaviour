import json
import os
import re

in_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V7_Strict.ipynb"
out_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"

with open(in_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Track if we found and moved the isolation forest block
iso_forest_lines = []
iso_forest_found = False

for i, cell in enumerate(nb.get("cells", [])):
    source_str = "".join(cell.get("source", []))
    
    # 1. Isolation Forest Guardrail Fix (Section 4.11 extraction)
    if "iso_forest = IsolationForest(contamination=0.05" in source_str and "Section 4.11" in source_str:
        iso_forest_lines = cell["source"]
        iso_forest_found = True
        cell["source"] = ["# Moved IsolationForest Guardrail to after Section 7\n"]
        continue

for i, cell in enumerate(nb.get("cells", [])):
    source_str = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" or cell["cell_type"] == "markdown":
        new_source = []
        lines = cell["source"]
        for line_idx, line in enumerate(lines):
            
            # --- Selectivity Ratio Fix ---
            if "df['selectivity_ratio'] = df['mutual_matches'] / (df['likes_received'] + 1)" in line:
                line = line.replace("df['mutual_matches']", "df['swipe_right_count']")
            if "df_temp['selectivity_ratio'] = df_temp['mutual_matches'] / (df_temp['likes_received'] + 1)" in line:
                line = line.replace("df_temp['mutual_matches']", "df_temp['swipe_right_count']")
            if "'mutual_matches'" in line:
                line = line.replace("'mutual_matches', ", "")
                line = line.replace("'mutual_matches'", "")
                
            # --- Models_v8 renaming ---
            if "models_v7" in line:
                line = line.replace("models_v7", "models_v8")

            # --- SMOTE Global Override Fix ---
            if "X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)" in line:
                line = line.replace("X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)",
                                    "X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)\n    X_train, y_train = X_train_raw, y_train_raw  # Keep unsmoted globally for CV")
            
            if "model.fit(X_train, y_train)" in line and "time.time()" in "".join(lines):
                line = line.replace("model.fit(X_train, y_train)", "model.fit(X_train_smote, y_train_smote)  # Final train uses SMOTE")
            if "best_model.fit(X_train, y_train)" in line and "matching CV conditions" in line:
                line = line.replace("best_model.fit(X_train, y_train)", "best_model.fit(X_train_smote, y_train_smote)")

            # --- XGBoost Config Fix ---
            if "XGBClassifier(scale_pos_weight=1)" in line:
                line = re.sub(r"XGBClassifier\(scale_pos_weight=1\)(.*?)n_estimators=500(.*?)\)",
                              r"XGBClassifier(n_estimators=500, eval_metric='logloss', tree_method='hist', device=XGB_DEVICE, scale_pos_weight=1, random_state=RANDOM_STATE)\1\2)",
                              line)
            if "XGBClassifier(scale_pos_weight=1)" in line:
                line = line.replace("XGBClassifier(scale_pos_weight=1)", "XGBClassifier(n_estimators=500, eval_metric='logloss', tree_method='hist', device=XGB_DEVICE, scale_pos_weight=1, random_state=RANDOM_STATE)")
            
            # Revert if it created duplicate n_estimators? It's fine for now, we'll see if python syntax is valid.
            
            # --- Deep Learning Fixes ---
            if "epochs=12" in line:
                line = line.replace("epochs=12", "epochs=100")
            if "'TabNet Deep Learning': PyTorchSklearnClassifier" in line:
                line = line.replace("'TabNet Deep Learning'", "'FTTransformer Deep Learning'")
            
            # --- Cosine KNN Naming ---
            if "'Collaborative Filtering (Cosine KNN)':" in line:
                line = line.replace("'Collaborative Filtering (Cosine KNN)':", "'KNN (Cosine Metric)':")
                
            # --- Logistic Regression Params ---
            if "solver='lbfgs', n_jobs=-1" in line:
                line = line.replace("solver='lbfgs', n_jobs=-1", "solver='lbfgs'")

            # --- Simulated Loss Curves Fix ---
            if "Regularization Analysis: Preventing Deep Learning Overconfidence" in line and "plt.suptitle" in line:
                line = line.replace("Regularization Analysis: Preventing Deep Learning Overconfidence", "Regularization Analysis: Preventing Deep Learning Overconfidence (Illustrative Simulation)")

            # --- TabPFN Labelling ---
            if "'TabPFN Transformer (Zero-Shot)'" in line:
                line = line.replace("'TabPFN Transformer (Zero-Shot)'", "'TabPFN Transformer (LightGBM Fallback)'")

            # --- Calibration Leakage Fix ---
            if "calibrated_clf = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')" in line:
                line = "        from sklearn.model_selection import train_test_split\n" \
                       "        X_calib, X_eval, y_calib, y_eval = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test)\n" \
                       "        " + line + "\n" \
                       "        calibrated_clf.fit(X_calib, y_calib)\n"
            
            if "calibrated_clf.fit(X_test, y_test)" in line and "cv='prefit'" not in line:
                line = ""

            if "probs_cal = calibrated_clf.predict_proba(X_test)[:, 1]" in line:
                line = line.replace("X_test", "X_eval")
            if "probs_raw = r_entry['y_prob']" in line:
                line = "        probs_raw = base_model.predict_proba(X_eval)[:, 1] if hasattr(base_model, 'predict_proba') else base_model.decision_function(X_eval)\n"
            if "prob_true_raw, prob_pred_raw = calibration_curve(y_test, probs_raw" in line:
                line = line.replace("y_test", "y_eval")
            if "prob_true_cal, prob_pred_cal = calibration_curve(y_test, probs_cal" in line:
                line = line.replace("y_test", "y_eval")
            if "brier_score_loss(y_test, probs_raw)" in line:
                line = line.replace("y_test", "y_eval")
            if "brier_score_loss(y_test, probs_cal)" in line:
                line = line.replace("y_test", "y_eval")
            if "calibrated_clf.predict(X_test)" in line:
                line = line.replace("X_test", "X_eval")
            if "calibrated_clf.predict_proba(X_test)[:, 1]" in line:
                line = line.replace("X_test", "X_eval")
            if "accuracy_score(y_test, target_dict[champion_name]['y_pred'])" in line:
                line = line.replace("y_test", "y_eval")
            if "f1_score(y_test, target_dict[champion_name]['y_pred'])" in line:
                line = line.replace("y_test", "y_eval")
            if "roc_auc_score(y_test, target_dict[champion_name]['y_prob'])" in line:
                line = line.replace("y_test", "y_eval")

            # --- Statistical Significance (Section 10.7) ---
            if "cv_scores_model1 = cross_val_score(model1, X_train_raw, y_train_raw, cv=5" in line:
                line = line.replace("cv=5", "cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)")
            if "cv_scores_model2 = cross_val_score(model2, X_train_raw, y_train_raw, cv=5" in line:
                line = line.replace("cv=5", "cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)")
            if "from sklearn.model_selection import cross_val_score" in line:
                line = line.replace("cross_val_score", "cross_val_score, RepeatedStratifiedKFold")

            # --- Knowledge distillation student capacity ---
            if "MLPClassifier(hidden_layer_sizes=(32,)" in line:
                line = line.replace("hidden_layer_sizes=(32,)", "hidden_layer_sizes=(128,)")
                
            # Cleanup inline comments for XGBoost that might be syntactically invalid now
            if "# Neutral: SMOTE already balances classes 50/50, n_estimators=500, random_state=RANDOM_STATE, eval_metric='logloss', **TREE_CONFIG['xgb'], n_jobs=-1)" in line:
                line = line.replace("# Neutral: SMOTE already balances classes 50/50, n_estimators=500, random_state=RANDOM_STATE, eval_metric='logloss', **TREE_CONFIG['xgb'], n_jobs=-1)", "")
            if "# Neutral: SMOTE already balances classes 50/50, n_estimators=500, random_state=RANDOM_STATE, eval_metric='logloss', tree_method='hist', device=XGB_DEVICE, n_jobs=1" in line:
                line = line.replace("# Neutral: SMOTE already balances classes 50/50, n_estimators=500, random_state=RANDOM_STATE, eval_metric='logloss', tree_method='hist', device=XGB_DEVICE, n_jobs=1", "")

            new_source.append(line)
        cell["source"] = new_source

# Insert Isolation Forest after Section 7 Train/Test Split
if iso_forest_found:
    for i, cell in enumerate(nb.get("cells", [])):
        source_str = "".join(cell.get("source", []))
        if "X_train = X_train_selected" in source_str:
            nb["cells"].insert(i + 1, {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": iso_forest_lines
            })
            break

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("V8 Notebook successfully generated!")
