import json
import re

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Isolation Forest cell
iso_forest_cell = None
iso_forest_idx = -1
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "IsolationForest(" in source and "iso_forest.fit_predict" in source:
            iso_forest_cell = cell
            iso_forest_idx = idx
            break

if iso_forest_idx != -1:
    nb["cells"].pop(iso_forest_idx)

# Find Section 5.1 RobustScaler
section_5_1_idx = -1
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "RobustScaler" in source and "X_train = scaler.fit_transform" in source:
            section_5_1_idx = idx
            break

if section_5_1_idx != -1 and iso_forest_cell is not None:
    nb["cells"].insert(section_5_1_idx + 1, iso_forest_cell)


for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        # 1. Section 17 summary artifacts
        new_source = []
        for line in cell.get("source", []):
            line = re.sub(r'^# (\d+)\. ', r'\1. ', line)
            if line.endswith('\\n"\n'):
                line = line[:-4] + "\n"
            new_source.append(line)
        cell["source"] = new_source

    elif cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        new_source = []
        
        # 2. Optuna missing seed
        if "optuna.create_study(" in source:
            for line in cell["source"]:
                if "optuna.create_study(direction='maximize')" in line:
                    line = line.replace("optuna.create_study(direction='maximize')", "optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))")
                new_source.append(line)
            cell["source"] = new_source
            continue
            
        # 3. Replace weak paired t-test
        if "stats.ttest_rel" in source or "stats.wilcoxon" in source:
            if "model1_scores" in source or "t_stat" in source:
                new_code = [
                    "# 9.6 Statistical Significance Testing\n",
                    "from scipy.stats import friedmanchisquare\n",
                    "\n",
                    "print('-' * 60)\n",
                    "print('Statistical Significance Test (Friedman test across all models):')\n",
                    "data = [cv_results[m] for m in cv_results]\n",
                    "stat, p_value = friedmanchisquare(*data)\n",
                    "print(f'Friedman p-value = {p_value:.4f}')\n",
                    "if p_value < 0.05:\n",
                    "    print('Result: The performance differences are statistically significant (p < 0.05).')\n",
                    "    try:\n",
                    "        import scikit_posthocs as sp\n",
                    "        print('Running Nemenyi post-hoc test for pairwise comparisons...')\n",
                    "        posthoc_res = sp.posthoc_nemenyi(data)\n",
                    "        print(posthoc_res)\n",
                    "    except ImportError:\n",
                    "        print('scikit-posthocs not installed, skipping Nemenyi test.')\n",
                    "else:\n",
                    "    print('Result: The performance differences are NOT statistically significant (p >= 0.05).')\n",
                    "print('-' * 60)\n"
                ]
                cell["source"] = new_code
            continue

        # 4. Precision-Recall curve
        if "calibrated_clf.predict_proba(X_eval)[:, 1]" in source and "champion_name" in source:
            new_code = []
            for line in cell["source"]:
                if "target_dict[champion_name]['y_pred'] = calibrated_clf.predict(X_eval)" in line:
                    new_code.extend([
                        "    from sklearn.metrics import precision_recall_curve, average_precision_score\n",
                        "    import numpy as np\n",
                        "    target_dict[champion_name]['y_prob'] = calibrated_clf.predict_proba(X_eval)[:, 1]\n",
                        "    prec, rec, thresholds = precision_recall_curve(y_eval, target_dict[champion_name]['y_prob'])\n",
                        "    f1_scores = 2*prec*rec / (prec+rec+1e-8)\n",
                        "    best_thresh = thresholds[np.argmax(f1_scores)]\n",
                        "    target_dict[champion_name]['y_pred'] = (target_dict[champion_name]['y_prob'] >= best_thresh).astype(int)\n",
                        "    print(f'\\n✅ Optimized Champion Threshold (PR Curve max F1): {best_thresh:.4f}')\n"
                    ])
                elif "target_dict[champion_name]['y_prob'] = calibrated_clf.predict_proba(X_eval)[:, 1]" in line:
                    pass # Handled above
                else:
                    new_code.append(line)
            cell["source"] = new_code
            continue
            
        # 5. Expand demographic parity
        if "roc_auc_score" in source and "gender" in source and "Demographic Parity" in source:
            # We rewrite the fairness block to use fairlearn MetricFrame
            new_code = [
                "# Per-group accuracy breakdown (Testing for Demographic Parity)\n",
                "from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate\n",
                "from sklearn.metrics import roc_auc_score\n",
                "import numpy as np\n",
                "\n",
                "print('ROC-AUC Breakdown by Demographic Groups:')\n",
                "print('-' * 40)\n",
                "\n",
                "# Use predictions from the first baseline model to demonstrate parity check\n",
                "y_prob_demo = tuned_results[best_name]['y_prob'] if best_name in tuned_results else results[best_name]['y_prob']\n",
                "y_pred_demo = (y_prob_demo >= 0.5).astype(int) # Standard threshold for TPR/FPR\n",
                "\n",
                "def tpr(y_true, y_prob): return true_positive_rate(y_true, y_prob >= 0.5)\n",
                "def fpr(y_true, y_prob): return false_positive_rate(y_true, y_prob >= 0.5)\n",
                "\n",
                "for attr in ['gender', 'sexual_orientation']:\n",
                "    print(f'\\n--- Fairness Audit for {attr} ---')\n",
                "    sensitive_col = df_raw.iloc[X_test.index][attr]\n",
                "    mf = MetricFrame(\n",
                "        metrics={'TPR': tpr, 'FPR': fpr, 'ROC-AUC': roc_auc_score},\n",
                "        y_true=y_test, y_pred=y_prob_demo,\n",
                "        sensitive_features=sensitive_col\n",
                "    )\n",
                "    print(mf.by_group)\n"
            ]
            cell["source"] = new_code
            continue
            
        # 7. FLAML uses X_train_smote
        if "automl.fit(" in source and "FLAML" in source:
            for line in cell["source"]:
                if "automl.fit(" in line:
                    if "X_train," in line:
                        line = line.replace("X_train,", "X_train_smote,")
                    if "y_train," in line:
                        line = line.replace("y_train,", "y_train_smote,")
                new_source.append(line)
            cell["source"] = new_source
            continue

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Applied methodology upgrades successfully!")
