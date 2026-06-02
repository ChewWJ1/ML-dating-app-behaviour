import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.2.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Iterate and patch cells
modified = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        new_source = []
        for line in source:
            new_line = line
            
            # 1. Fix Cross Validation and Learning Curve Bottleneck
            if "cv=10, scoring='roc_auc', n_jobs=1" in new_line:
                new_line = new_line.replace("cv=10, scoring='roc_auc', n_jobs=1", "cv=5, scoring='roc_auc', n_jobs=-1")
                modified = True
            
            # 2. Fix Optuna Leakage - Step 1: Replace fitting on full X_tr with split
            if "clf.fit(X_tr, y_tr)" in new_line and "XGBClassifier" not in line:
                new_line = "        from sklearn.model_selection import train_test_split\n" + \
                           "        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)\n" + \
                           "        clf.fit(X_train_opt, y_train_opt)\n"
                modified = True
            
            # 2. Fix Optuna Leakage - Step 2: Replace prediction and evaluation on X_te/y_te with X_val_opt/y_val_opt
            if "preds = clf.predict(X_te)" in new_line:
                new_line = new_line.replace("preds = clf.predict(X_te)", "preds = clf.predict(X_val_opt)")
                modified = True
            
            if "return matthews_corrcoef(y_te, preds)" in new_line:
                new_line = new_line.replace("return matthews_corrcoef(y_te, preds)", "return matthews_corrcoef(y_val_opt, preds)")
                modified = True
                
            new_source.append(new_line)
        cell["source"] = new_source

if modified:
    # Save modified notebook safely
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        # Add newline at end of file for standard git format
        f.write("\n")
    print("Notebook successfully patched.")
else:
    print("No changes were necessary or patterns did not match.")
