import json

nb_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V7_Strict.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            # 1. Optuna Objective
            line = line.replace("return f1_score(y_te, preds)", "from sklearn.metrics import matthews_corrcoef\n        return matthews_corrcoef(y_te, preds)")
            line = line.replace("print(f\"  Best F1 score: {study.best_value:.4f}\")", "print(f\"  Best MCC score: {study.best_value:.4f}\")")
            
            # 2. RETUNE_MODELS
            line = line.replace("RETUNE_MODELS = False       \n", "RETUNE_MODELS = True        \n")
            
            # 3. RandomizedSearchCV
            line = line.replace("scoring='f1',", "scoring='matthews_corrcoef',")
            line = line.replace("print(f'  Best CV F1:     {search.best_score_:.4f}')", "print(f'  Best CV MCC:    {search.best_score_:.4f}')")
            
            new_source.append(line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
