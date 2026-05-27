import json
import re

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

heavy_keywords = [
    r"RandomForestClassifier", r"RandomForestRegressor", r"Optuna", r"optuna", 
    r"SMOTE", r"KNeighborsClassifier", r"SVC", r"MLPClassifier", r"GAT", 
    r"dice_ml", r"T-Learner", r"pycaret", r"PyCaret", r"shap", r"SHAP", 
    r"TabNet", r"tabnet", r"tabpfn", r"TabPFN", r"fit\(", r"GridSearchCV", 
    r"RandomizedSearchCV", r"cross_val_score", r"StackingClassifier", 
    r"kneighbors_graph"
]

results = []
for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        matched = []
        for kw in heavy_keywords:
            if re.search(kw, source):
                matched.append(kw)
        
        if matched:
            has_joblib = "joblib" in source
            results.append({
                "index": idx,
                "matches": matched,
                "has_joblib": has_joblib,
                "first_few_lines": "\n".join(cell.get('source', [])[:4])
            })

out_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\scratch\scan_heavy_output.txt"
with open(out_path, 'w', encoding='utf-8') as out_f:
    out_f.write("Scanning notebook cells for heavy processing operations...\n\n")
    for res in results:
        out_f.write(f"Cell {res['index']}:\n")
        out_f.write(f"  Matches: {res['matches']}\n")
        out_f.write(f"  Has joblib: {res['has_joblib']}\n")
        out_f.write(f"  Snippet:\n{res['first_few_lines']}\n")
        out_f.write("-" * 50 + "\n")
print("Done scanning notebook heavy operations.")
