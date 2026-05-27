import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source_str = "".join(cell.get('source', []))
        # Check GNN cell
        if "DatingGAT" in source_str:
            print(f"Cell Index: {idx} (GNN cell)")
            print("  Has joblib load:", "joblib.load" in source_str)
        # Check DiCE cell
        if "dice_ml" in source_str or "dice" in source_str and "Counterfactual" in source_str:
            print(f"Cell Index: {idx} (DiCE Recourse cell)")
            print("  Has joblib load:", "joblib.load" in source_str)
        # Check DML cell
        if "Double Machine Learning" in source_str or "propensity" in source_str and "bootstrap" in source_str:
            print(f"Cell Index: {idx} (DML Causal cell)")
            print("  Has joblib load:", "joblib.load" in source_str)
        # Check Uplift cell
        if "Uplift" in source_str and "T-Learner" in source_str:
            print(f"Cell Index: {idx} (Uplift cell)")
            print("  Has joblib load:", "joblib.load" in source_str)
