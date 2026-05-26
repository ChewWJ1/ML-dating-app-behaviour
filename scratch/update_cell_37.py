import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V3.ipynb"

# Read the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell and replace it
found = False
for cell in nb["cells"]:
    if cell.get("id") == "90ed5ee9":
        # Modify the source array
        source = cell["source"]
        # Find where SVC is imported
        for idx, line in enumerate(source):
            if "from sklearn.svm import SVC" in line:
                source.insert(idx + 1, "from sklearn.neural_network import MLPClassifier\n")
                source.insert(idx + 2, "from lightgbm import LGBMClassifier\n")
                source.insert(idx + 3, "from catboost import CatBoostClassifier\n")
                source.insert(idx + 4, "from imblearn.ensemble import BalancedRandomForestClassifier\n")
                found = True
                break
        if found:
            break

if found:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Imports cell 37 updated successfully.")
else:
    print("Could not find the target import cell 37.")
