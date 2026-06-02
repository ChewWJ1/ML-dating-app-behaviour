import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Section 12 or tuning cell
for idx, cell in enumerate(nb.get("cells", [])):
    source = "".join(cell.get("source", []))
    if "tuned_results.joblib" in source or "RandomizedSearchCV" in source:
        if cell.get("cell_type") == "code":
            print(f"\n====================================")
            print(f"Cell {idx} (code) containing tuning/caching:")
            print("====================================")
            lines = cell.get("source", [])
            for line_idx, line in enumerate(lines[:60]):
                print(f"{line_idx+1}: {line}", end="")
            if len(lines) > 60:
                print("\n... and more lines")
            print("====================================")
