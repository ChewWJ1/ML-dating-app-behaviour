import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    source = "".join(cell.get("source", []))
    if "XGBoost (Tuned)" in source:
        print(f"Cell {idx} ({cell.get('cell_type')}):")
        # Print lines that contain the keyword or modify results
        for line_num, line in enumerate(cell.get("source", [])):
            if "XGBoost" in line or "results" in line:
                print(f"  L{line_num+1}: {line.strip()[:150]}")
