import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Search cells 110 to 125 for metrics calculation
for idx in range(109, 126):
    cell = nb.get("cells", [])[idx]
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    if cell_type == "code":
        print(f"\n====================================")
        print(f"Cell {idx} (code):")
        print("====================================")
        for line_num, line in enumerate(cell.get("source", [])):
            if any(term in line for term in ["score", "acc", "f1", "auc", "precision", "recall", "report", "print"]):
                print(f"  L{line_num+1}: {line.strip()[:120]}")
