import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell containing FGSM
for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "FGSM" in source or "fgsm" in source:
            print(f"Cell {idx} contains FGSM/fgsm:")
            print("====================================")
            # Print first 60 lines of the source
            lines = cell.get("source", [])
            for line_idx, line in enumerate(lines[:60]):
                print(f"{line_idx+1}: {line}", end="")
            if len(lines) > 60:
                print("\n... and more lines")
            print("====================================")
