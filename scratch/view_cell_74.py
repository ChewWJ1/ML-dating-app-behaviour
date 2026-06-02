import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb.get("cells", [])[74]
print("Cell 74 (code):")
print("====================================")
for line_idx, line in enumerate(cell.get("source", [])):
    print(f"{line_idx+1}: {line}", end="")
print("\n====================================")
