import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb.get("cells", [])[129]
print("Cell 129 (code) - Lines 50 onwards:")
print("====================================")
lines = cell.get("source", [])
for line_idx, line in enumerate(lines[49:]):
    print(f"{line_idx+50}: {line}", end="")
print("\n====================================")
