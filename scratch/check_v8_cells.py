import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

if not os.path.exists(nb_path):
    print("Notebook not found!")
    sys.exit(1)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells in notebook: {len(cells)}")

current_heading = "None"
heading_start_idx = 0

sections_found = []

for idx, cell in enumerate(cells):
    cell_type = cell.get("cell_type")
    source = cell.get("source", [])
    source_str = "".join(source).strip()
    
    if cell_type == "markdown" and source_str.startswith("#"):
        # We found a new heading
        sections_found.append((idx, source_str.split("\n")[0]))

for idx, heading in sections_found:
    print(f"Cell {idx:3d}: {heading}")
