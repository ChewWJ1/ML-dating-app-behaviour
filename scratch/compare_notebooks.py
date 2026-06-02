import json
import os

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"

def inspect_nb(filename):
    path = os.path.join(root_dir, "notebooks", filename)
    print(f"\nNotebook: {filename}")
    if not os.path.exists(path):
        print("Does not exist!")
        return None
    print(f"Size: {os.path.getsize(path):,} bytes")
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        print(f"Total cells: {len(cells)}")
        cell_types = {}
        for c in cells:
            t = c.get("cell_type")
            cell_types[t] = cell_types.get(t, 0) + 1
        print(f"Cell types: {cell_types}")
        return cells
    except Exception as e:
        print(f"Error: {e}")
        return None

cells_patched = inspect_nb("ML_dating_app_behaviour V8_patched_v4.ipynb")
cells_82 = inspect_nb("ML_dating_app_behaviour V8.2.ipynb")
cells_8 = inspect_nb("ML_dating_app_behaviour V8.ipynb")
