import os
import json

notebooks_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks"
print("Scanning all notebooks for 'calib'...")

for file in os.listdir(notebooks_dir):
    if file.endswith('.ipynb'):
        path = os.path.join(notebooks_dir, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            for idx, cell in enumerate(nb.get('cells', [])):
                if cell.get('cell_type') == 'code':
                    source = "".join(cell.get('source', []))
                    if "calib" in source or "conformal" in source:
                        print(f"Notebook: {file} | Cell Index: {idx} contains match")
        except Exception as e:
            print(f"Error reading {file}: {e}")
