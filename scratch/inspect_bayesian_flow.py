import json

with open(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in range(150, 163):
    cell = nb['cells'][idx]
    print(f"=== Cell {idx} ({cell['cell_type'].upper()}) ===")
    text = "".join(cell.get('source', []))
    print(text[:800] + ("..." if len(text) > 800 else ""))
    print("-" * 50)
