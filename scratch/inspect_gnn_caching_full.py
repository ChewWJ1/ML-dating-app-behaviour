import json

with open(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][117]
print("=== CELL 117 FULL SOURCE ===")
print("".join(cell['source']))
print("=" * 50)
