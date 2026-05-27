import json

with open(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if 'gnn_gat.joblib' in source_text:
            print(f"Cell Index: {idx}")
            print(source_text[:1000])
            print("=" * 50)
