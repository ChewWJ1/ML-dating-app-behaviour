import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Searching all cells for 'MapieClassifier'...")
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "MapieClassifier" in source:
            print(f"Cell Index {idx} contains 'MapieClassifier'")
            print("--- Snippet ---")
            lines = source.split('\n')
            for i in range(min(15, len(lines))):
                print(lines[i])
            print("-" * 30)
