import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Search for the cell containing 'Running hyperparameter tuning'
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Running hyperparameter tuning" in source:
            print(f"Tuning Cell Index: {idx}")
            with open('scratch/tuning_cell_source.txt', 'w', encoding='utf-8') as out_f:
                out_f.write(source)
            print("Wrote tuning cell source to scratch/tuning_cell_source.txt")
            break
