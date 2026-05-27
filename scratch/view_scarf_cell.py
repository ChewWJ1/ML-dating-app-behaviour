import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Loading cached SCARF representations" in source:
            print(f"SCARF Cell Index: {idx}")
            with open('scratch/scarf_cell_source.txt', 'w', encoding='utf-8') as out_f:
                out_f.write(source)
            print("Wrote SCARF cell source to scratch/scarf_cell_source.txt")
            break
