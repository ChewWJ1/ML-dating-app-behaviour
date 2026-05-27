import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][167]
with open('scratch/cell_167_source.txt', 'w', encoding='utf-8') as out_f:
    out_f.write("".join(cell['source']))
print("Wrote Cell 167 source to scratch/cell_167_source.txt")
