import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V4.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 146 is mapie
cell = nb['cells'][146]
with open('scratch/v4_cell_146_source.txt', 'w', encoding='utf-8') as out_f:
    out_f.write("".join(cell['source']))
print("Wrote V4 Cell 146 source to scratch/v4_cell_146_source.txt")
