import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_84_code = "".join(nb['cells'][84]['source'])

with open('scratch/cell_84_source.py', 'w', encoding='utf-8') as f:
    f.write(cell_84_code)

print("Wrote Cell 84 code to scratch/cell_84_source.py")
