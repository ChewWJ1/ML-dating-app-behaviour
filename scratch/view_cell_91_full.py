import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][91]
with open('scratch/cell_91_full.txt', 'w', encoding='utf-8') as out_f:
    out_f.write("".join(cell['source']))
print("Done writing Cell 91 to file.")
