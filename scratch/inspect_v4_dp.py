import json

nb_path = 'notebooks/ML_dating_app_behaviour V4.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'cache_dp' in source:
            with open('scratch/v4_dp_cell_source.py', 'w', encoding='utf-8') as f:
                f.write(source)
            print(f"Successfully wrote V4 Cell {idx} source to scratch/v4_dp_cell_source.py")
            break
