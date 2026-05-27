import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'results[' in source and '=' in source:
            print(f"--- Cell {idx} ---")
            # print lines containing results[
            for line in cell['source']:
                if 'results[' in line:
                    print(line.strip())
