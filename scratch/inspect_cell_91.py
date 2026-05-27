import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 85 is where models are trained and results are stored (in V5, after our injections Cell 91 may have shifted, let's search)
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'results[name] = {' in source:
            print(f"--- Cell {idx} ---")
            lines = source.split("\n")
            for i, line in enumerate(lines):
                if 'results[name] = {' in line:
                    # Print next 10 lines
                    for j in range(i, min(i+15, len(lines))):
                        print(lines[j])
                    break
            break
