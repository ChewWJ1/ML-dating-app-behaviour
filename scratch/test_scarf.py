import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if 'SCARF' in source:
            clean_source = source.encode('ascii', 'ignore').decode('ascii')
            print(f"MD Cell {idx}: {clean_source[:100]}")
    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'scarf' in source.lower():
            clean_source = source.encode('ascii', 'ignore').decode('ascii')
            print(f"Code Cell {idx}: {clean_source[:100]}")
