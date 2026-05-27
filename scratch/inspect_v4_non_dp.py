import json

nb_path = 'notebooks/ML_dating_app_behaviour V4.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if 'non_dp_predictions' in source:
        clean = source.encode('ascii', 'ignore').decode('ascii')
        print(f"Cell {idx} contains 'non_dp_predictions':")
        print(clean[:300])
        print("="*40)
