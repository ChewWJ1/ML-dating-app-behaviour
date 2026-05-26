import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

queries = ['class ', 'MLP', 'nn.Module', 'SMOTE', 'RobustScaler', 'optuna', 'Friedman', 'conformal', 'Isotonic', 'Calibrated', 'PyTorch', 'deep', 'mlp']

lines_out = []
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = "".join(cell['source'])
        for q in queries:
            if q.lower() in code.lower():
                lines_out.append(f"Cell {idx} contains '{q}': {code[:200].strip()}...\n")

with open('scratch/code_matches.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out))
print("Successfully wrote matches to scratch/code_matches.md")
