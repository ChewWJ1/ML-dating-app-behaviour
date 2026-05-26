import json

nb_path = 'notebooks/ML_dating_app_behaviour V5.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines_out = []
for idx in range(80, len(nb['cells'])):
    cell = nb['cells'][idx]
    cell_type = cell['cell_type']
    source = cell['source']
    first_lines = "".join(source[:3]) if source else ""
    first_lines = first_lines.replace("\n", " ")
    lines_out.append(f"Cell {idx} ({cell_type}): {first_lines[:120]}")

with open('scratch/nb_cells_80_end.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out))
print("Successfully wrote range to scratch/nb_cells_80_end.md")
