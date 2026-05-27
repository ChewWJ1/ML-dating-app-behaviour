import json

nb_path = "notebooks/ML_dating_app_behaviour V5.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

sota_cells = [33, 54, 90, 115, 152, 167, 176]
for cell_idx in sota_cells:
    if cell_idx < len(nb['cells']):
        cell = nb['cells'][cell_idx]
        code = "".join(cell['source'])
        print(f"=== Cell {cell_idx} Source Snippet ===")
        # Find lines with savefig
        save_lines = [line.strip() for line in code.split('\n') if "savefig" in line or "plt.show" in line or "os.path.join" in line]
        for line in save_lines:
            print(f"  {line}")
        print()
