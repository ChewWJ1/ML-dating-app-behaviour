import json

nb_path = "notebooks/ML_dating_app_behaviour V5.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V5 notebook: {len(nb['cells'])}")

plot_cells = []
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = "".join(cell['source'])
        if "plt.show" in code or "plt.savefig" in code or "sns." in code:
            # Look for comments or headings before this cell or title in code
            plot_cells.append((idx, code.split('\n')[0][:80]))

print(f"\nFound {len(plot_cells)} code cells generating plots:")
for idx, first_line in plot_cells:
    print(f"Cell {idx}: {first_line}")
