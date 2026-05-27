import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

variables = ['best_model', 'X_calib', 'y_calib']
out_path = 'scratch/calib_vars_search.txt'
with open(out_path, 'w', encoding='utf-8') as out_f:
    for var in variables:
        out_f.write(f"Searching for definition of '{var}'...\n")
        found = False
        for idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                if f"{var} =" in source or f"{var}=" in source or f"global {var}" in source:
                    out_f.write(f"  Defined in Cell Index: {idx}\n")
                    # Print first 5 lines of the cell
                    lines = source.split('\n')
                    for j in range(min(5, len(lines))):
                        out_f.write(f"    {lines[j]}\n")
                    found = True
        if not found:
            out_f.write(f"  ❌ No definition found for '{var}'!\n")
print("Done searching.")
