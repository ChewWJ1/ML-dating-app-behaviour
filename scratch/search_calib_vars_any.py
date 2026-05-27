import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

out_path = 'scratch/calib_vars_search_any.txt'
with open(out_path, 'w', encoding='utf-8') as out_f:
    out_f.write("Searching for occurrences of X_calib in any context...\n")
    for idx, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if "X_calib" in source:
            out_f.write(f"Cell Index: {idx} ({cell.get('cell_type')})\n")
            out_f.write("--- FIRST 5 LINES ---\n")
            lines = source.split('\n')
            for j in range(min(5, len(lines))):
                out_f.write(f"  {lines[j]}\n")
            out_f.write("---------------------\n")
print("Done searching.")
