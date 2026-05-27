import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

indices = [35, 117, 174, 176]
out_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\scratch\heavy_source_output.txt"
with open(out_path, 'w', encoding='utf-8') as out_f:
    for idx in indices:
        cell = nb.get('cells', [])[idx]
        out_f.write(f"==========================================\n")
        out_f.write(f"Cell Index: {idx}\n")
        out_f.write(f"==========================================\n")
        out_f.write("".join(cell.get('source', [])))
        out_f.write("\n\n")
print("Done writing to heavy_source_output.txt")
