import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V3.ipynb"

# Read the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell and replace it
found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell.get("source", [])
        for idx, line in enumerate(source):
            if "mutual_info_classif(X, y, random_state=RANDOM_STATE, n_jobs=-1)" in line:
                source[idx] = line.replace(", n_jobs=-1", "")
                found = True
                break
        if found:
            break

if found:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook cell 28 updated successfully.")
else:
    print("Could not find the target cell in cell 28.")
