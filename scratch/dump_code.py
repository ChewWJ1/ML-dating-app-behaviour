import json

nb_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V7_Strict.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("notebook_code_dump.py", "w", encoding="utf-8") as f:
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            f.write("\n" + "#" * 80 + "\n")
            for line in cell.get("source", []):
                f.write("# " + line.strip() + "\n")
            f.write("#" * 80 + "\n")
        elif cell["cell_type"] == "code":
            f.write("\n")
            for line in cell.get("source", []):
                f.write(line)
            f.write("\n")
