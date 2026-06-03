import json

in_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"
out_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\v8_dump.py"

with open(in_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(out_path, "w", encoding="utf-8") as out:
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            for line in cell.get("source", []):
                out.write(line)
            out.write("\n\n")
        elif cell["cell_type"] == "markdown":
            for line in cell.get("source", []):
                out.write("# " + line)
            out.write("\n\n")

print("V8 dump successfully generated!")
