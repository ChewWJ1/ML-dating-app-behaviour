import json
import ast

notebook_path = r"notebooks\ML_dating_app_behaviour V8.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source)
        except Exception as e:
            print(f"Cell {i} failed to compile: {e}")
