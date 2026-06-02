import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        # Filter out IPython magics
        source_clean = "\n".join([line for line in source.split("\n") if not line.startswith("!") and not line.startswith("%")])
        try:
            compile(source_clean, f"cell_{i}", "exec")
        except SyntaxError as e:
            print(f"SyntaxError in cell {i}: {e}")
            print("--- Snippet ---")
            print("\n".join(source_clean.split("\n")[max(0, e.lineno-3):e.lineno+2]))
            print("---------------")
            exit(1)

print("✅ No syntax errors found in any code cell!")
