import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 93 or cell that builds comparison table
for idx, cell in enumerate(nb.get("cells", [])):
    source = "".join(cell.get("source", []))
    if "Model Comparison (sorted" in source or "Model Comparison Table" in source or "pd.DataFrame(results)" in source or "pd.DataFrame({" in source:
        outputs = cell.get("outputs", [])
        if outputs:
            print(f"\n====================================")
            print(f"Cell {idx} Output:")
            print("====================================")
            for out in outputs:
                if out.get("output_type") == "stream":
                    print("".join(out.get("text", [])))
                elif "data" in out and "text/plain" in out["data"]:
                    print("".join(out["data"]["text/plain"]))
            print("====================================")
