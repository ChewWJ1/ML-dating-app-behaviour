import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Extracting cell outputs from V8_patched_v4...")

with open("scratch/v8_actual_outputs.txt", "w", encoding="utf-8") as out_f:
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        
        # Write headings or code block indicators
        if cell_type == "markdown":
            if source.strip().startswith("#"):
                out_f.write(f"\n\n======================\n[Cell {idx} Markdown] {source.strip()[:100]}\n======================\n")
        elif cell_type == "code":
            outputs = cell.get("outputs", [])
            if outputs:
                out_f.write(f"\n[Cell {idx} Code Output]\n")
                for out in outputs:
                    if out.get("output_type") == "stream":
                        out_f.write("".join(out.get("text", [])))
                    elif "data" in out:
                        if "text/plain" in out["data"]:
                            out_f.write("".join(out["data"]["text/plain"]))
                        elif "text/html" in out["data"]:
                            out_f.write("[HTML DATA]\n")

print("Done! Extracted to scratch/v8_actual_outputs.txt")
