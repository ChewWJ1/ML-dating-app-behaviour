import json

notebook_path = r"notebooks\ML_dating_app_behaviour V8.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell.get("source", [])
        if any("sv_2d" in line for line in source) and any("cache_shap" in line for line in source):
            new_source = []
            for line in source:
                if line.startswith("            joblib.dump({"):
                    line = "        " + line[12:]
                elif line.startswith("                'values': sv_2d,"):
                    line = "            " + line[16:]
                elif line.startswith("                'interactions': shap_interaction_values,"):
                    line = "            " + line[16:]
                elif line.startswith("                'base_values': shap_values_obj.base_values"):
                    line = "            " + line[16:]
                elif line.startswith("            }, cache_shap)"):
                    line = "        " + line[12:]
                new_source.append(line)
            cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Indentation fixed.")
