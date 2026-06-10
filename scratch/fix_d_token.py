import json

notebook_path = r"notebooks\ML_dating_app_behaviour V8.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell.get("source", [])
        if any("class FeatureTokenizer" in line for line in source):
            new_source = []
            for line in source:
                if line.lstrip().startswith("super().__init__()"):
                    new_source.append(line)
                    indent = line[:len(line) - len(line.lstrip())]
                    new_source.append(indent + "self.d_token = d_token\n")
                    continue
                if "torch.zeros(x_num.size(0), 0, d_token," in line:
                    line = line.replace("d_token", "self.d_token")
                new_source.append(line)
            cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("d_token fixed.")
