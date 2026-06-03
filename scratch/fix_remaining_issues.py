import json
import re

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            # 1. Remove duplicate FTTransformer
            if "'FTTransformer Deep Learning': PyTorchSklearnClassifier" in line:
                continue
            
            # 2. Upgrade CV from 5 to 10 folds
            if "cv=5" in line:
                line = line.replace("cv=5", "cv=10")
                
            # 3. Swap ttest_rel for wilcoxon
            if "stats.ttest_rel" in line:
                line = line.replace("stats.ttest_rel", "stats.wilcoxon")
                
            new_source.append(line)
        cell["source"] = new_source
        
    elif cell["cell_type"] == "markdown":
        new_source = []
        for line in cell["source"]:
            if "t-test" in line or "ttest" in line:
                line = line.replace("t-test", "Wilcoxon signed-rank test").replace("ttest", "Wilcoxon test")
            new_source.append(line)
        cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed duplicate FTTransformer and statistical tests.")
