import json

path = "notebooks/ML_dating_app_behaviour V3.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open("scratch/v3_cells_inspect.txt", "w", encoding="utf-8") as out:
    out.write("=== Cell 75 (Baseline Training) ===\n")
    out.write("".join(nb['cells'][75]['source']))
    out.write("\n\n=== Cell 85 (Cross-Validation) ===\n")
    out.write("".join(nb['cells'][85]['source']))
    out.write("\n\n=== Cell 96 (Hyperparameter Tuning) ===\n")
    out.write("".join(nb['cells'][96]['source']))
    out.write("\n")

print("V3 cells extracted successfully!")
