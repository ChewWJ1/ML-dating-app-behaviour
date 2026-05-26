"""
Third pass: Fix the remaining duplicate documentation text.
"""
import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "ML_dating_app_behaviour V3.ipynb"
)

def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    changes = 0

    old_text = "* **Double-Path Directory Routing:** Dedicated dual-path directory routing has been implemented, reading the computationally heavy pre-trained SVM from `../models/` while saving the new training runs dynamically to `../models_advanced/` (or `../models_advanced/` for the bypass pipeline), ensuring 100% thread-safety and protecting original files.\\n"
    new_text = "* **Smart SVM Bypass + Fresh Retrain Strategy:** Dedicated dual-path directory routing reads the computationally heavy pre-trained SVM weights from `../models/` (original V1 checkpoints) while **always retraining all 13 other models from scratch** on every run. All new results are saved to `../models_advanced/`, ensuring complete isolation from original files and guaranteeing fresh, reproducible model weights on every execution.\\n"

    old_how = '**How it works:** When a teammate opens this notebook and clicks **\\"Run All\\"**, the code automatically detects these `.joblib` files on disk. If found, it **loads them instantly in 0.1 seconds** instead of running the training algorithms, completing the entire notebook in **less than 15 seconds!**\\n'
    new_how = '**How it works:** When a teammate opens this notebook and clicks **\\"Run All\\"**, the code **always retrains all 13 non-SVM models from scratch** using the best available hardware (NVIDIA CUDA, AMD DirectML, or CPU). Only the computationally expensive SVM model is bypassed by loading its pre-trained weights from `../models/`. All fresh results are then saved to `../models_advanced/` for complete reproducibility.\\n'

    for i, cell in enumerate(cells):
        if cell.get("cell_type") in ("code", "markdown"):
            new_source = []
            cell_changed = False
            for line in cell.get("source", []):
                if old_text in line:
                    line = line.replace(old_text, new_text)
                    cell_changed = True
                if old_how in line:
                    line = line.replace(old_how, new_how)
                    cell_changed = True
                new_source.append(line)
            if cell_changed:
                cell["source"] = new_source
                print(f"  Cell {i}: Fixed documentation text")
                changes += 1

    print(f"\nChanges: {changes}")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Done!")

if __name__ == "__main__":
    main()
