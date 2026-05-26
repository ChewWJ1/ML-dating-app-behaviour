"""
Final pass: Fix remaining documentation text using direct JSON manipulation.
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

    for i, cell in enumerate(cells):
        if cell.get("cell_type") not in ("code", "markdown"):
            continue
        
        new_source = []
        cell_changed = False
        for line in cell.get("source", []):
            # Fix the Double-Path description
            if "Double-Path Directory Routing" in line:
                line = "* **Smart SVM Bypass + Fresh Retrain Strategy:** Dedicated dual-path directory routing reads the computationally heavy pre-trained SVM weights from `../models/` (original V1 checkpoints) while **always retraining all 13 other models from scratch** on every run. All new results are saved to `../models_advanced/`, ensuring complete isolation from original files and guaranteeing fresh, reproducible model weights on every execution.\n"
                cell_changed = True
                print(f"  Cell {i}: Fixed Double-Path line")
            
            # Fix the "How it works" description
            if "loads them instantly in 0.1 seconds" in line:
                line = "**How it works:** When a teammate opens this notebook and clicks **\"Run All\"**, the code **always retrains all 13 non-SVM models from scratch** using the best available hardware (NVIDIA CUDA, AMD DirectML, or CPU). Only the computationally expensive SVM model is bypassed by loading its pre-trained weights from `../models/`. All fresh results are then saved to `../models_advanced/` for complete reproducibility.\n"
                cell_changed = True
                print(f"  Cell {i}: Fixed How it works line")
            
            new_source.append(line)
        
        if cell_changed:
            cell["source"] = new_source
            changes += 1

    print(f"\nChanges: {changes}")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Done!")

if __name__ == "__main__":
    main()
