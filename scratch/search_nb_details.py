import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
nb_path = os.path.join(root_dir, "notebooks", "ML_dating_app_behaviour V8_patched_v4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Collect all code lines
code_lines = []
for idx, cell in enumerate(nb.get("cells", [])):
    cell_type = cell.get("cell_type")
    source = cell.get("source", [])
    for line_num, line in enumerate(source):
        code_lines.append((idx, cell_type, line_num + 1, line))

search_terms = [
    r"IPW", r"propensity", r"inverse probability", r"T-Learner",
    r"FGSM", r"adversarial", r"continuous", r"mask",
    r"RobustScaler", r"leakage", r"pre-split", r"split before",
    r"PCA", r"benchmark", r"inferiority",
    r"kci", r"fisherz", r"PC", r"causal",
    r"MAPIE", r"conformal", r"coverage", r"calibration", r"calib",
    r"TabPFN", r"dilution", r"fallback",
    r"transductive", r"GNN", r"MLP"
]

print("Matches in ML_dating_app_behaviour V8_patched_v4.ipynb:")
for term in search_terms:
    pattern = re.compile(term, re.IGNORECASE)
    count = 0
    print(f"\nSearching for '{term}':")
    for cell_idx, cell_type, line_idx, line in code_lines:
        if pattern.search(line):
            print(f"  Cell {cell_idx} ({cell_type}) L{line_idx}: {line.strip()[:120]}")
            count += 1
            if count >= 8:
                print("  ... too many matches, truncated")
                break
