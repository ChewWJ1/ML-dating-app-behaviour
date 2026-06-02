import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"

notebooks = [
    "ML_dating_app_behaviour V8.ipynb",
    "ML_dating_app_behaviour V8.2.ipynb",
    "ML_dating_app_behaviour V8_patched_v4.ipynb"
]

keywords = {
    "Calibration Threshold Leakage": "X_calib",
    "FGSM Adversarial Attacks continuous": "continuous_mask",
    "DiCE Recourse Mutable Features": "features_to_vary",
    "IPW T-Learner": "propensity",
    "PC Algorithm KCI": "kci",
    "Pre-Split Scaling Leakage": "Deferred to Section",
    "Regularization Loss Curves Mixup": "Mixup",
    "TabPFN Hybrid Evaluation Dilution": "dilution",
    "Causal Scaler Confusion": "scaler.fit_transform",
    "Conformal Prediction Leakage": "SplitConformal",
    "PCA Benchmarking RF": "pca_benchmark",
    "GNN Transductive Impact": "transductive",
    "Opacus Privacy target_epsilon": "target_epsilon"
}

for nb_name in notebooks:
    path = os.path.join(root_dir, "notebooks", nb_name)
    if not os.path.exists(path):
        print(f"{nb_name} does not exist!")
        continue
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\nNotebook: {nb_name}")
    
    # Concatenate all cell sources
    full_text = ""
    for cell in nb.get("cells", []):
        full_text += "".join(cell.get("source", [])) + "\n"
        
    for key, kw in keywords.items():
        found = kw in full_text
        print(f"  - {key} ('{kw}'): {'FOUND' if found else 'NOT FOUND'}")
