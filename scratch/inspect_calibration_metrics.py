import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8_patched_v4.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", ""))
    if "CalibratedClassifierCV" in src or "Brier Score" in src or "brier" in src:
        print(f"\n--- Cell {idx} ---")
        print("Source:")
        print(src[:400])
        print("Outputs:")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                print("".join(out.get("text", [])))
            elif out.get("output_type") == "execute_result":
                print(out.get("data", {}).get("text/plain", ""))
