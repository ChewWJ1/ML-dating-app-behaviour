import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for "LightGBM (Tuned)" and look a few lines above it
idx = text.find("LightGBM (Tuned)")
if idx != -1:
    print("--- Found final model ranking around LightGBM (Tuned) ---")
    start = max(0, idx - 500)
    end = min(len(text), idx + 200)
    print(text[start:end])
else:
    print("LightGBM (Tuned) not found in outputs!")
