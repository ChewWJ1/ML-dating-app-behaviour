import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for "Model Comparison (sorted" in the file
idx = 0
while True:
    idx = text.find("Model Comparison", idx)
    if idx == -1:
        break
    print(f"\n--- Found Model Comparison at char {idx} ---")
    start = max(0, idx - 100)
    end = min(len(text), idx + 2000)
    print(text[start:end])
    idx += len("Model Comparison")
