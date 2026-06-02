import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

# Search for tuned in outputs
idx = 0
while True:
    idx = text.find("Tuned", idx)
    if idx == -1:
        break
    # Check if there is some metrics print
    print(f"\n--- Found 'Tuned' at char {idx} ---")
    start = max(0, idx - 100)
    end = min(len(text), idx + 800)
    print(text[start:end])
    idx += len("Tuned")
    if idx > 200000:  # Prevent infinite loop or too many outputs
        break
