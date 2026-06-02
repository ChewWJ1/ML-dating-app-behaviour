import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

# Find the start of tuning output
idx = 0
while True:
    idx = text.find("Tuning: ", idx)
    if idx == -1:
        break
    print(f"\n--- Found Tuning Section at char {idx} ---")
    start = max(0, idx - 10)
    end = min(len(text), idx + 1000)
    print(text[start:end])
    idx += len("Tuning: ")
