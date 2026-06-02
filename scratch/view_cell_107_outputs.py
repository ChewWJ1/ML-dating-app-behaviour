import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

# Search for Cell 107 Code Output
idx = text.find("[Cell 107 Code Output]")
if idx != -1:
    print(text[idx:idx+1000])
else:
    print("Cell 107 Code Output not found!")
