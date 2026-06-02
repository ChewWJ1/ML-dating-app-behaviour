import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

idx = text.find("[Cell 121 Code Output]")
if idx != -1:
    print(text[idx:idx+1500])
else:
    print("Cell 121 Code Output not found!")
