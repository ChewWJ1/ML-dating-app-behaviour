import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

keywords = ["GAT", "SCARF", "Opacus", "TabPFN", "Mixup", "Attentive Selection"]

for kw in keywords:
    print(f"\nSearching for '{kw}' in outputs:")
    idx = 0
    count = 0
    while True:
        idx = text.find(kw, idx)
        if idx == -1:
            break
        print(f"  --- Found '{kw}' at char {idx} ---")
        start = max(0, idx - 100)
        end = min(len(text), idx + 400)
        print(text[start:end].replace("\n", "  "))
        idx += len(kw)
        count += 1
        if count >= 3:
            print("  ... too many matches, truncated")
            break
