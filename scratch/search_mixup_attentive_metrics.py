import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"
outputs_path = os.path.join(root_dir, "scratch", "v8_actual_outputs.txt")

with open(outputs_path, "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for "Label Smoothing" or "Mixup" or "Attentive" in outputs
search_terms = ["Label Smoothing & Mixup", "Attentive Tabular Network", "Soft-Mask Attention"]

for term in search_terms:
    print(f"\nSearching for '{term}':")
    idx = 0
    count = 0
    while True:
        idx = text.find(term, idx)
        if idx == -1:
            break
        print(f"  --- Found '{term}' at char {idx} ---")
        start = max(0, idx - 100)
        end = min(len(text), idx + 1000)
        print(text[start:end])
        idx += len(term)
        count += 1
        if count >= 3:
            print("  ... too many matches, truncated")
            break
