import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"

def search_text_file(filename, keywords):
    path = os.path.join(root_dir, filename)
    print(f"\nSearching file: {filename}")
    if not os.path.exists(path):
        print("Does not exist!")
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    print(f"Total lines: {len(lines)}")
    
    # Print lines containing any of the keywords, and a few lines after them
    for idx, line in enumerate(lines):
        if any(kw in line for kw in keywords):
            print(f"\n--- Line {idx} ---")
            for j in range(max(0, idx-2), min(len(lines), idx+8)):
                print(f"  {j}: {lines[j]}")
            print("-" * 30)

search_text_file("metrics.txt", ["Model Comparison Table", "Classifier Model", "LightGBM", "CatBoost"])
search_text_file("model_results.txt", ["Model Comparison Table", "Classifier Model"])
