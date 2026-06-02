import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"

def search_text(filename, keyword):
    path = os.path.join(root_dir, filename)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    idx = 0
    while True:
        idx = content.find(keyword, idx)
        if idx == -1:
            break
        print(f"\n--- Found '{keyword}' in {filename} at char {idx} ---")
        start = max(0, idx - 100)
        end = min(len(content), idx + 200)
        print(content[start:end].replace("\n", "  "))
        idx += len(keyword)

search_text("model_results.txt", "0.5684")
search_text("model_results.txt", "0.3135")
search_text("metrics.txt", "0.5684")
search_text("metrics.txt", "0.3135")
