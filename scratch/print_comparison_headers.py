import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"

def print_around_comparison(filename):
    path = os.path.join(root_dir, filename)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find Model Comparison Table and print the header and first few lines
    idx = content.find("Model Comparison Table")
    if idx != -1:
        print(f"\n--- Found in {filename} ---")
        lines = content[idx:idx+1500].split("\n")
        for line in lines[:20]:
            print(line)
            
print_around_comparison("metrics.txt")
print_around_comparison("model_results.txt")
print_around_comparison("extracted_outputs.txt")
