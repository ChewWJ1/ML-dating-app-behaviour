import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour"

def read_nb(filename):
    path = os.path.join(root_dir, "notebooks", filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

nb_patched = read_nb("ML_dating_app_behaviour V8_patched_v4.ipynb")
nb_82 = read_nb("ML_dating_app_behaviour V8.2.ipynb")

print("Patched cells:", len(nb_patched["cells"]))
print("8.2 cells:", len(nb_82["cells"]))

# Let's check for differences in code cells
diff_count = 0
for idx, (c_p, c_82) in enumerate(zip(nb_patched["cells"], nb_82["cells"])):
    type_p = c_p.get("cell_type")
    type_82 = c_82.get("cell_type")
    if type_p != type_82:
        print(f"Cell {idx} type mismatch: Patched={type_p}, 8.2={type_82}")
        diff_count += 1
        continue
    
    src_p = "".join(c_p.get("source", [])).strip()
    src_82 = "".join(c_82.get("source", [])).strip()
    
    if src_p != src_82:
        print(f"Cell {idx} ({type_p}) content mismatch! (first 100 chars):")
        print(f"  Patched: {src_p[:100].replace(chr(10), ' ')}")
        print(f"  8.2:     {src_82[:100].replace(chr(10), ' ')}")
        diff_count += 1
        if diff_count > 10:
            print("Too many differences, stopping print...")
            break
