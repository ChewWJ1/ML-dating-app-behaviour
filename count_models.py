import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')

with open(r'notebooks\ML_dating_app_behaviour V8.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CLASSIFIER DICT SEARCH ===")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    # Find cell with the main classifiers dict
    if all(x in src for x in ["'KNN'", "'SVM'", "'LightGBM'", "classifiers"]):
        print(f"Cell {i} — classifiers dict:")
        # Extract quoted keys that look like model names
        lines = src.split('\n')
        model_lines = [l for l in lines if re.match(r"\s+'[A-Z].*'\s*:", l)]
        print(f"  Count: {len(model_lines)}")
        for l in model_lines:
            print(f"  {l.strip()}")
        print()

print("=== WHAT GOES INTO results{} ===")
# Find cells that add to results
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if "results['" in src and ("'TabPFN'" in src or "TabPFN" in src):
        print(f"Cell {i}: TabPFN result key:")
        for l in src.split('\n'):
            if "results['" in l:
                print(f"  {l.strip()}")

print("\n=== PRINTED OUTPUTS WITH MODEL LISTS ===")
for i, cell in enumerate(nb['cells']):
    for out in cell.get('outputs', []):
        text = ''.join(out.get('text', out.get('data', {}).get('text/plain', [])))
        if 'Logistic Regression' in text and ('LightGBM' in text or 'CatBoost' in text):
            # Count how many known model names appear
            known = ['Logistic Regression','KNN','Decision Tree','Random Forest',
                     'XGBoost','SVM','LightGBM','CatBoost','Balanced Random Forest',
                     'KNN (Cosine','FT-Transformer','SAINT','NODE','TabPFN','GNN','SCARF']
            found = [m for m in known if m in text]
            if len(found) >= 5:
                print(f"Cell {i}: {len(found)} models in output: {found}")
                print(f"  First 400 chars: {text[:400]}")
                print()
