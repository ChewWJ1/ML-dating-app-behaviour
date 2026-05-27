import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("SUCCESS: Notebook JSON is 100% valid and parsed successfully!")
    print(f"Total cells: {len(nb.get('cells', []))}")
except Exception as e:
    print(f"ERROR parsing notebook JSON: {e}")
