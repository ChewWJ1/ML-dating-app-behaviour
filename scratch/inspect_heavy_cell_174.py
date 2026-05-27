import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb.get('cells', [])[174]
print("Cell Index: 174")
print("--- CELL CONTENT ---")
print("".join(cell.get('source', [])))
print("--------------------")
