import json

def inspect_notebook(path, name):
    print(f"=== Inspecting {name} ===")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb['cells']):
        if 'source' in cell:
            cell_text = "".join(cell['source'])
            if 'classification_coverage_score' in cell_text:
                print(f"Cell Index: {idx}")
                print("Source:")
                print(cell_text)
                print("-" * 40)

inspect_notebook(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V4.ipynb", "V4")
inspect_notebook(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb", "V5")
