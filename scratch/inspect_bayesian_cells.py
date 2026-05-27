import json

def inspect_notebook(path, name):
    print(f"=== Inspecting {name} ===")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb['cells']):
        if 'source' in cell:
            cell_text = "".join(cell['source'])
            if 'mc_dropout_predict' in cell_text:
                print(f"Cell Index: {idx}")
                # print previous 2 cells
                for p_idx in range(max(0, idx - 2), idx):
                    p_cell = nb['cells'][p_idx]
                    p_text = "".join(p_cell.get('source', []))
                    print(f"[{p_cell['cell_type'].upper()}] Cell {p_idx}:")
                    print(p_text[:300] + ("..." if len(p_text) > 300 else ""))
                print("-" * 20)
                print(f"[CODE] Cell {idx} (target):")
                print(cell_text)
                print("=" * 40)

inspect_notebook(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb", "V5")
