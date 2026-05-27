import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

indices = [35, 117, 174, 176]
with open('scratch/notebook_markdowns_output.txt', 'w', encoding='utf-8') as out_f:
    for idx in indices:
        out_f.write(f"================== CELL {idx} ==================\n")
        # Look back up to 3 cells to find markdown cells
        for lookback in range(1, 4):
            prev_idx = idx - lookback
            if prev_idx >= 0:
                cell = nb['cells'][prev_idx]
                if cell['cell_type'] == 'markdown':
                    out_f.write(f"Markdown Cell at Index {prev_idx}:\n")
                    out_f.write("".join(cell['source']))
                    out_f.write("\n" + "-" * 30 + "\n")
                    break
print("Done writing markdowns to file.")
